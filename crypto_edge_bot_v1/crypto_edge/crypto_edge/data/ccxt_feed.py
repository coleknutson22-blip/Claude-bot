"""Live exchange data via CCXT, public endpoints only.

NO API KEY IS EVER READ OR PASSED HERE. The client is constructed without
credentials, which means it is structurally incapable of placing an order even
if some future bug tried to.

TWO THINGS THIS ADAPTER GETS RIGHT THAT ARE EASY TO GET WRONG
-------------------------------------------------------------
1. **Precision.** CCXT reports market granularity in one of three modes.
   `DECIMAL_PLACES` gives an integer count; `TICK_SIZE` gives an absolute tick
   as a float (0.001, and also non-decimal ticks like 0.05); `SIGNIFICANT_DIGITS`
   gives a digit count. Every major venue in current CCXT uses TICK_SIZE, so
   code that only accepts an `int` silently falls back to a default and sizes
   orders finer than the exchange permits. `_precision_fields()` handles all
   three and reports both a tick and a decimal-place fallback.

2. **Ticker timestamps.** Many venues do not stamp their ticker, and CCXT
   normalises that to `None`. Substituting local time silently would make a
   stale quote look perfectly fresh to the entry age check. Instead the
   substitution is recorded on the Quote as `ts_source`, so the rest of the
   system knows whether "quote age" measures the venue's freshness or merely
   our own clock.

STATUS: REQUIRES VERIFICATION ON YOUR MACHINE. The build environment has no
outbound network (egress policy denies exchange hosts), so this class has been
written, import-checked and unit-tested against recorded CCXT response shapes,
but its network calls have NOT been executed against a live venue.
Run `python -m crypto_edge.cli verify-live` to exercise it for real.
"""
from __future__ import annotations

import math
import time

from ..logging_setup import log_event
from ..models import TS_LOCAL, TS_VENUE, MarketMeta, Quote, Series
from ..timeutils import last_closed_open_ms, now_ms, tf_ms
from .feed import DataUnavailable

# CCXT precision-mode constants, restated so this module can interpret market
# metadata without importing ccxt at module import time.
DECIMAL_PLACES = 2
SIGNIFICANT_DIGITS = 3
TICK_SIZE = 4


def decimals_from_tick(tick: float) -> int:
    """Decimal places implied by a tick size (0.001 -> 3). Ticks that are not a
    power of ten (0.05) round UP to the finer count; the tick itself is what
    actually gets enforced, this is only the fallback."""
    if not tick or tick <= 0 or not math.isfinite(tick):
        return 8
    text = f"{tick:.12f}".rstrip("0")
    if "." not in text:
        return 0
    return min(12, len(text.split(".", 1)[1]))


def _precision_fields(raw, mode: int) -> tuple[int, float]:
    """Translate one CCXT precision value into (decimal_places, tick_size).

    Returns tick_size 0.0 when the venue expresses precision as a count rather
    than an absolute tick.
    """
    if raw is None:
        return 8, 0.0
    if mode == TICK_SIZE:
        try:
            tick = float(raw)
        except (TypeError, ValueError):
            return 8, 0.0
        if tick <= 0 or not math.isfinite(tick):
            return 8, 0.0
        return decimals_from_tick(tick), tick
    # DECIMAL_PLACES, SIGNIFICANT_DIGITS, or an exchange that reports a plain
    # integer regardless of the declared mode.
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 8, 0.0
    if not math.isfinite(val) or val < 0:
        return 8, 0.0
    if 0 < val < 1:
        # A fractional value cannot be a COUNT of decimal places, so this is a
        # tick reported under the wrong mode. Reading it as a count would be
        # worse than the bug this function exists to fix: int(0.001) is 0, i.e.
        # "round every quantity to a whole unit".
        return decimals_from_tick(val), val
    return min(int(val), 12), 0.0


class CCXTFeed:
    def __init__(self, exchange: str = "binance", quote: str = "USDT",
                 rate_limit_ms: int = 250, timeout_s: int = 20,
                 quote_ts_fallback: str = "local",
                 page_limit: int = 300, close_buffer_ms: int = 0,
                 cache_bars: int = 0) -> None:
        """`quote_ts_fallback` controls what happens when a venue ticker has no
        timestamp: "local" stamps it with receive time and flags it as such,
        "reject" discards the quote entirely (fails closed for entries)."""
        try:
            import ccxt  # imported lazily so the core runs without the dependency
        except ImportError as e:
            raise DataUnavailable(
                "ccxt is not installed -- run: pip install -r requirements.txt") from e
        if not hasattr(ccxt, exchange):
            raise DataUnavailable(f"unknown exchange '{exchange}'")
        self.name = exchange
        self.quote = quote
        self.rate_limit_ms = rate_limit_ms
        self.quote_ts_fallback = quote_ts_fallback
        # Bars per REQUEST (a paging hint), not the total history we want.
        self.page_limit = page_limit
        # No apiKey / secret. Public data only. This is deliberate.
        self.client = getattr(ccxt, exchange)({
            "enableRateLimit": True,
            "timeout": timeout_s * 1000,
            "options": {"defaultType": "spot"},
        })
        self.precision_mode = getattr(self.client, "precisionMode", DECIMAL_PLACES)
        self._markets: dict[str, MarketMeta] = {}
        # observability for the quote-timestamp policy (see cli verify-live)
        self.quote_ts_venue = 0
        self.quote_ts_local = 0
        # closed-candle cache (see fetch_ohlcv)
        self.close_buffer_ms = int(close_buffer_ms)
        self.cache_bars = int(cache_bars)
        self._ohlcv_cache: dict[tuple, list] = {}
        self.cache_hits = 0
        self.cache_refreshes = 0
        self.cache_bootstraps = 0

    def _sleep(self) -> None:
        time.sleep(self.rate_limit_ms / 1000.0)

    # --------------------------------------------------------- candle cache
    def _cache(self) -> dict:
        cache = getattr(self, "_ohlcv_cache", None)
        if cache is None:
            cache = self._ohlcv_cache = {}
        return cache

    def _closed_rows(self, rows: list, step: int, now: int) -> list:
        """Rows whose candle has genuinely finished, buffer included."""
        buf = int(getattr(self, "close_buffer_ms", 0) or 0)
        return [r for r in rows if r[0] + step + buf <= now]

    def _fetch_recent_page(self, symbol: str, timeframe: str) -> list:
        """ONE request for the venue's most recent window.

        No `since`, deliberately: CCXT's `filter_by_since_limit` only slices
        oldest-first when `since` is given, so a since-less request comes back
        as `array[-limit:]` -- the NEWEST rows -- on every venue.
        """
        page = max(2, int(self.page_limit))
        try:
            batch = self.client.fetch_ohlcv(symbol, timeframe=timeframe,
                                            since=None, limit=page)
        except Exception as e:
            raise DataUnavailable(
                f"fetch_ohlcv {symbol} {timeframe} refresh failed: {e}") from e
        rows = sorted((r for r in batch if _row_ok(r)), key=lambda r: r[0])
        return self._contiguous_tail(rows, tf_ms(timeframe))

    def _store_candles(self, key: tuple, closed_rows: list) -> None:
        """Keep CLOSED bars only, capped, so the cache cannot hold a live bar.

        The candle currently in progress changes on every tick. Storing it and
        serving it later is precisely the "old data masquerading as current"
        failure, so it never enters the cache -- it is passed straight through
        from the request that fetched it and then forgotten.
        """
        if not closed_rows:
            return
        cap = max(1, int(getattr(self, "cache_bars", 0) or 0))
        self._cache()[key] = closed_rows[-cap:]

    def ohlcv_cache_stats(self) -> dict:
        return {"series": len(self._cache()),
                "hits": getattr(self, "cache_hits", 0),
                "refreshes": getattr(self, "cache_refreshes", 0),
                "bootstraps": getattr(self, "cache_bootstraps", 0)}

    # ------------------------------------------------------------- markets
    def load_markets(self) -> dict[str, MarketMeta]:
        try:
            raw = self.client.load_markets()
        except Exception as e:
            raise DataUnavailable(f"load_markets failed: {e}") from e
        mode = getattr(self.client, "precisionMode", self.precision_mode)
        self.precision_mode = mode
        out: dict[str, MarketMeta] = {}
        for sym, m in raw.items():
            if not m.get("spot", False):
                continue
            if m.get("quote") != self.quote:
                continue
            precision = m.get("precision") or {}
            limits = m.get("limits") or {}
            amount_dp, amount_step = _precision_fields(precision.get("amount"), mode)
            price_dp, price_step = _precision_fields(precision.get("price"), mode)
            out[sym] = MarketMeta(
                symbol=sym, base=m.get("base", ""), quote=m.get("quote", ""),
                active=bool(m.get("active", True)),
                amount_precision=amount_dp, price_precision=price_dp,
                min_amount=_limit(limits, "amount"),
                min_cost=_limit(limits, "cost"),
                amount_step=amount_step, price_step=price_step,
                created_ms=_ts(m.get("created")),
            )
        if not out:
            raise DataUnavailable(
                f"no active spot {self.quote} markets on {self.name} -- check the "
                f"exchange name and quote currency")
        self._markets = out
        return out

    def fetch_tickers(self) -> dict[str, dict]:
        if not self.client.has.get("fetchTickers"):
            raise DataUnavailable(
                f"{self.name} does not support fetchTickers; the universe "
                f"liquidity screen needs it")
        try:
            return self.client.fetch_tickers()
        except Exception as e:
            raise DataUnavailable(f"fetch_tickers failed: {e}") from e

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> Series:
        """Return up to `limit` candles, PAGING if the venue caps a request.

        Venues do not agree on how much history one request may return, and some
        ignore `limit` entirely: Kraken's OHLC endpoint disregards it and caps
        every response at 720 candles. Treating one response as "all the history
        there is" makes a decade-old market look days old, which the universe
        filters then read as a quality signal rather than the fetch limitation
        it actually is. So we page backwards with `since` until we have what was
        asked for or the venue genuinely runs out.
        """
        want = max(1, int(limit))
        step = tf_ms(timeframe)
        now = now_ms()
        key = (symbol, timeframe)
        cache = self._cache()
        cached = cache.get(key)
        enabled = int(getattr(self, "cache_bars", 0) or 0) > 0

        # The newest candle that COULD have finished by now. This, not a
        # time-to-live, is what decides whether the cache is current: a closed
        # candle is immutable, so while no newer one exists the cached series
        # IS the present state of the market and re-downloading it would return
        # byte-identical data.
        newest_possible = last_closed_open_ms(
            timeframe, now, int(getattr(self, "close_buffer_ms", 0) or 0))

        clean: list | None = None
        if enabled and cached and len(cached) >= want:
            if cached[-1][0] >= newest_possible:
                self.cache_hits += 1
                clean = cached[-want:]
            else:
                # A candle has closed since we last looked: one request for the
                # newest page, merged onto what we hold. If the fresh page does
                # not JOIN the cache the merge is abandoned rather than spliced
                # across the hole, and we bootstrap from scratch instead.
                fresh = self._fetch_recent_page(symbol, timeframe)
                merged = {r[0]: r for r in cached}
                merged.update({r[0]: r for r in fresh})
                joined = self._contiguous_tail(
                    [merged[k] for k in sorted(merged)], step)
                closed = self._closed_rows(joined, step, now)
                if len(closed) >= want:
                    self.cache_refreshes += 1
                    self._store_candles(key, closed)
                    live = [r for r in joined if r[0] > closed[-1][0]]
                    clean = closed[-want:] + live[:1]

        if clean is None:
            if enabled:
                self.cache_bootstraps += 1
            clean = self._fetch_ohlcv_paged(symbol, timeframe, want)
            if enabled and clean:
                self._store_candles(
                    key, self._closed_rows(clean, step, now))

        if not clean:
            raise DataUnavailable(f"empty OHLCV for {symbol} {timeframe}")
        try:
            return Series.from_ohlcv(symbol, timeframe, [list(r[:6]) for r in clean])
        except (TypeError, ValueError) as e:
            raise DataUnavailable(
                f"unparseable OHLCV for {symbol} {timeframe}: {e}") from e

    @staticmethod
    def _contiguous_tail(rows: list, step: int) -> list:
        """The longest run of evenly spaced bars ending at the NEWEST one.

        A page that comes back truncated leaves a hole in the middle of the
        history. Splicing across that hole would hand the indicators a series
        whose bars are not the interval they claim to be, so the hole is cut
        away instead and only the unbroken recent run is kept. Short but honest
        beats long but wrong, and the depth check downstream sees the shortfall.
        """
        if len(rows) < 2:
            return rows
        cut = 0
        for i in range(len(rows) - 1, 0, -1):
            if rows[i][0] - rows[i - 1][0] != step:
                cut = i
                break
        return rows[cut:]

    def _fetch_ohlcv_paged(self, symbol: str, timeframe: str, want: int,
                           since_ms: int | None = None) -> list:
        """Accumulate `want` closed candles plus the in-progress one, oldest-first.

        WHY THE FIRST REQUEST CARRIES NO `since`
        ----------------------------------------
        CCXT truncates from the WRONG END when `since` and `limit` are both
        given. `Exchange.filter_by_since_limit` sets
        `shouldFilterFromStart = not tail and sinceIsDefined`, and
        `filter_by_limit` then returns `array[0:limit]` -- the OLDEST `limit`
        rows of the response, discarding the newest. With `since` absent it
        returns `array[-limit:]` instead: the newest.

        The previous implementation reached back `(want + 2)` bars "for safety"
        and passed `limit=page`. On Kraken, which returns every bar from `since`
        to the present, that produced `want + 2` rows which CCXT cut to the
        oldest `want`. The two rows discarded were the in-progress candle AND
        THE MOST RECENTLY CLOSED ONE, so the freshest candle the strategy could
        ever see was a full bar behind reality -- 118 minutes stale on a 1h
        series, 298 on a 4h series, with no error raised anywhere. The safety
        margin was taken off the end that mattered.

        Binance hid it: it honours `limit` server-side and returns exactly
        `limit` rows, so the slice was a no-op there.

        So: the newest page is fetched FIRST and WITHOUT `since`, which every
        venue answers with its most recent bars. Older history is then walked
        backwards in spans no wider than the venue will actually return, and
        `_contiguous_tail` guarantees the result has no holes.
        """
        step = tf_ms(timeframe)
        want = max(1, int(want))
        page = max(2, int(self.page_limit))
        now = now_ms()
        # Two extra bars. One covers the candle currently in progress, which is
        # fetched and then discarded by the caller's drop_unclosed(). The second
        # covers the close BUFFER: for the ~20s after a candle closes it is not
        # yet trusted, and without the margin the usable depth dips one bar
        # below the requirement on every single bar boundary -- briefly
        # un-tradable for a reason that has nothing to do with the market.
        need = want + 2
        oldest_wanted = (since_ms if since_ms is not None
                         else now - need * step)

        by_open: dict[int, list] = {}
        pages = 0
        dropped_malformed = 0
        max_pages = max(1, math.ceil(need / (page - 1)) + 3)
        # What the venue will actually put in one response. Starts optimistic
        # and is lowered to whatever the venue demonstrates it will return, so
        # later spans never ask for more than fits and get head-truncated.
        capacity = page
        anchor: int | None = None          # None == "the present"

        while pages < max_pages:
            pages += 1
            since = None
            if anchor is not None:
                # A span of at most `capacity` bars ending at `anchor`, so
                # CCXT's oldest-first slice has nothing to discard.
                since = max(anchor - (capacity - 1) * step, oldest_wanted - step)
            try:
                batch = self.client.fetch_ohlcv(symbol, timeframe=timeframe,
                                                since=since, limit=page)
            except Exception as e:
                if by_open:
                    log_event("data", "WARNING",
                              "OHLCV pagination stopped early", symbol=symbol,
                              timeframe=timeframe, have=len(by_open), error=str(e))
                    break
                raise DataUnavailable(
                    f"fetch_ohlcv {symbol} {timeframe} failed: {e}") from e
            if not batch:
                break
            if len(batch) < capacity:
                # The venue caps responses lower than we assumed. Believe it.
                capacity = max(2, len(batch))

            before = len(by_open)
            for row in batch:
                if _row_ok(row):
                    by_open[int(row[0])] = row
                else:
                    dropped_malformed += 1
            # No NEW bars means the venue has nothing further -- it has hit its
            # own history cap. Stop rather than spin: this is the guard that
            # makes the loop always terminate.
            if len(by_open) == before:
                break

            usable = self._contiguous_tail([by_open[k] for k in sorted(by_open)], step)
            if usable[0][0] <= oldest_wanted or len(usable) >= need:
                break
            anchor = usable[0][0] - step
            self._sleep()

        if dropped_malformed:
            log_event("data", "WARNING", "dropped malformed OHLCV rows",
                      symbol=symbol, timeframe=timeframe,
                      dropped=dropped_malformed, kept=len(by_open))
        # Malformed rows are removed BEFORE the contiguity cut, so a single bad
        # row is treated as the hole it really is rather than being spliced over.
        ordered = self._contiguous_tail(
            [by_open[k] for k in sorted(by_open)], step)
        if len(ordered) < need:
            log_event("data", "DEBUG", "venue supplied less history than requested",
                      symbol=symbol, timeframe=timeframe,
                      wanted=want, got=len(ordered), pages=pages)
        return ordered[-need:]

    def fetch_quote(self, symbol: str) -> Quote | None:
        try:
            t = self.client.fetch_ticker(symbol)
        except Exception as e:
            log_event("data", "WARNING", "fetch_ticker failed",
                      symbol=symbol, error=str(e))
            return None
        bid = _num(t.get("bid"))
        ask = _num(t.get("ask"))
        last = _num(t.get("last")) or _num(t.get("close"))
        if last <= 0 and bid > 0 and ask > 0:
            last = (bid + ask) / 2
        if last <= 0:
            return None

        raw_ts = t.get("timestamp")
        ts, source = _resolve_quote_ts(raw_ts)
        if source == TS_LOCAL:
            self.quote_ts_local += 1
            if self.quote_ts_fallback == "reject":
                log_event("data", "WARNING",
                          "ticker has no venue timestamp; rejecting under policy",
                          symbol=symbol, exchange=self.name)
                return None
            log_event("data", "DEBUG",
                      "ticker has no venue timestamp; stamped locally",
                      symbol=symbol, exchange=self.name)
        else:
            self.quote_ts_venue += 1
        return Quote(symbol, bid, ask, last, ts, ts_source=source)

    def server_time_ms(self) -> int | None:
        """Used by the startup self-check to detect a badly wrong local clock."""
        try:
            if self.client.has.get("fetchTime"):
                return int(self.client.fetch_time())
        except Exception as e:
            log_event("data", "WARNING", "fetch_time failed", error=str(e))
        return None

    # --------------------------------------------------------- diagnostics
    def quote_ts_stats(self) -> dict:
        total = self.quote_ts_venue + self.quote_ts_local
        return {"venue_stamped": self.quote_ts_venue,
                "locally_stamped": self.quote_ts_local,
                "venue_pct": (self.quote_ts_venue / total * 100.0) if total else 0.0}


def _resolve_quote_ts(raw) -> tuple[int, str]:
    """Venue timestamp when it is present and plausible, else local receive time.

    A zero, negative, non-numeric or absurd timestamp is treated as absent
    rather than trusted -- the whole point of the age check is that the number
    it works on means something.
    """
    now = now_ms()
    try:
        ts = int(raw)
    except (TypeError, ValueError):
        return now, TS_LOCAL
    if ts <= 0:
        return now, TS_LOCAL
    # guard against seconds-vs-milliseconds mistakes and nonsense clocks:
    # anything more than a day out in either direction is not a usable stamp.
    if abs(now - ts) > 86_400_000:
        return now, TS_LOCAL
    return ts, TS_VENUE


def _row_ok(row) -> bool:
    """One definition of a usable OHLCV row, used everywhere it matters."""
    return bool(row) and len(row) >= 6 and all(v is not None for v in row[:6])


def _num(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def _ts(raw) -> int:
    """A venue listing timestamp, or 0 when absent or implausible."""
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return 0
    return v if 0 < v <= now_ms() else 0


def _limit(limits: dict, key: str) -> float:
    return _num(((limits.get(key) or {}).get("min")))
