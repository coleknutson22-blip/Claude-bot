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
from ..timeutils import now_ms, tf_ms
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
                 page_limit: int = 300) -> None:
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

    def _sleep(self) -> None:
        time.sleep(self.rate_limit_ms / 1000.0)

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
        rows = self._fetch_ohlcv_paged(symbol, timeframe, limit)
        if not rows:
            raise DataUnavailable(f"empty OHLCV for {symbol} {timeframe}")
        clean = [r for r in rows
                 if r and len(r) >= 6 and all(v is not None for v in r[:6])]
        if len(clean) != len(rows):
            log_event("data", "WARNING", "dropped malformed OHLCV rows",
                      symbol=symbol, timeframe=timeframe,
                      dropped=len(rows) - len(clean), kept=len(clean))
        if not clean:
            raise DataUnavailable(
                f"all OHLCV rows for {symbol} {timeframe} were malformed")
        try:
            return Series.from_ohlcv(symbol, timeframe, [list(r[:6]) for r in clean])
        except (TypeError, ValueError) as e:
            raise DataUnavailable(
                f"unparseable OHLCV for {symbol} {timeframe}: {e}") from e

    def _fetch_ohlcv_paged(self, symbol: str, timeframe: str, want: int) -> list:
        """Accumulate `want` candles, oldest-first, across as many requests as
        the venue needs. Returns whatever it managed to get."""
        step = tf_ms(timeframe)
        want = max(1, int(want))
        page = max(1, int(self.page_limit))
        # Reach back a little further than needed: venues round `since` to their
        # own bar boundaries and some drop the first partial bar.
        since = now_ms() - (want + 2) * step
        by_open: dict[int, list] = {}
        pages = 0
        max_pages = max(1, math.ceil(want / page) + 3)

        while pages < max_pages:
            pages += 1
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

            before = len(by_open)
            for row in batch:
                if row and len(row) >= 6 and row[0] is not None:
                    by_open[int(row[0])] = row
            # No NEW bars means the venue has nothing further; stop rather than
            # spin. This is the guard that makes the loop always terminate.
            if len(by_open) == before:
                break
            newest = max(by_open)
            if len(by_open) >= want or newest + step >= now_ms():
                break
            since = newest + step
            self._sleep()

        ordered = [by_open[k] for k in sorted(by_open)]
        if len(ordered) < want:
            log_event("data", "DEBUG", "venue supplied less history than requested",
                      symbol=symbol, timeframe=timeframe,
                      wanted=want, got=len(ordered), pages=pages)
        return ordered[-want:]

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
