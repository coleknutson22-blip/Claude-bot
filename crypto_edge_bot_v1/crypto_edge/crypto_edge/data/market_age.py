"""Market age, established independently of indicator history.

WHY THIS IS A SEPARATE MODULE
-----------------------------
These are two different questions that were previously answered with one
number, and conflating them made the whole universe unusable on Kraken:

  A. "Do we have enough candles to compute the indicators?"  -- a BAR COUNT
  B. "Has this market existed long enough to be worth trading?" -- a CALENDAR
     SPAN, a property of the asset, not of our request

The old code answered B by measuring the span of the 1h window it had fetched.
Venues cap OHLCV responses (Kraken at 720 candles), so at 1h that span is 30
days no matter how old the market is -- and a ten-year-old market was rejected
as "too new", permanently.

The fix is that the cap is PER TIMEFRAME. At 1d, 720 Kraken candles span 720
days; at 1w, over thirteen years. So the venue can answer B perfectly well when
asked at the right resolution. The indicators keep asking at 1h, where they need
density; age asks at 1d, where it needs reach.

SOURCES, IN PRIORITY ORDER
--------------------------
  1. EXCHANGE_METADATA -- the venue's own listing timestamp (ccxt `created`).
     Authoritative where offered; binance provides it, kraken does not.
  2. COARSE_OHLCV      -- oldest bar at a coarse timeframe from this venue.
     Venue-native, deterministic, and reproducible from a timestamp.
  3. CACHED_OBSERVATION -- the earliest bar we have EVER seen for this symbol,
     persisted. Only ever extends backwards, so knowledge accumulates and never
     silently regresses if a venue trims history later.
  4. ASSET_HINT       -- the broad-universe provider's asset history (CoinGecko
     all-time-low/high dates bound the asset's existence from below). This is
     ASSET age, not listing age on this venue, so it ranks last and is labelled.

If none of them can answer, the age is UNKNOWN and the caller must fail closed.
An unverifiable market is not a young market, but it is not a tradable one
either -- and the two are reported differently so the operator can tell.

Every answer carries its source and the timestamp it was established at, so a
universe decision made on any past day can be reconstructed exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..logging_setup import log_event
from ..timeutils import now_ms, tf_ms
from .feed import DataUnavailable

EXCHANGE_METADATA = "exchange_metadata"
COARSE_OHLCV = "coarse_ohlcv"
CACHED_OBSERVATION = "cached_observation"
ASSET_HINT = "asset_hint"
UNKNOWN = "unknown"

# Sources that describe THIS venue's listing rather than the asset in general.
VENUE_SOURCES = (EXCHANGE_METADATA, COARSE_OHLCV, CACHED_OBSERVATION)


@dataclass
class AgeVerdict:
    """What we know about a market's age, and how we came to know it."""
    symbol: str
    age_days: float | None          # None == could not be established
    source: str
    first_ms: int = 0               # earliest moment we can evidence
    observed_ms: int = 0            # when this was established
    detail: str = ""

    @property
    def known(self) -> bool:
        return self.age_days is not None

    def meets(self, min_days: float) -> bool:
        return self.known and self.age_days >= min_days

    def reason_if_blocked(self, min_days: float) -> str:
        """Empty when the market passes; otherwise why it does not."""
        if self.meets(min_days):
            return ""
        if not self.known:
            return (f"market age could not be established ({self.detail or 'no source'}); "
                    f"failing closed")
        return (f"market too new ({self.age_days:.0f}d < {min_days:.0f}d, "
                f"via {self.source})")

    def as_dict(self) -> dict:
        return {"symbol": self.symbol, "age_days": self.age_days,
                "source": self.source, "first_ms": self.first_ms,
                "observed_ms": self.observed_ms, "detail": self.detail}


class MarketAgeService:
    """Establishes and remembers how old each market is."""

    def __init__(self, repo, *, probe_timeframe: str = "1d",
                 probe_bars: int = 400, cache_hours: int = 24) -> None:
        self.repo = repo
        self.probe_timeframe = probe_timeframe
        self.probe_bars = probe_bars
        self.cache_ms = max(0, cache_hours) * 3_600_000

    # ------------------------------------------------------------ resolution
    def age_of(self, symbol: str, *, meta=None, feed=None, asset=None,
               now: int | None = None, allow_fetch: bool = True) -> AgeVerdict:
        now = now_ms() if now is None else now

        # ---- 1. the venue's own listing timestamp ------------------------
        created = int(getattr(meta, "created_ms", 0) or 0)
        if created > 0 and created < now:
            return self._remember(symbol, created, EXCHANGE_METADATA, now,
                                  "venue-reported listing time")

        # ---- 3(a). a cached answer that is still fresh -------------------
        cached = self._cached(symbol)
        if cached is not None and self.cache_ms and \
                now - int(cached["observed_ms"]) < self.cache_ms:
            return self._verdict(symbol, int(cached["first_ms"]),
                                 str(cached["source"]), int(cached["observed_ms"]),
                                 now, str(cached["detail"] or ""))

        # ---- 2. ask the venue at a COARSE timeframe ----------------------
        if allow_fetch and feed is not None:
            first_ms, detail = self._probe(symbol, feed)
            if first_ms:
                return self._remember(symbol, first_ms, COARSE_OHLCV, now, detail)

        # ---- 3(b). fall back to whatever we recorded previously ----------
        if cached is not None and int(cached["first_ms"]) > 0:
            return self._verdict(symbol, int(cached["first_ms"]),
                                 CACHED_OBSERVATION, int(cached["observed_ms"]),
                                 now, "reused stored observation")

        # ---- 4. asset-level hint from the broad-universe provider --------
        hint = int(getattr(asset, "first_known_ms", 0) or 0) if asset else 0
        if hint > 0 and hint < now:
            return self._remember(symbol, hint, ASSET_HINT, now,
                                  "provider asset history (asset age, not listing age)")

        return AgeVerdict(symbol, None, UNKNOWN, 0, now,
                          "no venue metadata, no coarse history, nothing cached, "
                          "no provider hint")

    # ---------------------------------------------------------------- probe
    def _probe(self, symbol: str, feed) -> tuple[int, str]:
        """Oldest bar at the coarse timeframe. 720 daily bars is ~2 years."""
        try:
            series = feed.fetch_ohlcv(symbol, self.probe_timeframe, self.probe_bars)
        except DataUnavailable as e:
            log_event("data", "DEBUG", "age probe failed", symbol=symbol,
                      timeframe=self.probe_timeframe, error=str(e))
            return 0, f"{self.probe_timeframe} probe failed: {e}"
        except Exception as e:                        # never break the caller
            log_event("data", "WARNING", "age probe raised", symbol=symbol,
                      error=str(e))
            return 0, f"{self.probe_timeframe} probe raised: {e}"
        if series is None or len(series) == 0:
            return 0, f"no {self.probe_timeframe} candles"
        first = int(series.open_ms[0])
        return first, (f"oldest of {len(series)} {self.probe_timeframe} bars"
                       + ("" if len(series) >= self.probe_bars
                          else " (venue supplied all it has)"))

    # ---------------------------------------------------------------- store
    def _cached(self, symbol: str) -> dict | None:
        try:
            return self.repo.get_market_age(symbol)
        except Exception:
            return None

    def _remember(self, symbol: str, first_ms: int, source: str, now: int,
                  detail: str) -> AgeVerdict:
        """Persist, keeping the EARLIEST evidence ever seen.

        Monotonic on purpose: a venue that trims its history later must not make
        a market appear to get younger, which would silently drop an asset that
        had already been verified.
        """
        try:
            first_ms = self.repo.record_market_age(symbol, first_ms, source,
                                                   now, detail)
        except Exception as e:
            log_event("data", "WARNING", "could not persist market age",
                      symbol=symbol, error=str(e))
        return self._verdict(symbol, first_ms, source, now, now, detail)

    @staticmethod
    def _verdict(symbol: str, first_ms: int, source: str, observed_ms: int,
                 now: int, detail: str) -> AgeVerdict:
        if not first_ms or first_ms <= 0 or first_ms > now:
            return AgeVerdict(symbol, None, UNKNOWN, 0, observed_ms,
                              detail or "implausible first timestamp")
        return AgeVerdict(symbol, (now - first_ms) / 86_400_000.0, source,
                          first_ms, observed_ms, detail)


def probe_bars_for(timeframe: str, min_days: float, margin: float = 1.25) -> int:
    """Bars of `timeframe` needed to evidence `min_days`, with headroom."""
    step = tf_ms(timeframe)
    if step <= 0:
        return 1
    return max(2, int(min_days * 86_400_000 / step * margin) + 2)
