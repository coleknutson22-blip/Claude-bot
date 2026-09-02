"""The two-phase opportunity scan: rank cheaply, then analyse a bounded few.

THE COST PROBLEM THIS SOLVES
----------------------------
Strategy B reads three timeframes. Fetching 5m, 15m and 1h for every market that
survives the liquidity filters would be three series per symbol per cycle -- on
the live Kraken/USD universe that is roughly 40 fetches where Strategy A needs
about 22, and it would undo the caching work that took a measured cycle from
74.6 seconds down to near zero between bar closes.

So the scan is tiered. Ranking runs on the HOURLY series the engine already
holds for Strategy A plus the single venue-wide ticker snapshot, and costs no
additional requests at all. Only the shortlist -- `shortlist_size`, 12 by
default -- is deepened with 5m and 15m.

The consequence worth stating plainly: deep-fetch cost is bounded by
CONFIGURATION, not by how many markets happen to qualify. A quiet universe and a
frantic one cost the same. `ScanResult.deep_fetches` records the actual count so
a test can assert the bound rather than trusting this comment.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .data.feed import DataUnavailable
from .logging_setup import log_event
from .models import Series
from .strategy import ranking
from .strategy.aggressive_momentum import AggressiveMomentumStrategy
from .strategy.base import MarketContext, Signal
from .strategy.features import required_bars


@dataclass
class ScanResult:
    ranked: list = field(default_factory=list)          # every candidate, ordered
    shortlist: list = field(default_factory=list)       # the deep-analysis set
    signals: list[Signal] = field(default_factory=list)  # one per shortlisted symbol
    deep_fetches: int = 0                                # series actually requested
    fetch_failures: dict = field(default_factory=dict)

    @property
    def entries(self) -> list[Signal]:
        return [s for s in self.signals if s.is_entry]

    def by_side(self, side: str) -> list[Signal]:
        return [s for s in self.signals if s.side == side]


def deep_frames(feed, symbol: str, timeframes: list[str], rank_tf: str,
                have: dict[str, Series], now_ms: int, buffer_ms: int,
                counter: list[int]) -> dict[str, Series]:
    """Fetch only the timeframes not already in memory.

    The ranking timeframe is passed in via `have`, so a shortlisted symbol costs
    two requests (5m and 15m) rather than three -- and zero when the candle
    cache is warm and no new bar has closed.
    """
    frames: dict[str, Series] = {}
    for tf in timeframes:
        if tf == rank_tf and symbol in have:
            frames[tf] = have[symbol]
            continue
        want = required_bars().get(tf, 60) + 5
        counter[0] += 1
        s = feed.fetch_ohlcv(symbol, tf, want)
        frames[tf] = s.drop_unclosed(now_ms, buffer_ms)
    return frames


def scan(cfg: Config, feed, ctx: MarketContext, *,
         rank_series: dict[str, Series], btc_1h: Series | None,
         meta_by_symbol: dict[str, dict], now_ms: int,
         buffer_ms: int = 0) -> ScanResult:
    """Rank the universe, deepen the best few, and decide a side for each.

    `rank_series` is the hourly series already held for the candidate symbols --
    supplying it rather than fetching it is what makes the ranking pass free.
    """
    a = cfg.aggressive
    strategy = AggressiveMomentumStrategy(a)
    res = ScanResult()

    res.ranked = ranking.rank(
        rank_series, btc_1h, meta_by_symbol,
        min_dollar_volume=cfg.universe.min_dollar_volume_24h,
        atr_band=(a.rank_atr_band_lo, a.rank_atr_band_hi))
    res.shortlist = ranking.shortlist(res.ranked, a.shortlist_size)

    counter = [0]
    btc_frames: dict[str, Series] | None = None
    if btc_1h is not None:
        btc_frames = {"1h": btc_1h}

    for cand in res.shortlist:
        try:
            frames = deep_frames(feed, cand.symbol, a.timeframes, a.rank_timeframe,
                                 rank_series, now_ms, buffer_ms, counter)
        except DataUnavailable as e:
            res.fetch_failures[cand.symbol] = str(e)
            log_event("data", "WARNING", "deep fetch failed",
                      symbol=cand.symbol, error=str(e))
            continue
        meta = dict(meta_by_symbol.get(cand.symbol, {}))
        meta["symbol"] = cand.symbol
        sig = strategy.evaluate_frames(frames, ctx, btc_frames=btc_frames, meta=meta)
        sig.features["rank"] = cand.rank
        sig.features["rank_score"] = cand.score
        sig.features["rank_components"] = cand.components
        res.signals.append(sig)

    res.deep_fetches = counter[0]
    return res
