"""Direction-agnostic opportunity ranking.

WHAT THIS IS FOR
----------------
Deep analysis costs network requests. Running it over every liquid market each
cycle is what turned an 11-symbol Kraken cycle into 74.6 seconds. Ranking is
cheap -- it reads the hourly series the engine already holds plus the venue
ticker snapshot -- and cuts the field to a bounded shortlist before anything
expensive happens.

DIRECTION-AGNOSTIC, ON PURPOSE
------------------------------
Ranking asks "is something happening here?", not "which way?". A market
collapsing is exactly as interesting to a long/short strategy as one running, so
every movement term uses ABSOLUTE magnitude. Deciding the side is the entry
logic's job, and doing it here would quietly build in a long bias.

DETERMINISTIC, ALSO ON PURPOSE
------------------------------
The bot this strategy is modelled on computed a ranking and then called
`random.shuffle()` on it before picking a candidate, which threw the entire
ranking away -- the effect was a random choice among the top twelve. Ties here
break on symbol name, so the same inputs always produce the same order, and the
tests assert that rather than trusting it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..indicators import atr, last_valid, rel_volume, roc
from ..models import Series
from .features import trend_quality

NAN = float("nan")

# factor -> (weight, why it earns its place)
RANK_WEIGHTS: dict[str, float] = {
    "movement": 0.25,
    "rel_volume": 0.20,
    "trend_quality": 0.18,
    "rel_strength": 0.15,
    "liquidity": 0.12,
    "volatility_fit": 0.10,
}

RANK_JUSTIFICATIONS = {
    "movement": "Absolute 1h and 6h return in ATR units; a momentum strategy "
                "needs something to be moving, either way.",
    "rel_volume": "Volume against its own 24-bar average; an unconfirmed move "
                  "is a thin print, not participation.",
    "trend_quality": "R-squared of the log-price fit; separates a trend from a "
                     "single spike that has already finished.",
    "rel_strength": "Absolute divergence from BTC; idiosyncratic movers beat "
                    "assets simply carried by beta.",
    "liquidity": "24h quote-currency notional and spread; an illiquid fill "
                 "destroys realised edge no matter how good the signal.",
    "volatility_fit": "ATR% inside a usable band; too quiet cannot pay for "
                      "costs, too wild cannot be sized.",
}


@dataclass
class RankedCandidate:
    symbol: str
    score: float
    rank: int = 0
    components: dict[str, float] = field(default_factory=dict)
    inputs: dict[str, float] = field(default_factory=dict)


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(min(hi, max(lo, x)))


def _scale(value: float, lo: float, hi: float) -> float:
    """Linear map [lo, hi] -> [0, 100], clamped."""
    if not np.isfinite(value) or hi <= lo:
        return 0.0
    return _clamp((value - lo) / (hi - lo) * 100.0)


def score_movement(roc_1h: float, roc_6h: float, atr_pct: float) -> float:
    """Absolute movement in ATR units. Direction is deliberately discarded."""
    if not np.isfinite(atr_pct) or atr_pct <= 0:
        return 0.0
    parts = [abs(v) / atr_pct for v in (roc_1h, roc_6h) if np.isfinite(v)]
    if not parts:
        return 0.0
    return _scale(sum(parts) / len(parts), 0.2, 3.0)


def score_rel_volume(rv: float) -> float:
    return _scale(rv, 0.8, 3.0)


def score_trend_quality(r2_1h: float, r2_15m: float) -> float:
    vals = [v for v in (r2_1h, r2_15m) if np.isfinite(v)]
    if not vals:
        return 0.0
    return _scale(sum(vals) / len(vals), 0.05, 0.75)


def score_rel_strength(asset_roc: float, btc_roc: float) -> float:
    if not (np.isfinite(asset_roc) and np.isfinite(btc_roc)):
        return 50.0          # unknown is neutral, never a bonus
    return _scale(abs(asset_roc - btc_roc), 0.0, 8.0)


def score_liquidity(dollar_volume: float, spread_bps: float,
                    min_dollar_volume: float) -> float:
    if dollar_volume <= 0 or min_dollar_volume <= 0:
        return 0.0
    depth = _scale(math.log10(dollar_volume / min_dollar_volume + 1e-9), 0.0, 1.3)
    if not np.isfinite(spread_bps):
        return depth * 0.8   # unmeasured spread is a discount, not a free pass
    tightness = 100.0 - _scale(spread_bps, 2.0, 30.0)
    return _clamp(0.6 * depth + 0.4 * tightness)


def score_volatility_fit(atr_pct: float, lo: float, hi: float) -> float:
    """A band, not a ramp: usable volatility has a floor AND a ceiling."""
    if not np.isfinite(atr_pct) or atr_pct <= 0:
        return 0.0
    if atr_pct < lo:
        return _scale(atr_pct, lo * 0.3, lo)
    if atr_pct > hi:
        return _clamp(100.0 - _scale(atr_pct, hi, hi * 3.0))
    return 100.0


def combine(components: dict[str, float]) -> float:
    total = sum(RANK_WEIGHTS.values())
    return _clamp(sum(components.get(k, 0.0) * w
                      for k, w in RANK_WEIGHTS.items()) / total)


def rank_inputs(series_1h: Series, btc_1h: Series | None,
                meta: dict) -> dict[str, float]:
    """The cheap inputs a ranking needs, from the hourly series alone."""
    close, high, low, vol = (series_1h.close, series_1h.high,
                             series_1h.low, series_1h.volume)
    price = float(close[-1]) if len(close) else NAN
    a = last_valid(atr(high, low, close, 14))
    atr_pct = (a / price * 100.0) if (np.isfinite(a) and price > 0) else NAN
    _, r2_1h = trend_quality(close, 24)
    r1 = last_valid(roc(close, 1))
    r6 = last_valid(roc(close, 6))
    btc_6h = last_valid(roc(btc_1h.close, 6)) if btc_1h is not None and len(btc_1h) > 6 else NAN
    return {
        "price": price,
        "atr_pct": atr_pct,
        "roc_1h": r1,
        "roc_6h": r6,
        "btc_roc_6h": btc_6h,
        "rel_volume": last_valid(rel_volume(vol, 24)) if len(vol) > 25 else NAN,
        "trend_r2_1h": r2_1h,
        "trend_r2_15m": NAN,        # 15m is not fetched yet at ranking time
        "dollar_volume": float(meta.get("dollar_volume", 0.0)),
        "spread_bps": float(meta.get("spread_bps", NAN)),
    }


def score_candidate(inputs: dict[str, float], *, min_dollar_volume: float,
                    atr_band: tuple[float, float]) -> dict[str, float]:
    lo, hi = atr_band
    return {
        "movement": score_movement(inputs.get("roc_1h", NAN),
                                   inputs.get("roc_6h", NAN),
                                   inputs.get("atr_pct", NAN)),
        "rel_volume": score_rel_volume(inputs.get("rel_volume", NAN)),
        "trend_quality": score_trend_quality(inputs.get("trend_r2_1h", NAN),
                                             inputs.get("trend_r2_15m", NAN)),
        "rel_strength": score_rel_strength(inputs.get("roc_6h", NAN),
                                           inputs.get("btc_roc_6h", NAN)),
        "liquidity": score_liquidity(inputs.get("dollar_volume", 0.0),
                                     inputs.get("spread_bps", NAN),
                                     min_dollar_volume),
        "volatility_fit": score_volatility_fit(inputs.get("atr_pct", NAN), lo, hi),
    }


def rank(series_by_symbol: dict[str, Series], btc_1h: Series | None,
         meta_by_symbol: dict[str, dict], *, min_dollar_volume: float,
         atr_band: tuple[float, float] = (0.3, 6.0)) -> list[RankedCandidate]:
    """Order every candidate best-first. Deterministic; never shuffled.

    Ties break on symbol name so the order is a pure function of the inputs --
    two runs over identical data produce an identical list, which is what makes
    a shortlist reproducible and a research record meaningful.
    """
    out: list[RankedCandidate] = []
    for symbol in sorted(series_by_symbol):
        s = series_by_symbol[symbol]
        if s is None or len(s) < 26:
            continue
        inputs = rank_inputs(s, btc_1h, meta_by_symbol.get(symbol, {}))
        comps = score_candidate(inputs, min_dollar_volume=min_dollar_volume,
                                atr_band=atr_band)
        out.append(RankedCandidate(symbol=symbol, score=combine(comps),
                                   components=comps, inputs=inputs))
    out.sort(key=lambda c: (-c.score, c.symbol))
    for i, c in enumerate(out, start=1):
        c.rank = i
    return out


def shortlist(ranked: list[RankedCandidate], n: int) -> list[RankedCandidate]:
    """The best `n`, in rank order. The bound on deep-analysis cost."""
    return list(ranked[:max(0, int(n))])
