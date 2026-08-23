"""Regime and breadth calculations. All inputs must already be closed candles."""
from __future__ import annotations

import numpy as np

from ..indicators import adx, ema, last_valid
from ..models import Series

BULL, WEAK_BULL, NEUTRAL, BEAR, UNKNOWN = "bull", "weak_bull", "neutral", "bear", "unknown"


def trend_regime(series: Series, ema_period: int = 50,
                 slope_lookback: int = 6) -> tuple[str, float]:
    """Classify a series' structure. Returns (label, 0-100 score).

    Deliberately simple and explainable: price vs a medium EMA, plus whether
    that EMA is rising. No curve-fitted thresholds.
    """
    if len(series) < ema_period + slope_lookback + 1:
        return UNKNOWN, 50.0
    e = ema(series.close, ema_period)
    price = float(series.close[-1])
    cur = last_valid(e)
    if not np.isfinite(cur) or cur <= 0:
        return UNKNOWN, 50.0
    prev = e[-1 - slope_lookback]
    if not np.isfinite(prev) or prev <= 0:
        return UNKNOWN, 50.0

    above = price > cur
    slope_pct = (cur - prev) / prev * 100.0
    rising = slope_pct > 0

    if above and rising:
        label = BULL
    elif above and not rising:
        label = WEAK_BULL
    elif not above and rising:
        label = NEUTRAL
    else:
        label = BEAR

    # score: distance above the EMA plus slope, both squashed into 0-100
    dist = (price - cur) / cur * 100.0
    score = 50.0 + np.clip(dist * 4.0, -30, 30) + np.clip(slope_pct * 8.0, -20, 20)
    return label, float(np.clip(score, 0.0, 100.0))


def btc_regime(btc_htf: Series | None, ema_period: int = 50) -> tuple[str, float]:
    """BTC as the crypto risk switch. Unknown is treated conservatively upstream."""
    if btc_htf is None or len(btc_htf) == 0:
        return UNKNOWN, 50.0
    return trend_regime(btc_htf, ema_period)


def market_breadth(series_map: dict[str, Series], ema_period: int = 50) -> float:
    """Percentage of the eligible universe trading above its EMA. A cheap,
    honest measure of how many boats the tide is lifting."""
    total = ok = 0
    for s in series_map.values():
        if len(s) < ema_period + 2:
            continue
        e = last_valid(ema(s.close, ema_period))
        if not np.isfinite(e) or e <= 0:
            continue
        total += 1
        if float(s.close[-1]) > e:
            ok += 1
    if total == 0:
        return 50.0
    return ok / total * 100.0


def volatility_regime(series: Series, atr_pct: float) -> str:
    if not np.isfinite(atr_pct):
        return UNKNOWN
    if atr_pct < 1.0:
        return "low_vol"
    if atr_pct < 3.0:
        return "normal_vol"
    return "high_vol"


def classify_environment(btc_label: str, breadth: float, btc_atr_pct: float) -> str:
    """Coarse environment label, recorded for research (spec section 41).
    It does NOT currently gate anything beyond the BTC filter -- we record
    first and test whether it predicts anything before acting on it."""
    if btc_label == BEAR and breadth < 35:
        return "panic" if btc_atr_pct > 4.0 else "bear"
    if btc_label in (BULL,) and breadth > 60:
        return "strong_bull"
    if btc_label in (BULL, WEAK_BULL):
        return "weak_bull"
    if btc_atr_pct > 3.0:
        return "sideways_high_vol"
    return "sideways_low_vol"
