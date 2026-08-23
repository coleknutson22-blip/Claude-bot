"""Opportunity scoring.

Every component returns 0-100 and every weight has a written justification.
The weights below were chosen from trading rationale, NOT fitted to a
backtest. If a component cannot later be shown to improve out-of-sample
expectancy, the rule in spec section 42 is to delete it -- so each one is
recorded separately in the research database to make that test possible.

Scores are RELATIVE rankers, not probabilities. A score of 70 does not mean a
70% win rate; it means "better than a 60 on the factors we believe matter".
"""
from __future__ import annotations

import numpy as np

# weight -> (value, justification)
WEIGHTS: dict[str, float] = {
    "htf_trend": 0.18,      # trading with higher-timeframe structure is the
                            # single most defensible edge in trend following
    "breakout": 0.15,       # strength of the actual trigger
    "adx": 0.12,            # distinguishes trend from chop; low ADX breakouts fail
    "momentum": 0.12,       # multi-period persistence
    "rel_strength": 0.12,   # outperformance vs BTC: leaders tend to keep leading
    "rel_volume": 0.10,     # participation confirms the move is real
    "extension": 0.08,      # penalise chasing: late entries have worse R:R
    "liquidity": 0.08,      # tradability is a real cost, not a nicety
    "market_ctx": 0.05,     # BTC regime + breadth as a mild tilt, not a veto
}

JUSTIFICATIONS = {
    "htf_trend": "Alignment with 4h structure; counter-trend breakouts fail more often.",
    "breakout": "Distance above the Donchian level in ATR units; marginal breaks are noise.",
    "adx": "Trend strength filter; breakouts inside chop mean-revert.",
    "momentum": "Blended 24/72/168h ROC; persistence is the core trend premise.",
    "rel_strength": "Return vs BTC over the momentum window; leadership tends to persist.",
    "rel_volume": "Volume vs its own 24-bar average; unconfirmed breaks are suspect.",
    "extension": "Penalty for entering far above the trigger; protects reward-to-risk.",
    "liquidity": "Dollar volume and spread; illiquid fills destroy realised edge.",
    "market_ctx": "BTC regime and breadth; a mild tilt because BTC is already a hard filter.",
}


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(min(hi, max(lo, x)))


def _scale(value: float, lo: float, hi: float) -> float:
    """Linear map [lo, hi] -> [0, 100], clamped."""
    if not np.isfinite(value) or hi <= lo:
        return 0.0
    return _clamp((value - lo) / (hi - lo) * 100.0)


def score_htf_trend(htf_score: float) -> float:
    return _clamp(htf_score)


def score_breakout(close: float, donchian_high: float, atr: float) -> float:
    """How decisively did price clear the channel, measured in ATR?"""
    if atr <= 0 or not np.isfinite(donchian_high):
        return 0.0
    excess_atr = (close - donchian_high) / atr
    return _scale(excess_atr, 0.0, 1.0)


def score_adx(adx_val: float) -> float:
    # below 20 is chop; above 45 is often late-stage exhaustion
    if not np.isfinite(adx_val):
        return 0.0
    if adx_val <= 45:
        return _scale(adx_val, 15.0, 45.0)
    return _clamp(100.0 - (adx_val - 45.0) * 2.0, 40.0, 100.0)


def score_momentum(roc_values: list[float]) -> float:
    vals = [v for v in roc_values if np.isfinite(v)]
    if not vals:
        return 0.0
    blended = float(np.mean(vals))
    return _scale(blended, -5.0, 25.0)


def score_rel_strength(asset_roc: float, btc_roc: float) -> float:
    if not (np.isfinite(asset_roc) and np.isfinite(btc_roc)):
        return 50.0
    return _scale(asset_roc - btc_roc, -10.0, 20.0)


def score_rel_volume(rvol: float) -> float:
    return _scale(rvol, 0.8, 3.0)


def score_extension(close: float, donchian_high: float, atr: float,
                    max_atr: float) -> float:
    """Inverted: further above the trigger scores worse."""
    if atr <= 0 or not np.isfinite(donchian_high) or max_atr <= 0:
        return 0.0
    ext = (close - donchian_high) / atr
    return _clamp(100.0 - (ext / max_atr) * 100.0)


def score_liquidity(dollar_volume: float, spread_bps: float,
                    min_volume: float) -> float:
    if not np.isfinite(dollar_volume) or dollar_volume <= 0:
        return 0.0
    vol_score = _scale(np.log10(max(dollar_volume, 1.0)),
                       np.log10(max(min_volume, 1.0)), 9.0)   # 1e9 = deep
    spread_score = _clamp(100.0 - spread_bps * 5.0)
    return 0.6 * vol_score + 0.4 * spread_score


def score_market_context(btc_score: float, breadth: float) -> float:
    return _clamp(0.6 * btc_score + 0.4 * breadth)


def combine(components: dict[str, float]) -> float:
    """Weighted mean over the components actually present."""
    total_w = sum(WEIGHTS[k] for k in components if k in WEIGHTS)
    if total_w <= 0:
        return 0.0
    acc = sum(components[k] * WEIGHTS[k] for k in components if k in WEIGHTS)
    return float(round(acc / total_w, 2))


def apply_news_penalty(score: float, news: dict | None) -> tuple[float, str]:
    """News reduces attractiveness; only a severe, high-confidence event is
    allowed to veto outright, and that veto happens in the news module, not here."""
    if not news:
        return score, ""
    sev = int(news.get("severity", 0))
    conf = float(news.get("confidence", 0.0))
    if sev <= 0 or conf < 0.5:
        return score, ""
    penalty = min(40.0, sev * 8.0 * conf)
    return max(0.0, score - penalty), f"news penalty -{penalty:.0f} ({news.get('category','')})"
