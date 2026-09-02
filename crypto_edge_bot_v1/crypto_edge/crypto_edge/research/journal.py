"""Research journal: record EVERY evaluation, not just the trades we took.

Executed, rejected and near-miss signals all land in `observations` with their
full feature vector. This is what later answers "are our filters helping?"
"""
from __future__ import annotations

from typing import Any

from ..logging_setup import log_event
from ..storage.repo import Repo
from ..strategy.base import Signal

ENTERED, REJECTED_STRATEGY, REJECTED_RISK, RANKED_OUT = (
    "ENTERED", "REJECTED_STRATEGY", "REJECTED_RISK", "RANKED_OUT")


class ResearchJournal:
    def __init__(self, repo: Repo) -> None:
        self.repo = repo

    def record(self, sig: Signal, decision: str, reject_reason: str = "",
               rank: int | None = None, extra: dict[str, Any] | None = None) -> str | None:
        features = dict(sig.features)
        if sig.components:
            features["score_components"] = sig.components
        if extra:
            features.update(extra)
        # numpy arrays never make it into the DB
        features.pop("returns", None)
        return self.repo.add_observation({
            "ts_ms": sig.ts_ms, "symbol": sig.symbol, "candle_id": sig.candle_id,
            "strategy": sig.strategy, "strategy_version": sig.version,
            "decision": decision, "reject_reason": reject_reason or sig.reject_reason,
            "side": getattr(sig, "side", "long"),
            "score": sig.score, "rank": rank,
            "price": sig.ref_price if sig.ref_price > 0 else None,
            "features": _jsonable(features)})

    def entry_journal(self, sig: Signal, ctx_extra: dict[str, Any]) -> dict:
        """The snapshot attached to a position at entry (spec section 7)."""
        f = sig.features
        return _jsonable({
            "symbol": sig.symbol, "signal_ts": sig.ts_ms, "candle_id": sig.candle_id,
            "strategy": sig.strategy, "strategy_version": sig.version,
            "entry_ref_price": sig.ref_price, "atr": f.get("atr"),
            "atr_pct": f.get("atr_pct"), "adx": f.get("adx"), "rsi": f.get("rsi"),
            "rel_volume": f.get("rel_volume"), "ema_fast": f.get("ema_fast"),
            "ema_slow": f.get("ema_slow"), "ema_trend": f.get("ema_trend"),
            "donchian_high": f.get("donchian_high"),
            "extension_atr": f.get("extension_atr"),
            "momentum": f.get("momentum"), "realized_vol_pct": f.get("realized_vol_pct"),
            "htf_regime": f.get("htf_regime"), "btc_regime": f.get("btc_regime"),
            "breadth_pct": f.get("breadth_pct"), "signal_score": sig.score,
            "score_components": sig.components, "initial_stop": sig.stop_price,
            **ctx_extra})


def _jsonable(obj):
    """Convert numpy scalars/arrays and NaN into JSON-safe values."""
    import math

    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
