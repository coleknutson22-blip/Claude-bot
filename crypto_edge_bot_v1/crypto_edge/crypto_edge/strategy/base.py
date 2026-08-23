"""Strategy interface.

A strategy is a PURE function of (market data, context) -> Signal. It never
touches the account, never places orders, and never consults an LLM. The
portfolio manager decides whether a signal becomes a trade.

This purity is what makes the backtester and the live engine provably run the
same logic: both call `evaluate()` with a closed-candle Series and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..models import Series


@dataclass
class MarketContext:
    """Everything a strategy may know about the wider market at decision time.
    Built once per cycle from closed candles only."""
    ts_ms: int
    btc_regime: str = "unknown"
    btc_regime_score: float = 50.0
    breadth_pct: float = 50.0
    n_candidates: int = 0
    news_by_symbol: dict[str, dict] = field(default_factory=dict)
    derivatives_by_symbol: dict[str, dict] = field(default_factory=dict)
    blocked_symbols: dict[str, str] = field(default_factory=dict)


@dataclass
class Signal:
    symbol: str
    strategy: str
    version: str
    candle_id: str
    ts_ms: int
    ref_price: float
    stop_price: float
    atr: float
    score: float
    passed: bool
    reject_reason: str = ""
    components: dict[str, float] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    returns: np.ndarray | None = None     # for correlation checks

    @property
    def is_entry(self) -> bool:
        return self.passed and self.score > 0


class Strategy(Protocol):
    name: str
    version: str

    def evaluate(self, series: Series, htf: Series,
                 ctx: MarketContext) -> Signal: ...

    def should_exit(self, position, series: Series, ctx: MarketContext) -> str | None: ...
