"""Central portfolio risk manager.

Every proposed trade -- from any strategy -- must pass through `check_entry`.
That is the single choke point the multi-strategy architecture depends on: a
strategy proposes, the portfolio manager disposes.

Rejections are returned as structured reasons rather than silently dropped, so
the research database can later answer "are our filters actually helping?".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import RiskCfg
from ..indicators import correlation
from ..logging_setup import log_event


@dataclass
class RiskDecision:
    allowed: bool
    reason: str = ""
    detail: dict | None = None


@dataclass
class CircuitState:
    halted: bool
    reason: str = ""


class RiskManager:
    def __init__(self, cfg: RiskCfg) -> None:
        self.cfg = cfg

    # ------------------------------------------------------- circuit breakers
    def check_circuit_breakers(self, equity: float, peak_equity: float,
                               daily_start_equity: float) -> CircuitState:
        """Two independent kill switches. Both fail closed: once tripped, the
        engine stops opening positions until the condition clears (daily) or a
        human intervenes (max drawdown)."""
        if peak_equity > 0:
            dd = (peak_equity - equity) / peak_equity * 100.0
            if dd >= self.cfg.max_drawdown_pct:
                return CircuitState(True,
                    f"MAX DRAWDOWN kill switch: {dd:.2f}% >= {self.cfg.max_drawdown_pct}%")
        if daily_start_equity > 0:
            day_loss = (daily_start_equity - equity) / daily_start_equity * 100.0
            if day_loss >= self.cfg.daily_loss_limit_pct:
                return CircuitState(True,
                    f"DAILY LOSS limit: -{day_loss:.2f}% >= {self.cfg.daily_loss_limit_pct}%")
        return CircuitState(False)

    def daily_loss_remaining(self, equity: float, daily_start_equity: float) -> float:
        if daily_start_equity <= 0:
            return 0.0
        limit = daily_start_equity * self.cfg.daily_loss_limit_pct / 100.0
        used = daily_start_equity - equity
        return max(0.0, limit - used)

    # ------------------------------------------------------------ entry gate
    def check_entry(self, *, symbol: str, equity: float, exposure: float,
                    n_open: int, open_symbols: list[str],
                    candidate_returns: np.ndarray | None = None,
                    open_returns: dict[str, np.ndarray] | None = None,
                    entries_this_cycle: int = 0) -> RiskDecision:
        if symbol in open_symbols:
            return RiskDecision(False, "position already open in symbol")
        if n_open >= self.cfg.max_open_positions:
            return RiskDecision(False,
                f"max open positions reached ({n_open}/{self.cfg.max_open_positions})")
        if entries_this_cycle >= self.cfg.max_new_entries_per_cycle:
            return RiskDecision(False, "max new entries per cycle reached")
        if equity <= 0:
            return RiskDecision(False, "non-positive equity")

        exposure_pct = exposure / equity * 100.0 if equity else 0.0
        if exposure_pct >= self.cfg.max_portfolio_exposure_pct:
            return RiskDecision(False,
                f"max portfolio exposure ({exposure_pct:.1f}% >= "
                f"{self.cfg.max_portfolio_exposure_pct}%)")

        # Correlation: crypto longs are frequently the same trade wearing
        # different tickers. Reject redundancy rather than calling it diversity.
        if candidate_returns is not None and open_returns:
            worst_sym, worst = "", 0.0
            for sym, rets in open_returns.items():
                c = correlation(candidate_returns, rets)
                if c > worst:
                    worst, worst_sym = c, sym
            if worst >= self.cfg.max_correlation:
                return RiskDecision(False,
                    f"correlation with existing {worst_sym} position = {worst:.2f}",
                    {"correlation": worst, "with": worst_sym})
        return RiskDecision(True)

    # --------------------------------------------------------------- helpers
    def risk_amount(self, equity: float) -> float:
        return equity * self.cfg.risk_per_trade_pct / 100.0

    def log_rejection(self, symbol: str, reason: str, **extra) -> None:
        log_event("strategy", "INFO", f"{symbol} SIGNAL REJECTED",
                  symbol=symbol, reason=reason, **extra)
