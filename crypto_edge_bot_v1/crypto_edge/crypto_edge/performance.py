"""Performance accounting.

Two principles:
  1. Net P&L is the headline. Gross, fees and slippage are reported alongside
     it so the cost of trading is never hidden.
  2. Ratios that need a sample are reported WITH their sample size and a
     `reliable` flag. A Sharpe from nine trades is decoration, not evidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .models import Position
from .storage.repo import Repo
from .timeutils import fmt_duration, now_ms, utc_date

MIN_TRADES_FOR_RATIOS = 30
MIN_DAYS_FOR_RATIOS = 20


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


@dataclass
class PerformanceReport:
    account: dict = field(default_factory=dict)
    trading: dict = field(default_factory=dict)
    risk: dict = field(default_factory=dict)
    advanced: dict = field(default_factory=dict)
    open_positions: list[dict] = field(default_factory=list)
    sample: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"account": self.account, "trading": self.trading, "risk": self.risk,
                "advanced": self.advanced, "open_positions": self.open_positions,
                "sample": self.sample}


class PerformanceCalculator:
    """Reports on ONE strategy's ledger.

    `strategy` may be left unset only while a single sub-account exists; once a
    second strategy has a ledger the caller must say which one it means, rather
    than silently reporting whichever row sorted first.
    """

    def __init__(self, repo: Repo, strategy: str | None = None) -> None:
        self.repo = repo
        self.strategy = strategy

    def _resolve(self, strategy: str | None) -> str:
        if strategy:
            return strategy
        if self.strategy:
            return self.strategy
        names = [a["strategy"] for a in self.repo.all_accounts()]
        if len(names) == 1:
            return names[0]
        raise ValueError(
            f"which strategy? {len(names)} sub-accounts exist: {names}")

    # ------------------------------------------------------------- helpers
    def _open_rows(self, marks: dict[str, float],
                   strategy: str | None = None) -> list[dict]:
        rows = []
        for p in self.repo.get_positions(strategy):
            px = marks.get(p.symbol, p.entry_fill_price)
            rows.append({
                "symbol": p.symbol, "qty": p.qty, "entry": p.entry_fill_price,
                "current": px, "value": p.mark_value(px),
                "collateral": p.collateral_value(px),
                "side": p.side,
                "unrealized": p.unrealized(px),
                "unrealized_pct": _safe_div((px - p.entry_fill_price),
                                            p.entry_fill_price) * 100.0 * p.direction,
                "stop": p.current_stop, "initial_stop": p.initial_stop,
                "risk_amount": p.risk_amount, "score": p.signal_score,
                "held": fmt_duration((now_ms() - p.entry_ms) / 1000.0),
                "strategy": p.strategy, "version": p.strategy_version})
        return rows

    # -------------------------------------------------------------- report
    def report(self, marks: dict[str, float] | None = None,
               strategy: str | None = None,
               version: str | None = None) -> PerformanceReport:
        marks = marks or {}
        name = self._resolve(strategy)
        acct = self.repo.get_account(name)
        trades = self.repo.get_trades(name, version)
        open_rows = self._open_rows(marks, name)

        cash = float(acct["cash"])
        deployed = sum(r["value"] for r in open_rows)          # gross exposure
        unrealized = sum(r["unrealized"] for r in open_rows)
        # Equity uses COLLATERAL, not gross exposure: a short contributes the
        # capital reserved against it plus its P&L, not the market value of
        # stock it does not own.
        equity = cash + sum(r["collateral"] for r in open_rows)
        start = float(acct["starting_equity"])
        realized = float(acct["realized_pnl"])

        gross = sum(t["gross_pnl"] for t in trades)
        fees = sum(t["fees"] for t in trades)
        slip = sum(t["slippage_cost"] for t in trades)
        net = sum(t["net_pnl"] for t in trades)

        peak = max(float(acct["peak_equity"]), equity)
        dd = _safe_div(peak - equity, peak) * 100.0

        rep = PerformanceReport()
        rep.account = {
            "starting_balance": start, "current_equity": equity, "cash": cash,
            "capital_deployed": deployed,
            "available_capital": cash,
            "realized_pnl": realized, "unrealized_pnl": unrealized,
            "total_pnl": realized + unrealized,
            "gross_pnl": gross, "net_pnl": net,
            "total_fees": fees, "estimated_slippage_cost": slip,
            "account_return_pct": _safe_div(equity - start, start) * 100.0,
            "halted": bool(acct["halted"]), "halt_reason": acct["halt_reason"],
        }

        wins = [t for t in trades if t["net_pnl"] > 0]
        losses = [t for t in trades if t["net_pnl"] <= 0]
        gross_profit = sum(t["net_pnl"] for t in wins)
        gross_loss = abs(sum(t["net_pnl"] for t in losses))
        n = len(trades)
        win_rate = _safe_div(len(wins), n) * 100.0
        avg_win = _safe_div(gross_profit, len(wins))
        avg_loss = _safe_div(gross_loss, len(losses))

        rep.trading = {
            "closed_trades": n, "wins": len(wins), "losses": len(losses),
            "win_rate_pct": win_rate,
            "average_winner": avg_win, "average_loser": -avg_loss,
            "largest_winner": max((t["net_pnl"] for t in wins), default=0.0),
            "largest_loser": min((t["net_pnl"] for t in losses), default=0.0),
            "average_trade": _safe_div(net, n),
            "expectancy_per_trade": (win_rate / 100.0) * avg_win - (1 - win_rate / 100.0) * avg_loss,
            "profit_factor": _safe_div(gross_profit, gross_loss,
                                       float("inf") if gross_profit > 0 else 0.0),
            "gross_profit": gross_profit, "gross_loss": -gross_loss,
            "average_holding": fmt_duration(_safe_div(
                sum(t["duration_s"] for t in trades), n)),
            "average_holding_s": _safe_div(sum(t["duration_s"] for t in trades), n),
            "max_consecutive_wins": _max_streak(trades, True),
            "max_consecutive_losses": _max_streak(trades, False),
        }

        daily = self.repo.get_all_daily()
        daily_pnls = [d["realized_pnl"] for d in daily]
        rep.risk = {
            "peak_equity": peak, "current_drawdown_pct": dd,
            "max_drawdown_pct": self._max_drawdown_pct(equity),
            "average_risk_per_trade": _safe_div(
                sum(t["journal"].get("risk_amount", 0.0) or 0.0 for t in trades), n),
            "max_simultaneous_exposure_pct": self._max_exposure_pct(),
            "largest_daily_loss": min(daily_pnls, default=0.0),
            "largest_daily_gain": max(daily_pnls, default=0.0),
            "open_positions": len(open_rows),
            "current_exposure_pct": _safe_div(deployed, equity) * 100.0,
        }

        rep.advanced = self._advanced(trades, daily)
        rep.open_positions = open_rows
        rep.sample = {
            "closed_trades": n,
            "trading_days": len(daily),
            "ratios_reliable": n >= MIN_TRADES_FOR_RATIOS and len(daily) >= MIN_DAYS_FOR_RATIOS,
            "min_trades_for_ratios": MIN_TRADES_FOR_RATIOS,
            "min_days_for_ratios": MIN_DAYS_FOR_RATIOS,
            "note": ("SAMPLE TOO SMALL -- ratios below are not yet meaningful"
                     if not (n >= MIN_TRADES_FOR_RATIOS and len(daily) >= MIN_DAYS_FOR_RATIOS)
                     else "sample size adequate for indicative ratios"),
        }
        return rep

    # ------------------------------------------------------------ advanced
    def _advanced(self, trades: list[dict], daily: list[dict]) -> dict:
        out: dict = {"sample_trades": len(trades), "sample_days": len(daily)}
        rets = []
        for d in daily:
            if d["start_equity"] > 0:
                rets.append((d["end_equity"] - d["start_equity"]) / d["start_equity"])
        arr = np.asarray(rets, dtype=float)
        reliable = len(trades) >= MIN_TRADES_FOR_RATIOS and len(daily) >= MIN_DAYS_FOR_RATIOS
        out["reliable"] = reliable

        if len(arr) >= 2 and np.std(arr, ddof=1) > 0:
            mean, sd = float(np.mean(arr)), float(np.std(arr, ddof=1))
            out["return_volatility_daily_pct"] = sd * 100.0
            out["sharpe_annualised"] = mean / sd * math.sqrt(365)
            downside = arr[arr < 0]
            dsd = float(np.std(downside, ddof=1)) if len(downside) >= 2 else 0.0
            out["sortino_annualised"] = (mean / dsd * math.sqrt(365)) if dsd > 0 else None
        else:
            out["return_volatility_daily_pct"] = None
            out["sharpe_annualised"] = None
            out["sortino_annualised"] = None

        mdd = self._max_drawdown_pct()
        if daily and mdd > 0:
            total_ret = _safe_div(daily[-1]["end_equity"] - daily[0]["start_equity"],
                                  daily[0]["start_equity"])
            years = max(len(daily) / 365.0, 1e-9)
            cagr = ((1 + total_ret) ** (1 / years) - 1) if total_ret > -1 else -1.0
            out["calmar"] = cagr / (mdd / 100.0)
        else:
            out["calmar"] = None

        nets = [t["net_pnl"] for t in trades]
        out["trade_expectancy"] = float(np.mean(nets)) if nets else 0.0
        out["trade_stdev"] = float(np.std(nets, ddof=1)) if len(nets) >= 2 else None
        if not reliable:
            out["warning"] = ("Insufficient sample: these figures are reported for "
                              "completeness only and should not be interpreted as edge.")
        return out

    def _max_drawdown_pct(self, current_equity: float | None = None) -> float:
        curve = [r["equity"] for r in self.repo.get_equity_curve()]
        if current_equity is not None:
            curve.append(current_equity)
        if not curve:
            return 0.0
        peak, mdd = curve[0], 0.0
        for v in curve:
            peak = max(peak, v)
            if peak > 0:
                mdd = max(mdd, (peak - v) / peak * 100.0)
        return mdd

    def _max_exposure_pct(self) -> float:
        rows = self.repo.conn.execute(
            """SELECT MAX((equity - cash) / equity * 100.0) AS m
               FROM equity_snapshots WHERE equity > 0""").fetchone()
        return float(rows["m"]) if rows and rows["m"] is not None else 0.0

    # ---------------------------------------------------- category breakdown
    def by_category(self, min_sample: int = 10) -> dict[str, list[dict]]:
        """Performance sliced by recorded context. Buckets below `min_sample`
        are marked rather than shown as if meaningful (spec section 45)."""
        trades = self.repo.get_trades()
        out: dict[str, list[dict]] = {}

        def bucket(name: str, keyfn):
            groups: dict[str, list[float]] = {}
            for t in trades:
                try:
                    k = keyfn(t)
                except Exception:
                    k = None
                if k is None:
                    continue
                groups.setdefault(str(k), []).append(t["net_pnl"])
            rows = []
            for k, vals in sorted(groups.items()):
                wins = sum(1 for v in vals if v > 0)
                rows.append({"bucket": k, "n": len(vals),
                             "net_pnl": sum(vals), "avg": float(np.mean(vals)),
                             "win_rate_pct": wins / len(vals) * 100.0,
                             "sufficient_sample": len(vals) >= min_sample})
            out[name] = rows

        bucket("symbol", lambda t: t["symbol"])
        bucket("strategy", lambda t: f"{t['strategy']}@{t['strategy_version']}")
        bucket("exit_reason", lambda t: t["exit_reason"])
        bucket("btc_regime", lambda t: t["journal"].get("btc_regime"))
        bucket("htf_regime", lambda t: t["journal"].get("htf_regime"))
        bucket("score_bucket", lambda t: _bucket_num(t["journal"].get("signal_score"), 10))
        bucket("adx_bucket", lambda t: _bucket_num(t["journal"].get("adx"), 10))
        bucket("rel_volume_bucket", lambda t: _bucket_num(t["journal"].get("rel_volume"), 1))
        bucket("day_of_week", lambda t: utc_date(t["entry_ms"]))
        bucket("holding_bucket", lambda t: _bucket_num(t["duration_s"] / 3600.0, 24))
        return out


def _bucket_num(v, width: float):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    lo = math.floor(f / width) * width
    return f"{lo:g}-{lo + width:g}"


def _max_streak(trades: list[dict], want_win: bool) -> int:
    best = cur = 0
    for t in trades:
        is_win = t["net_pnl"] > 0
        if is_win == want_win:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best
