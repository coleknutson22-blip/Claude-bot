"""The persistent paper account.

Every mutation goes through a transaction that touches both the account row and
the position/trade tables, so a crash can never leave cash disagreeing with
holdings. Nothing here rewrites history: closed trades are append-only, and
there is no code path that deletes or edits a losing trade.
"""
from __future__ import annotations

from typing import Any

from ..execution.paper_broker import PaperBroker, realise_pnl
from ..logging_setup import log_event
from ..models import ClosedTrade, Fill, Position
from ..storage.repo import Repo, new_id
from ..timeutils import now_ms, utc_date


class PaperAccount:
    def __init__(self, repo: Repo, broker: PaperBroker, starting_equity: float) -> None:
        self.repo = repo
        self.broker = broker
        self.repo.ensure_account(starting_equity)

    # ----------------------------------------------------------- state read
    @property
    def state(self) -> dict:
        return self.repo.get_account()

    def positions(self) -> list[Position]:
        return self.repo.get_positions()

    def cash(self) -> float:
        return float(self.state["cash"])

    def exposure(self, marks: dict[str, float]) -> float:
        total = 0.0
        for p in self.positions():
            total += p.mark_value(marks.get(p.symbol, p.entry_fill_price))
        return total

    def equity(self, marks: dict[str, float]) -> float:
        return self.cash() + self.exposure(marks)

    def unrealized(self, marks: dict[str, float]) -> float:
        return sum(p.unrealized(marks.get(p.symbol, p.entry_fill_price))
                   for p in self.positions())

    def drawdown_pct(self, marks: dict[str, float]) -> float:
        eq = self.equity(marks)
        peak = max(float(self.state["peak_equity"]), eq)
        if peak <= 0:
            return 0.0
        return max(0.0, (peak - eq) / peak * 100.0)

    # ---------------------------------------------------------------- open
    def open_position(self, *, symbol: str, strategy: str, strategy_version: str,
                      qty: float, ref_price: float, fill: Fill, initial_stop: float,
                      risk_amount: float, candle_id: str, signal_score: float,
                      journal: dict[str, Any]) -> Position | None:
        """Claims the candle, debits cash, records the position -- atomically.

        Returns None if the candle was already processed or a position already
        exists for this symbol. Both are duplicate-entry guards that survive a
        restart because they are enforced by the database, not by memory.
        """
        pos = Position(
            id=new_id("pos"), symbol=symbol, strategy=strategy,
            strategy_version=strategy_version, side="long", qty=qty,
            entry_ref_price=ref_price, entry_fill_price=fill.fill_price,
            entry_ms=fill.ts_ms, entry_fee=fill.fee, entry_slippage=fill.slippage_cost,
            initial_stop=initial_stop, current_stop=initial_stop,
            highest_price=fill.fill_price, lowest_price=fill.fill_price,
            risk_amount=risk_amount, candle_id=candle_id,
            signal_score=signal_score, mfe=0.0, mae=0.0, journal=journal)

        cost = fill.notional + fill.fee
        try:
            with self.repo.tx():
                if not self.repo.mark_candle_processed(candle_id):
                    return None
                if self.repo.get_position_by_symbol(symbol) is not None:
                    raise _Abort("position already open for symbol")
                acct = self.repo.get_account()
                if cost > float(acct["cash"]) + 1e-9:
                    raise _Abort("insufficient cash")
                self.repo.add_position(pos)
                self.repo.update_account(
                    cash=float(acct["cash"]) - cost,
                    total_fees=float(acct["total_fees"]) + fill.fee,
                    total_slippage=float(acct["total_slippage"]) + fill.slippage_cost)
        except _Abort as e:
            log_event("trades", "WARNING", "entry aborted", symbol=symbol, reason=str(e))
            return None

        log_event("trades", "INFO", "ENTRY", symbol=symbol, qty=qty,
                  ref=ref_price, fill=fill.fill_price, fee=fill.fee,
                  stop=initial_stop, risk=risk_amount, score=signal_score,
                  position_id=pos.id, candle_id=candle_id)
        return pos

    # --------------------------------------------------------------- close
    def close_position(self, pos: Position, fill: Fill, exit_reason: str,
                       marks: dict[str, float] | None = None) -> ClosedTrade | None:
        """Credits cash, writes the closed trade, deletes the open position.

        Deleting the position row inside the same transaction is what prevents
        a double close: the second attempt finds nothing to close.
        """
        pnl = realise_pnl(pos.entry_ref_price, pos.entry_fill_price,
                          fill.ref_price, fill.fill_price, pos.qty,
                          pos.entry_fee, fill.fee)
        proceeds = fill.notional - fill.fee

        try:
            with self.repo.tx():
                live = self.repo.conn.execute(
                    "SELECT 1 FROM positions WHERE id=?", (pos.id,)).fetchone()
                if live is None:
                    raise _Abort("position already closed")
                acct = self.repo.get_account()
                new_cash = float(acct["cash"]) + proceeds
                marks_ = dict(marks or {})
                marks_.pop(pos.symbol, None)

                # equity after this close = new cash + remaining positions
                remaining = 0.0
                for p in self.repo.get_positions():
                    if p.id == pos.id:
                        continue
                    remaining += p.mark_value(marks_.get(p.symbol, p.entry_fill_price))
                equity_after = new_cash + remaining
                peak = max(float(acct["peak_equity"]), equity_after)

                entry_notional = pos.qty * pos.entry_fill_price
                trade = ClosedTrade(
                    id=new_id("trd"), position_id=pos.id, symbol=pos.symbol,
                    strategy=pos.strategy, strategy_version=pos.strategy_version,
                    qty=pos.qty, entry_ref_price=pos.entry_ref_price,
                    entry_fill_price=pos.entry_fill_price, entry_ms=pos.entry_ms,
                    exit_ref_price=fill.ref_price, exit_fill_price=fill.fill_price,
                    exit_ms=fill.ts_ms, exit_reason=exit_reason,
                    initial_stop=pos.initial_stop, final_stop=pos.current_stop,
                    gross_pnl=pnl["gross_pnl"], fees=pnl["fees"],
                    slippage_cost=pnl["slippage_cost"], net_pnl=pnl["net_pnl"],
                    return_pct=(pnl["net_pnl"] / entry_notional * 100.0) if entry_notional else 0.0,
                    account_return_pct=(pnl["net_pnl"] / equity_after * 100.0) if equity_after else 0.0,
                    mfe=pos.mfe, mae=pos.mae,
                    duration_s=max(0.0, (fill.ts_ms - pos.entry_ms) / 1000.0),
                    equity_after=equity_after,
                    journal={**pos.journal, "exit_reason": exit_reason})

                self.repo.add_trade(trade)
                self.repo.remove_position(pos.id)
                self.repo.update_account(
                    cash=new_cash, peak_equity=peak,
                    realized_pnl=float(acct["realized_pnl"]) + pnl["net_pnl"],
                    total_fees=float(acct["total_fees"]) + fill.fee,
                    total_slippage=float(acct["total_slippage"]) + fill.slippage_cost)

                date = utc_date(fill.ts_ms)
                day = self.repo.get_daily(date)
                start_eq = day["start_equity"] if day else float(acct["daily_start_equity"])
                self.repo.upsert_daily(
                    date, start_eq, equity_after,
                    realized_pnl=pnl["net_pnl"], trades=1,
                    wins=1 if pnl["net_pnl"] > 0 else 0,
                    losses=1 if pnl["net_pnl"] <= 0 else 0,
                    fees=pnl["fees"])
        except _Abort as e:
            log_event("trades", "WARNING", "exit aborted", symbol=pos.symbol, reason=str(e))
            return None

        log_event("trades", "INFO", "EXIT", symbol=pos.symbol, reason=exit_reason,
                  gross=pnl["gross_pnl"], fees=pnl["fees"],
                  slippage=pnl["slippage_cost"], net=pnl["net_pnl"],
                  trade_id=trade.id)
        return trade

    # ------------------------------------------------------ mark to market
    def update_marks(self, marks: dict[str, float]) -> None:
        """Track excursions and the running peak. Called every cycle."""
        for p in self.positions():
            px = marks.get(p.symbol)
            if px is None or px <= 0:
                continue
            hi = max(p.highest_price, px)
            lo = min(p.lowest_price, px)
            mfe = max(p.mfe, (hi - p.entry_fill_price) * p.qty)
            mae = min(p.mae, (lo - p.entry_fill_price) * p.qty)
            if (hi != p.highest_price or lo != p.lowest_price
                    or mfe != p.mfe or mae != p.mae):
                self.repo.update_position(p.id, highest_price=hi, lowest_price=lo,
                                          mfe=mfe, mae=mae)
        eq = self.equity(marks)
        acct = self.state
        if eq > float(acct["peak_equity"]):
            self.repo.update_account(peak_equity=eq)

    def roll_daily(self, ts: int, marks: dict[str, float]) -> bool:
        """Start a new UTC trading day. Returns True if a rollover happened."""
        acct = self.state
        today = utc_date(ts)
        if acct["daily_date"] == today:
            return False
        eq = self.equity(marks)
        self.repo.update_account(daily_date=today, daily_start_equity=eq)
        self.repo.upsert_daily(today, eq, eq)
        log_event("performance", "INFO", "daily rollover", date=today, equity=eq)
        return True

    def set_stop(self, pos: Position, new_stop: float) -> None:
        self.repo.update_position(pos.id, current_stop=new_stop)


class _Abort(Exception):
    pass
