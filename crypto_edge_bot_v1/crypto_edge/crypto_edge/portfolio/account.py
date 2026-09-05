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
    """One strategy's ledger. Bound to a strategy at construction so no call
    site can accidentally read or spend another strategy's cash.

    ACCOUNTING MODEL
    ----------------
    Equity is free cash plus, for each position, the capital reserved against it
    and the profit or loss on it:

        equity = cash + SUM(margin_held + unrealized)

    For a 1x long, `margin_held` is the entry notional and `unrealized` is
    `(mark - entry) * qty`, so the two sum to `qty * mark` -- exactly the
    `cash + SUM(qty * mark)` the spot-only model computed before. That identity
    is why the change is invisible to Strategy A, and it is asserted directly in
    tests/test_strategy_a_equivalence.py rather than assumed here.

    Expressed this way the model also holds a short, where there is no `qty *
    mark` to add: collateral is reserved and P&L accrues against it.
    """

    def __init__(self, repo: Repo, broker: PaperBroker, starting_equity: float,
                 strategy: str, borrow_bps_per_day: float = 0.0) -> None:
        self.repo = repo
        self.broker = broker
        self.strategy = strategy
        # Simulated short borrow. Zero for a long-only strategy, so Strategy A
        # is untouched by its presence.
        self.borrow_bps_per_day = float(borrow_bps_per_day)
        self.repo.ensure_account(strategy, starting_equity)

    def financing(self, pos: Position, now: int | None = None) -> float:
        """Borrow accrued on one position so far. Longs always zero."""
        from .aggressive_exits import financing_cost
        return financing_cost(pos, now if now is not None else now_ms(),
                              self.borrow_bps_per_day)

    # ----------------------------------------------------------- state read
    @property
    def state(self) -> dict:
        return self.repo.get_account(self.strategy)

    def positions(self) -> list[Position]:
        return self.repo.get_positions(self.strategy)

    def cash(self) -> float:
        return float(self.state["cash"])

    def exposure(self, marks: dict[str, float]) -> float:
        """GROSS market exposure -- longs and shorts both add, never net off."""
        total = 0.0
        for p in self.positions():
            total += p.mark_value(marks.get(p.symbol, p.entry_fill_price))
        return total

    def position_value(self, marks: dict[str, float]) -> float:
        """What the open positions contribute to equity, financing deducted.

        Accrued borrow is subtracted while the position is still OPEN. Charging
        it only at exit would let a short's equity read high for as long as it
        stayed open and then drop on close -- the cost would be real but
        invisible until it was too late to act on.
        """
        total = 0.0
        now = now_ms()
        for p in self.positions():
            total += p.collateral_value(marks.get(p.symbol, p.entry_fill_price))
            total -= self.financing(p, now)
        return total

    def equity(self, marks: dict[str, float]) -> float:
        return self.cash() + self.position_value(marks)

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
                      journal: dict[str, Any], side: str = "long",
                      margin_held: float | None = None) -> Position | None:
        """Claims the candle, debits cash, records the position -- atomically.

        Returns None if the candle was already processed or a position already
        exists for this symbol. Both are duplicate-entry guards that survive a
        restart because they are enforced by the database, not by memory.
        """
        if strategy != self.strategy:
            raise ValueError(
                f"account is bound to '{self.strategy}', refusing to open a "
                f"position for '{strategy}'")
        # At 1x the collateral reserved IS the notional, for a long or a short.
        margin = fill.notional if margin_held is None else float(margin_held)
        pos = Position(
            id=new_id("pos"), symbol=symbol, strategy=strategy,
            strategy_version=strategy_version, side=side, qty=qty,
            entry_ref_price=ref_price, entry_fill_price=fill.fill_price,
            entry_ms=fill.ts_ms, entry_fee=fill.fee, entry_slippage=fill.slippage_cost,
            initial_stop=initial_stop, current_stop=initial_stop,
            highest_price=fill.fill_price, lowest_price=fill.fill_price,
            risk_amount=risk_amount, candle_id=candle_id,
            signal_score=signal_score, mfe=0.0, mae=0.0,
            margin_held=margin, journal=journal)

        cost = margin + fill.fee
        try:
            with self.repo.tx():
                if not self.repo.mark_candle_processed(candle_id, strategy):
                    return None
                if self.repo.get_position(strategy, symbol) is not None:
                    raise _Abort("position already open for symbol")
                acct = self.repo.get_account(strategy)
                if cost > float(acct["cash"]) + 1e-9:
                    raise _Abort("insufficient cash")
                self.repo.add_position(pos)
                self.repo.update_account(
                    strategy,
                    cash=float(acct["cash"]) - cost,
                    total_fees=float(acct["total_fees"]) + fill.fee,
                    total_slippage=float(acct["total_slippage"]) + fill.slippage_cost)
        except _Abort as e:
            log_event("trades", "WARNING", "entry aborted", symbol=symbol, reason=str(e))
            return None

        log_event("trades", "INFO", "ENTRY", symbol=symbol, side=side, qty=qty,
                  strategy=strategy, ref=ref_price, fill=fill.fill_price, fee=fill.fee,
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
                          pos.entry_fee, fill.fee, direction=pos.direction)
        # Release the collateral and settle the P&L against it. For a 1x long
        # this is identical to the old `notional - fee`:
        #   margin + (exit-entry)*qty - fee  ==  qty*exit - fee
        gross_move = (fill.fill_price - pos.entry_fill_price) * pos.qty * pos.direction
        borrow = self.financing(pos, fill.ts_ms)
        proceeds = pos.margin_held + gross_move - fill.fee - borrow
        pnl["net_pnl"] -= borrow
        pnl["financing"] = borrow

        try:
            with self.repo.tx():
                live = self.repo.conn.execute(
                    "SELECT 1 FROM positions WHERE id=?", (pos.id,)).fetchone()
                if live is None:
                    raise _Abort("position already closed")
                acct = self.repo.get_account(pos.strategy)
                new_cash = float(acct["cash"]) + proceeds
                marks_ = dict(marks or {})
                marks_.pop(pos.symbol, None)

                # equity after this close = new cash + remaining positions
                remaining = 0.0
                for p in self.repo.get_positions(pos.strategy):
                    if p.id == pos.id:
                        continue
                    remaining += p.collateral_value(
                        marks_.get(p.symbol, p.entry_fill_price))
                equity_after = new_cash + remaining
                peak = max(float(acct["peak_equity"]), equity_after)

                entry_notional = pos.qty * pos.entry_fill_price
                trade = ClosedTrade(
                    id=new_id("trd"), position_id=pos.id, symbol=pos.symbol,
                    strategy=pos.strategy, strategy_version=pos.strategy_version,
                    side=pos.side, qty=pos.qty, entry_ref_price=pos.entry_ref_price,
                    entry_fill_price=pos.entry_fill_price, entry_ms=pos.entry_ms,
                    exit_ref_price=fill.ref_price, exit_fill_price=fill.fill_price,
                    exit_ms=fill.ts_ms, exit_reason=exit_reason,
                    initial_stop=pos.initial_stop, final_stop=pos.current_stop,
                    gross_pnl=pnl["gross_pnl"], fees=pnl["fees"],
                    slippage_cost=pnl["slippage_cost"], net_pnl=pnl["net_pnl"],
                    financing=borrow,
                    return_pct=(pnl["net_pnl"] / entry_notional * 100.0) if entry_notional else 0.0,
                    account_return_pct=(pnl["net_pnl"] / equity_after * 100.0) if equity_after else 0.0,
                    mfe=pos.mfe, mae=pos.mae,
                    duration_s=max(0.0, (fill.ts_ms - pos.entry_ms) / 1000.0),
                    equity_after=equity_after,
                    journal={**pos.journal, "exit_reason": exit_reason})

                self.repo.add_trade(trade)
                self.repo.remove_position(pos.id)
                self.repo.update_account(
                    pos.strategy,
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
                  side=pos.side, gross=pnl["gross_pnl"], fees=pnl["fees"],
                  slippage=pnl["slippage_cost"], financing=borrow,
                  net=pnl["net_pnl"], trade_id=trade.id)
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
            # Favourable means "in the trade's favour". For a short the best
            # excursion is the LOW; scoring it off the high would report every
            # winning short as its worst moment.
            best = hi if p.direction > 0 else lo
            worst = lo if p.direction > 0 else hi
            mfe = max(p.mfe, (best - p.entry_fill_price) * p.qty * p.direction)
            mae = min(p.mae, (worst - p.entry_fill_price) * p.qty * p.direction)
            if (hi != p.highest_price or lo != p.lowest_price
                    or mfe != p.mfe or mae != p.mae):
                self.repo.update_position(p.id, highest_price=hi, lowest_price=lo,
                                          mfe=mfe, mae=mae)
        eq = self.equity(marks)
        acct = self.state
        if eq > float(acct["peak_equity"]):
            self.repo.update_account(self.strategy, peak_equity=eq)

    def roll_daily(self, ts: int, marks: dict[str, float]) -> bool:
        """Start a new UTC trading day. Returns True if a rollover happened."""
        acct = self.state
        today = utc_date(ts)
        if acct["daily_date"] == today:
            return False
        eq = self.equity(marks)
        self.repo.update_account(self.strategy, daily_date=today,
                                 daily_start_equity=eq)
        self.repo.upsert_daily(today, eq, eq)
        log_event("performance", "INFO", "daily rollover", date=today, equity=eq)
        return True

    def set_stop(self, pos: Position, new_stop: float) -> None:
        self.repo.update_position(pos.id, current_stop=new_stop)


class _Abort(Exception):
    pass
