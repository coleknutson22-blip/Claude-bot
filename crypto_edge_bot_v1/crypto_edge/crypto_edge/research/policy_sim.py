"""Offline exit-policy replay. Research only -- nothing here can trade.

THE ONE DESIGN DECISION THAT MATTERS
------------------------------------
This module does not REIMPLEMENT the exit engine. It re-drives it. Every bar
calls the same `paper_broker.stop_exit`, the same `aggressive_exits.check_exit`,
the same `aggressive_exits.update_stop` and the same `realise_pnl` that the live
runtime calls, against a real `Position` object.

A reimplementation would be a second copy of the rules, and the first thing it
would do is drift: someone changes the live trail and the simulator quietly
keeps answering with the old one. Re-driving means a change to the live exit
engine changes this replay too, and the reconciliation below would catch the
day it does not.

HOW TP2 IS EXPRESSED, AND WHY IT IS NOT NEW CODE
------------------------------------------------
`check_exit` fires the target on `progress_r(pos, close) >= cfg.target_r`, where
R is the initial stop distance. So a FIXED +2% target is exactly

    target_r = 2.0 / stop_distance_pct

because R = entry x stop_pct/100, and the threshold becomes entry x 0.02 --
`entry * 1.02` for a long and `entry * 0.98` for a short, evaluated on the
close, identical in mechanism to CONTROL. TP2 therefore runs the SAME code with
one number changed per path. There is no second target implementation to get
subtly wrong, and nothing else in the exit set reads `target_r`.

WHAT THE REPLAY CANNOT REPRODUCE, STATED UP FRONT
-------------------------------------------------
The live non-stop exits price their fill against a LIVE QUOTE fetched at the
moment of exit. Quotes are not on the tape and cannot be reconstructed, so the
replay prices them with `quote=None` -- which is a real live code path (the one
taken when the quote is missing or structurally broken), not an invention. The
cost is roughly half the book spread on non-stop exits, and it is reported as a
named reconciliation source rather than buried in a tolerance.

Stop exits involve no quote at all and should reconcile essentially exactly.
That split is the most informative thing in the reconciliation report.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..execution.paper_broker import realise_pnl
from ..models import Candle, MarketMeta, Position
from ..portfolio import aggressive_exits as ex

BAR_MS = 300_000
MIN_MS = 60_000

CONTROL, TP2 = "CONTROL", "TP2"

# Why a path could not be replayed. Never silently dropped, never reconstructed.
UNREPLAYABLE_NO_TAPE = "UNREPLAYABLE_NO_TAPE"
UNREPLAYABLE_OPEN = "UNREPLAYABLE_OPEN_PATH"
UNREPLAYABLE_SHORT_TAPE = "UNREPLAYABLE_TAPE_TOO_SHORT"
NO_EXIT_IN_WINDOW = "no_exit_within_tape"


@dataclass
class SimResult:
    """One policy's outcome on one signal."""
    policy: str
    exit_reason: str = ""
    exit_ms: int = 0
    exit_bar_index: int = -1
    exit_ref_price: float = 0.0      # the price the exit was decided at
    exit_fill_price: float = 0.0     # after slippage and venue rounding
    hold_minutes: float = 0.0
    gross_pnl: float = 0.0
    fees: float = 0.0
    slippage_cost: float = 0.0
    financing: float = 0.0
    net_pnl: float = 0.0
    gross_return_pct: float = 0.0
    net_return_pct: float = 0.0
    r_multiple: float = 0.0
    final_stop: float = 0.0
    bars_walked: int = 0
    unreplayable: str = ""

    @property
    def ok(self) -> bool:
        return not self.unreplayable


@dataclass
class PairedRecord:
    """CONTROL and TP2 on the SAME signal, plus the difference between them."""
    observation_id: str
    symbol: str
    side: str
    signal_ms: int
    setup_score: float | None
    confidence: float | None
    atr_pct: float | None
    stop_distance_pct: float
    control: SimResult | None = None
    tp2: SimResult | None = None
    unreplayable: str = ""

    @property
    def ok(self) -> bool:
        return (not self.unreplayable and self.control is not None
                and self.tp2 is not None and self.control.ok and self.tp2.ok)

    @property
    def net_diff_pct(self) -> float:
        """TP2 minus CONTROL, in net return percentage points."""
        if not self.ok:
            return 0.0
        return self.tp2.net_return_pct - self.control.net_return_pct

    @property
    def winner(self) -> str:
        if not self.ok:
            return ""
        d = self.net_diff_pct
        if abs(d) < 1e-9:
            return "tie"
        return TP2 if d > 0 else CONTROL

    def as_dict(self) -> dict:
        def side(r):
            return {} if r is None else {
                "exit_reason": r.exit_reason, "exit_ms": r.exit_ms,
                "exit_price": r.exit_fill_price,
                "hold_minutes": r.hold_minutes,
                "gross_return_pct": r.gross_return_pct,
                "net_return_pct": r.net_return_pct,
                "r_multiple": r.r_multiple}
        return {
            "observation_id": self.observation_id, "symbol": self.symbol,
            "side": self.side, "signal_ms": self.signal_ms,
            "setup_score": self.setup_score, "confidence": self.confidence,
            "atr_pct": self.atr_pct,
            "stop_distance_pct": self.stop_distance_pct,
            "CONTROL": side(self.control), "TP2": side(self.tp2),
            "DIFFERENCE": {"tp2_minus_control_net_pct": self.net_diff_pct,
                           "winner": self.winner},
            "unreplayable": self.unreplayable,
        }


# --------------------------------------------------------------- policies
def control_cfg(cfg):
    """The shipped configuration, untouched."""
    return cfg


def tp2_cfg(cfg, stop_distance_pct: float, fixed_tp_pct: float = 2.0):
    """The shipped configuration with ONE number changed: the target.

    Expressed as the R multiple that equals `fixed_tp_pct` for this path's stop,
    so the fixed target runs through the identical `check_exit` branch on the
    identical close-based comparison.
    """
    out = copy.copy(cfg)
    out.target_r = (fixed_tp_pct / stop_distance_pct
                    if stop_distance_pct > 0 else 0.0)
    return out


def fixed_target_price(entry: float, direction: int,
                       fixed_tp_pct: float = 2.0) -> float:
    """entry x 1.02 for a long, entry x 0.98 for a short. For tests and reports."""
    return entry * (1.0 + (fixed_tp_pct / 100.0) * (1 if direction >= 0 else -1))


# ----------------------------------------------------------------- replay
def _meta_from(d: dict | None, symbol: str) -> MarketMeta | None:
    if not d:
        return None
    return MarketMeta(
        symbol, symbol.split("/")[0], symbol.partition("/")[2] or "USD", True,
        amount_precision=int(d.get("amount_precision") or 8),
        price_precision=int(d.get("price_precision") or 8),
        min_amount=float(d.get("min_amount") or 0.0),
        min_cost=float(d.get("min_cost") or 0.0),
        amount_step=float(d.get("amount_step") or 0.0),
        price_step=float(d.get("price_step") or 0.0))


def _position(*, symbol, side, qty, entry_ref, entry_fill, entry_ms,
              initial_stop, entry_fee=0.0, entry_slippage=0.0,
              margin_held=None, leverage=1.0) -> Position:
    notional = qty * entry_fill
    return Position(
        id="sim", symbol=symbol, strategy="sim", strategy_version="sim",
        side=side, qty=qty, entry_ref_price=entry_ref,
        entry_fill_price=entry_fill, entry_ms=entry_ms, entry_fee=entry_fee,
        entry_slippage=entry_slippage, initial_stop=initial_stop,
        current_stop=initial_stop, highest_price=entry_fill,
        lowest_price=entry_fill, risk_amount=0.0, candle_id="sim",
        journal={},
        margin_held=(notional / max(1.0, leverage) if margin_held is None
                     else margin_held))


def replay(tape: list[dict], *, cfg, policy: str, symbol: str, side: str,
           entry_ref: float, entry_fill: float, entry_ms: int,
           initial_stop: float, qty: float = 1.0, entry_fee: float = 0.0,
           entry_slippage: float = 0.0, meta: dict | None = None,
           broker=None, fixed_tp_pct: float = 2.0,
           margin_held: float | None = None) -> SimResult:
    """Walk one tape under one exit policy, re-driving the live functions.

    The per-bar order is the live order, and it is not negotiable:

        1. the protective STOP, intrabar against the candle, using the stop as
           it stood at the START of this bar
        2. `check_exit` on the bar CLOSE -- which internally orders
           forced-short, target, momentum, regime, time, early-time
        3. the stop RATCHET, which therefore only takes effect from the NEXT
           bar, exactly as it does live

    Getting 3 before 2 would let a trade be stopped out on the same bar that
    moved its stop, which the live engine never does.
    """
    if broker is None:
        raise ValueError(
            "replay needs the broker built from the LIVE execution config -- "
            "see broker_from(). Defaulting the cost model here would let the "
            "two policies be priced differently from the live ledger.")
    res = SimResult(policy=policy)
    if not tape:
        res.unreplayable = UNREPLAYABLE_NO_TAPE
        return res

    d = -1 if side == "short" else 1
    stop_pct = abs(entry_ref - initial_stop) / entry_ref * 100.0 if entry_ref else 0.0
    a = tp2_cfg(cfg, stop_pct, fixed_tp_pct) if policy == TP2 else control_cfg(cfg)
    m = _meta_from(meta, symbol)
    pos = _position(symbol=symbol, side=side, qty=qty, entry_ref=entry_ref,
                    entry_fill=entry_fill, entry_ms=entry_ms,
                    initial_stop=initial_stop, entry_fee=entry_fee,
                    entry_slippage=entry_slippage, margin_held=margin_held,
                    leverage=getattr(cfg, "leverage", 1.0))

    for i, b in enumerate(tape):
        bar_end = int(b["open_ms"]) + BAR_MS
        # --- 1. protective stop, against the candle -----------------------
        candle = Candle(int(b["open_ms"]), float(b["open"]), float(b["high"]),
                        float(b["low"]), float(b["close"]), 0.0)
        fill = broker.stop_exit(symbol, qty, pos.current_stop, candle, m,
                                ts_ms=bar_end, direction=d)
        if fill is not None:
            reason = ex.STOP if fill.reason == "stop" else "stop_gap"
            return _settle(res, pos, fill, reason, i, bar_end, a, stop_pct)

        close = float(b["close"])
        # --- 2. the deterministic exit set, on the CLOSE ------------------
        reason = ex.check_exit(pos, close, now_ms=bar_end, cfg=a,
                               ema_struct=b.get("ema_struct_15m"),
                               btc_regime=b.get("btc_regime") or "unknown")
        if reason:
            # `quote=None` is the live path taken when no trustworthy quote is
            # available. See the module docstring: quotes are not on the tape.
            exit_fill = broker.exit_fill(symbol, qty, close, d, None, m,
                                         ts_ms=bar_end, reason=reason)
            return _settle(res, pos, exit_fill, reason, i, bar_end, a, stop_pct)

        # --- 3. ratchet the stop, effective from the NEXT bar -------------
        upd = ex.update_stop(pos, close, float(b.get("atr") or 0.0),
                             breakeven_at_r=a.breakeven_at_r,
                             breakeven_offset_r=a.breakeven_offset_r,
                             trail_start_r=a.trail_start_r,
                             trail_atr_mult=a.trail_atr_mult)
        pos = _advanced(pos, upd.new_stop if upd.changed else pos.current_stop,
                        float(b["high"]), float(b["low"]))

    res.unreplayable = NO_EXIT_IN_WINDOW
    res.bars_walked = len(tape)
    res.final_stop = pos.current_stop
    return res


def _advanced(pos: Position, new_stop: float, high: float,
              low: float) -> Position:
    """Carry the running extremes forward -- the chandelier trails from them."""
    return Position(**{**pos.__dict__,
                       "current_stop": new_stop,
                       "highest_price": max(pos.highest_price, high),
                       "lowest_price": min(pos.lowest_price, low)})


def _settle(res: SimResult, pos: Position, fill, reason: str, i: int,
            bar_end: int, cfg, stop_pct: float) -> SimResult:
    """Book the exit with the SAME P&L decomposition the live ledger uses."""
    pnl = realise_pnl(pos.entry_ref_price, pos.entry_fill_price, fill.ref_price,
                      fill.fill_price, pos.qty, pos.entry_fee, fill.fee,
                      direction=pos.direction)
    borrow = ex.financing_cost(pos, fill.ts_ms,
                               getattr(cfg, "short_borrow_bps_per_day", 0.0))
    notional = pos.qty * pos.entry_fill_price
    res.exit_reason = reason
    res.exit_ms = int(fill.ts_ms)
    res.exit_bar_index = i
    res.exit_ref_price = float(fill.ref_price)
    res.exit_fill_price = float(fill.fill_price)
    res.hold_minutes = (fill.ts_ms - pos.entry_ms) / MIN_MS
    res.gross_pnl = pnl["gross_pnl"]
    res.fees = pnl["fees"]
    res.slippage_cost = pnl["slippage_cost"]
    res.financing = borrow
    res.net_pnl = pnl["net_pnl"] - borrow
    res.gross_return_pct = pnl["gross_pnl"] / notional * 100.0 if notional else 0.0
    res.net_return_pct = res.net_pnl / notional * 100.0 if notional else 0.0
    risk = ex.risk_per_unit(pos) * pos.qty
    res.r_multiple = res.net_pnl / risk if risk > 0 else 0.0
    res.final_stop = pos.current_stop
    res.bars_walked = i + 1
    return res


def broker_from(execution_cfg):
    """The broker the live runtime uses, built from the same execution config.

    Both policies are priced by ONE broker instance, so a fee or slippage
    change cannot reach one arm and not the other.
    """
    from ..execution.paper_broker import PaperBroker
    return PaperBroker(execution_cfg.taker_fee_bps, execution_cfg.slippage_bps,
                       execution_cfg.stop_slippage_bps,
                       execution_cfg.use_book_spread,
                       execution_cfg.max_spread_bps_entry)
