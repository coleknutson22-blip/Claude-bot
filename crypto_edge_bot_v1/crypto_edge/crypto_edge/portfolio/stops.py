"""Stop management: initial ATR stop, chandelier trail, breakeven ratchet,
time stop. Stops only ever move UP for a long -- a ratchet, never a give-back.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import StrategyCfg
from ..models import Position


@dataclass
class StopUpdate:
    new_stop: float
    changed: bool
    kind: str = ""
    pct_move: float = 0.0


def initial_stop(entry_price: float, atr: float, mult: float) -> float:
    return entry_price - atr * mult


def chandelier_stop(highest_price: float, atr: float, mult: float) -> float:
    return highest_price - atr * mult


def breakeven_stop(entry_price: float, initial_stop: float, offset_r: float) -> float:
    r = entry_price - initial_stop
    return entry_price + offset_r * r


def update_stop(pos: Position, price: float, atr: float, cfg: StrategyCfg,
                min_notify_pct: float = 0.0) -> StopUpdate:
    """Compute the new stop for an open long.

    Order matters: we take the most protective of (existing, breakeven,
    chandelier) and never move a stop down.
    """
    candidate = pos.current_stop
    kind = ""

    r = pos.entry_fill_price - pos.initial_stop
    if r > 0 and cfg.breakeven_at_r > 0:
        if (price - pos.entry_fill_price) >= cfg.breakeven_at_r * r:
            be = breakeven_stop(pos.entry_fill_price, pos.initial_stop,
                                cfg.breakeven_offset_r)
            if be > candidate:
                candidate, kind = be, "breakeven"

    if atr and atr > 0:
        high = max(pos.highest_price, price)
        chand = chandelier_stop(high, atr, cfg.trail_atr_mult)
        if chand > candidate:
            candidate, kind = chand, "chandelier"

    if candidate <= pos.current_stop:
        return StopUpdate(pos.current_stop, False)

    pct = (candidate - pos.current_stop) / pos.current_stop * 100.0 if pos.current_stop else 0.0
    # `changed` always reflects reality; `pct_move` lets the notifier decide
    # whether the move is worth a message, so Telegram is not spammed.
    return StopUpdate(candidate, True, kind, pct)


def time_stop_hit(pos: Position, now_ms: int, price: float, cfg: StrategyCfg,
                  bar_ms: int) -> bool:
    """Exit stale trades that never went anywhere, freeing capital and risk slots."""
    if cfg.time_stop_bars <= 0:
        return False
    bars_held = (now_ms - pos.entry_ms) / bar_ms
    if bars_held < cfg.time_stop_bars:
        return False
    r = pos.entry_fill_price - pos.initial_stop
    if r <= 0:
        return True
    progress_r = (price - pos.entry_fill_price) / r
    return progress_r < cfg.time_stop_min_r
