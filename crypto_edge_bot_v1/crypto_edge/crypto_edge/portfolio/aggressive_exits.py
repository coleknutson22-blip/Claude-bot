"""Deterministic exits for Strategy B. Every one is price and clock only.

WHY DETERMINISM IS THE POINT
----------------------------
The bot this strategy is modelled on delegated its time-based exit to a local
LLM. When that model was unreachable `ollama_json` returned `None`, no exit
fired, and the position was simply held -- indefinitely, with no error. An exit
path that can fail open is not an exit path.

So every exit here is arithmetic on closed candles and a timestamp. No model, no
network, no optional dependency. The brain layer, when it arrives, may not
participate in any of this.

R IS SIGNED BY DIRECTION THROUGHOUT
-----------------------------------
`R` is the initial risk per unit: `(entry - initial_stop) * direction`, which is
positive on both sides because a short's stop sits ABOVE its entry. Progress is
`(price - entry) * direction / R`. Writing it this way once means the long and
short paths are the same code rather than two implementations that can drift.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..models import Position

# Exit reasons. Strings rather than an enum because they are written straight
# into the trade ledger and read back by research queries.
STOP = "stop"
TARGET = "target"
MOMENTUM = "momentum_invalidation"
TIME = "time_stop"
TIME_EARLY = "time_stop_no_progress"
REGIME = "hostile_regime"
FORCED_SHORT = "forced_close_collateral"

HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS


@dataclass
class StopUpdate:
    new_stop: float
    changed: bool
    kind: str = ""
    pct_move: float = 0.0


def risk_per_unit(pos: Position) -> float:
    """R: the distance from entry to the INITIAL stop. Positive on both sides."""
    return (pos.entry_fill_price - pos.initial_stop) * pos.direction


def progress_r(pos: Position, price: float) -> float:
    r = risk_per_unit(pos)
    if r <= 0:
        return 0.0
    return (price - pos.entry_fill_price) * pos.direction / r


def target_price(pos: Position, target_r: float) -> float:
    """Where the profit target sits. Above entry for a long, below for a short."""
    return pos.entry_fill_price + target_r * risk_per_unit(pos) * pos.direction


def breakeven_stop(pos: Position, offset_r: float) -> float:
    return pos.entry_fill_price + offset_r * risk_per_unit(pos) * pos.direction


def chandelier_stop(extreme: float, atr: float, mult: float,
                    direction: int) -> float:
    """Trail from the best price the trade has seen, `mult` ATRs behind it."""
    return extreme - atr * mult * (1 if direction >= 0 else -1)


def is_better_stop(candidate: float, current: float, direction: int) -> bool:
    """A stop only ever moves in the direction that reduces risk.

    Up for a long, DOWN for a short. Never the other way -- a stop that can
    loosen is not a stop, and the ratchet is what makes the initial risk figure
    an actual ceiling rather than an opening bid.
    """
    return (candidate > current) if direction >= 0 else (candidate < current)


def update_stop(pos: Position, price: float, atr: float, *,
                breakeven_at_r: float, breakeven_offset_r: float,
                trail_start_r: float, trail_atr_mult: float) -> StopUpdate:
    """Breakeven ratchet, then chandelier trail. Most protective wins."""
    candidate, kind = pos.current_stop, ""
    r = risk_per_unit(pos)
    if r <= 0:
        return StopUpdate(pos.current_stop, False)
    made = progress_r(pos, price)

    if breakeven_at_r > 0 and made >= breakeven_at_r:
        be = breakeven_stop(pos, breakeven_offset_r)
        if is_better_stop(be, candidate, pos.direction):
            candidate, kind = be, "breakeven"

    # The trail only starts once the trade is genuinely in profit. Trailing from
    # the first tick would convert normal noise into an exit and turn every
    # position into a scratch.
    if atr and atr > 0 and made >= trail_start_r:
        extreme = pos.highest_price if pos.direction >= 0 else pos.lowest_price
        extreme = (max(extreme, price) if pos.direction >= 0
                   else min(extreme, price))
        chand = chandelier_stop(extreme, atr, trail_atr_mult, pos.direction)
        if is_better_stop(chand, candidate, pos.direction):
            candidate, kind = chand, "chandelier"

    if not is_better_stop(candidate, pos.current_stop, pos.direction):
        return StopUpdate(pos.current_stop, False)
    pct = (abs(candidate - pos.current_stop) / pos.current_stop * 100.0
           if pos.current_stop else 0.0)
    return StopUpdate(candidate, True, kind, pct)


# ------------------------------------------------------------- financing
def financing_cost(pos: Position, now_ms: int, bps_per_day: float) -> float:
    """Simulated borrow on a short, accrued by elapsed time. Longs pay nothing.

    Computed from the timestamps rather than accumulated into a column: there
    is no accrual state to get out of step with the clock, a restart cannot
    lose or double-count it, and the figure is identical however many times it
    is asked for.
    """
    if pos.direction >= 0 or bps_per_day <= 0:
        return 0.0
    held_ms = max(0, now_ms - pos.entry_ms)
    days = held_ms / DAY_MS
    return pos.entry_notional * (bps_per_day / 10_000.0) * days


def short_loss_fraction(pos: Position, price: float, now_ms: int = 0,
                        bps_per_day: float = 0.0) -> float:
    """How much of a short's collateral the loss has eaten, as a fraction.

    A long cannot lose more than it paid -- price stops at zero. A SHORT has no
    such floor: the loss grows without limit as price rises, and at 1x it
    exceeds the collateral once price has merely doubled. Nothing in a paper
    ledger stops that on its own, so this is what the forced close watches.
    """
    if pos.direction >= 0 or pos.margin_held <= 0:
        return 0.0
    loss = -pos.unrealized(price) + financing_cost(pos, now_ms, bps_per_day)
    return max(0.0, loss / pos.margin_held)


def force_close_short(pos: Position, price: float, *, at_loss_pct: float,
                      now_ms: int = 0, bps_per_day: float = 0.0) -> bool:
    if pos.direction >= 0 or at_loss_pct <= 0:
        return False
    return short_loss_fraction(pos, price, now_ms, bps_per_day) >= at_loss_pct / 100.0


# ------------------------------------------------------------- exit check
def check_exit(pos: Position, price: float, *, now_ms: int, cfg,
               ema_struct: float | None = None,
               btc_regime: str = "unknown") -> str | None:
    """The non-stop exits, in order of severity. Returns a reason or None.

    The protective STOP itself is resolved by the execution layer against the
    candle's high/low, because a stop must fill at the price it was touched at,
    not at a close that may be far beyond it.
    """
    # 1. Collateral. Checked first because it is the only one that is about
    #    solvency rather than about the trade thesis.
    if force_close_short(pos, price,
                         at_loss_pct=cfg.short_force_close_at_loss_pct,
                         now_ms=now_ms,
                         bps_per_day=cfg.short_borrow_bps_per_day):
        return FORCED_SHORT

    made = progress_r(pos, price)
    if cfg.target_r > 0 and made >= cfg.target_r:
        return TARGET

    # 2. The thesis is gone: fast structure has flipped against the position.
    if (ema_struct is not None and cfg.min_ema_struct_15m > 0
            and ema_struct * pos.direction <= -cfg.min_ema_struct_15m):
        return MOMENTUM

    # 3. The market itself turned against this side.
    if cfg.exit_on_hostile_regime:
        if pos.direction > 0 and btc_regime == "bear":
            return REGIME
        if pos.direction < 0 and btc_regime == "bull":
            return REGIME

    # 4. Time. An intraday trade that has gone nowhere is capital and a slot
    #    doing nothing, and this strategy has only three slots.
    held_h = max(0, now_ms - pos.entry_ms) / HOUR_MS
    if cfg.time_stop_hours > 0 and held_h >= cfg.time_stop_hours:
        return TIME
    if (cfg.time_stop_early_hours > 0 and held_h >= cfg.time_stop_early_hours
            and made < cfg.time_stop_min_r):
        return TIME_EARLY
    return None
