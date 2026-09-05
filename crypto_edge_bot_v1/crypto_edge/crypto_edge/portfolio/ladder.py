"""The three-position capital ladder, and the risk cap that overrules it.

THE LADDER
----------
Each slot may use a percentage of the cash still FREE at the time it is opened:

    slot 1 -> 50% of free cash
    slot 2 -> 75% of what remains
    slot 3 -> 100% of what remains

At full confidence that deploys the entire balance across three positions. Two
consequences worth being explicit about, because both follow from the design
rather than from a bug:

  * at high confidence the account ends fully deployed with no cash buffer;
  * when early slots are small, later ceilings are computed on a barely-touched
    base, so slot 3 can be LARGER than slot 1. At the 40% multiplier the
    sequence is 2000 / 2400 / 2240 on a 10,000 balance.

THE LADDER PROPOSES, RISK DISPOSES
----------------------------------
A percentage of cash says nothing about how much can be LOST. The same
allocation is a 0.5% or a 5% account risk depending on where the stop sits, so
the ladder's number is only ever a ceiling. The final size is the smallest of
four independent limits, and which one bound is recorded on every trade:

    notional = min(ladder x confidence,   <- what the ladder offers
                   risk cap / stop%,      <- what the loss ceiling permits
                   exposure room,         <- portfolio limit
                   affordable cash)       <- what is actually there

THE RISK CAP TIGHTENS AS THE DAY GOES BADLY
-------------------------------------------
    max_loss = min(max_loss_pct of equity,
                   daily_buffer_fraction of the REMAINING daily buffer)

The second term is the idea worth keeping from the source bot: a flat 1% of
equity is the same size on a flat day and on a day already down 2.5% of a 3%
limit. Scaling to what is left means the last trade before a daily halt cannot
be the one that causes it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Which limit produced the final size. Recorded on every sized trade so
# "is the risk cap doing anything, or is the ladder always the binding one?"
# is a single query rather than a reconstruction.
LADDER, RISK_CAP, EXPOSURE, CASH, MIN_NOTIONAL, NONE = (
    "ladder", "risk_cap", "exposure", "cash", "min_notional", "none")


@dataclass
class LadderPlan:
    slot: int                       # 1-based; 0 means no slot available
    ceiling_cash: float = 0.0       # what the ladder offers this slot
    multiplier: float = 0.0         # confidence allocation multiplier
    target_notional: float = 0.0    # ceiling x multiplier x leverage
    max_loss_cash: float = 0.0      # the hard per-trade loss ceiling
    risk_notional: float = 0.0      # what that ceiling permits, given the stop
    exposure_room: float = 0.0
    affordable: float = 0.0
    notional: float = 0.0           # the answer
    binding_constraint: str = NONE
    reason: str = ""                # set only when nothing can be traded
    detail: dict = field(default_factory=dict)

    @property
    def tradable(self) -> bool:
        return self.notional > 0.0


def ceiling_for_slot(free_cash: float, n_open: int,
                     ceilings_pct: list[float]) -> tuple[int, float]:
    """The cash available to the next position.

    `free_cash` is the strategy's OWN uncommitted cash -- the ladder percentages
    are applied to what is left after the positions already open, which is what
    makes each successive slot smaller in absolute terms at equal confidence.
    """
    if n_open < 0 or n_open >= len(ceilings_pct):
        return 0, 0.0
    if free_cash <= 0:
        return n_open + 1, 0.0
    pct = float(ceilings_pct[n_open])
    return n_open + 1, free_cash * pct / 100.0


def loss_ceiling(equity: float, daily_buffer_remaining: float, *,
                 max_loss_pct: float, daily_buffer_fraction: float) -> float:
    """The most this trade is allowed to lose if the stop fills.

    Both terms are floors on recklessness, and the tighter one wins.
    """
    by_equity = max(0.0, equity) * max_loss_pct / 100.0
    by_buffer = max(0.0, daily_buffer_remaining) * daily_buffer_fraction
    return min(by_equity, by_buffer)


def plan(*, free_cash: float, equity: float, n_open: int, confidence: float,
         multiplier: float, entry_price: float, stop_price: float,
         direction: int, exposure: float, daily_buffer_remaining: float,
         ceilings_pct: list[float], max_loss_pct: float,
         daily_buffer_fraction: float, leverage: float = 1.0,
         max_exposure_pct: float = 100.0, min_notional: float = 0.0,
         max_positions: int = 3) -> LadderPlan:
    """Size one position. Pure arithmetic: no account, no clock, no I/O."""
    slot, ceiling = ceiling_for_slot(free_cash, n_open, ceilings_pct[:max_positions])
    p = LadderPlan(slot=slot, ceiling_cash=ceiling, multiplier=multiplier)

    if slot == 0 or n_open >= max_positions:
        p.reason = f"all {max_positions} position slots are in use"
        return p
    if multiplier <= 0.0:
        p.reason = f"confidence {confidence:.1f} is below the trading floor"
        return p
    if entry_price <= 0 or not math.isfinite(entry_price):
        p.reason = "invalid entry price"
        return p

    d = 1 if direction >= 0 else -1
    stop_distance = (entry_price - stop_price) * d
    if stop_distance <= 0 or not math.isfinite(stop_distance):
        p.reason = ("stop is on the wrong side of entry for a "
                    f"{'long' if d > 0 else 'short'}")
        return p
    stop_pct = stop_distance / entry_price

    # --- what each limit permits -----------------------------------------
    p.target_notional = ceiling * multiplier * max(0.0, leverage)
    p.max_loss_cash = loss_ceiling(
        equity, daily_buffer_remaining,
        max_loss_pct=max_loss_pct, daily_buffer_fraction=daily_buffer_fraction)
    p.risk_notional = p.max_loss_cash / stop_pct if stop_pct > 0 else 0.0
    p.exposure_room = max(0.0, equity * max_exposure_pct / 100.0 - exposure)
    # Collateral is posted from cash, so leverage buys notional per unit of it.
    p.affordable = max(0.0, free_cash) * max(1.0, leverage)

    limits = {
        LADDER: p.target_notional,
        RISK_CAP: p.risk_notional,
        EXPOSURE: p.exposure_room,
        CASH: p.affordable,
    }
    # Ties are broken by INTENT, not alphabetically. The last slot's ceiling is
    # "100% of what remains", which is numerically identical to the cash on
    # hand every time -- reporting that as a cash constraint would say the
    # account ran out of money when in fact the ladder was doing exactly what
    # it was designed to do, and would quietly overstate cash pressure in the
    # research record.
    priority = {LADDER: 0, RISK_CAP: 1, EXPOSURE: 2, CASH: 3}
    p.binding_constraint = min(limits, key=lambda k: (limits[k], priority[k]))
    p.notional = max(0.0, limits[p.binding_constraint])
    p.detail = dict(limits)
    p.detail["stop_pct"] = stop_pct * 100.0

    if p.notional <= 0.0:
        p.reason = f"{p.binding_constraint} allows nothing"
        return p
    if min_notional and p.notional < min_notional:
        p.binding_constraint = MIN_NOTIONAL
        p.reason = (f"size {p.notional:,.2f} is below the venue minimum "
                    f"{min_notional:,.2f}")
        p.notional = 0.0
    return p


def expected_loss_pct(notional: float, stop_pct: float, equity: float) -> float:
    """Account risk at the stop, as a percentage. What the cap constrains."""
    if equity <= 0:
        return 0.0
    return notional * stop_pct / 100.0 / equity * 100.0
