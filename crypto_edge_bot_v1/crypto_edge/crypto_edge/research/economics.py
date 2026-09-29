"""Exit quality on ONE accounting basis, and fee/ATR sensitivity. Research only.

WHY THIS MODULE EXISTS
----------------------
A previous analysis concluded the exit system was working. It did so by
combining `gross_pnl` (P&L BEFORE fees and slippage) with a win/loss count
taken from the performance report -- and that count partitions trades by
`net_pnl > 0`, AFTER costs. Those are different accounting bases, so the
algebra was invalid and the conclusion was withdrawn.

The collision is easy to make because the performance report uses "gross" for
two different things at once:

    summary["gross_pnl"]      = sum of (exit_ref - entry_ref) * qty * d
                                -> BEFORE fees and slippage
    trading["gross_profit"]   = sum of NET P&L over net-winning trades
                                -> AFTER fees and slippage

Both are correct in their own idiom -- the second is the standard profit-factor
sense -- and mixing them is silent. So every figure here carries its basis in
its own name, gross and net are computed side by side from the SAME per-trade
rows, and nothing is ever combined across the two.

THE NUMBER THAT MAKES THE DIFFERENCE CONCRETE
---------------------------------------------
`cost_flipped` counts trades with `gross_pnl > 0` and `net_pnl <= 0`: trades
the market paid for and the costs took back. Its size is exactly how wrong a
gross/net mix-up can be, measured rather than argued.

R-MULTIPLES ARE THE ACTUAL EXIT-QUALITY MEASURE
-----------------------------------------------
"Average winner vs average loser" in dollars confounds exit behaviour with
position size. What an exit system controls is how far a trade runs relative to
the risk it was opened with, which is `pnl / (|entry_fill - initial_stop| *
qty)` -- and every term is a stored column. A system that cuts losers at 0.6R
and rides winners to 2.5R is working whatever the dollars say.

Everything here reads the ledger. None of it models an exit; `policy_sim`
does that from the v8 tape, and that is what should answer exit questions
going forward. This module exists to stop an aggregate shortcut being taken
in the meantime, not to become one.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import FEE_TIER_CUSTOM, KRAKEN_SPOT_TAKER_BPS

# Reported for every scenario so a table can never be read without its basis.
GROSS = "gross"
NET = "net"


def _safe_div(a: float, b: float, default=None):
    return a / b if b else default


# ============================================================ exit quality
@dataclass
class BasisStats:
    """One accounting basis, computed end to end from the same rows.

    `basis` is carried in the data rather than implied by where the caller got
    it, so a row cannot be printed under the wrong heading.
    """
    basis: str
    n: int
    wins: int
    losses: int
    win_rate_pct: float | None
    total: float
    avg_win: float | None
    avg_loss: float | None           # negative
    win_loss_ratio: float | None     # |avg_win / avg_loss|
    expectancy: float | None         # per trade
    profit_factor: float | None
    largest_win: float
    largest_loss: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def basis_stats(values: list[float], basis: str) -> BasisStats:
    """Win/loss statistics over ONE list of per-trade P&L figures.

    The partition and the totals come from the same list, which is the whole
    point: a win rate measured on one basis and a total measured on another do
    not describe the same trades.
    """
    wins = [v for v in values if v > 0]
    losses = [v for v in values if v <= 0]
    n = len(values)
    gp = sum(wins)
    gl = abs(sum(losses))
    avg_w = _safe_div(gp, len(wins))
    avg_l = -_safe_div(gl, len(losses)) if losses else None
    pf = _safe_div(gp, gl, (float("inf") if gp > 0 else None))
    return BasisStats(
        basis=basis, n=n, wins=len(wins), losses=len(losses),
        win_rate_pct=_safe_div(len(wins) * 100.0, n),
        total=sum(values), avg_win=avg_w, avg_loss=avg_l,
        win_loss_ratio=(abs(avg_w / avg_l) if (avg_w and avg_l) else None),
        expectancy=_safe_div(sum(values), n),
        profit_factor=pf,
        largest_win=max(values, default=0.0),
        largest_loss=min(values, default=0.0))


def initial_risk(t) -> float | None:
    """What the trade was opened risking: the distance to its FIRST stop.

    The initial stop, not the final one -- a ratcheted stop measures the trail,
    and dividing by it would make every trailed winner look like a bigger
    multiple of a risk nobody took.
    """
    qty = float(t["qty"])
    d = abs(float(t["entry_fill_price"]) - float(t["initial_stop"]))
    r = d * qty
    return r if r > 0 else None


def r_multiples(trades, field: str) -> list[float]:
    """P&L in units of the risk each trade was opened with."""
    out = []
    for t in trades:
        r = initial_risk(t)
        if r:
            out.append(float(t[field]) / r)
    return out


def exit_quality(repo, strategy: str) -> dict:
    """Gross and net, side by side, never combined."""
    trades = repo.get_trades(strategy)
    gross = [float(t["gross_pnl"]) for t in trades]
    net = [float(t["net_pnl"]) for t in trades]
    flipped = [t for t in trades
               if float(t["gross_pnl"]) > 0 >= float(t["net_pnl"])]
    # net > gross means the costs summed NEGATIVE. That is reachable and
    # legitimate: `slippage_cost` is the signed difference between the fill and
    # the signal reference, and `buy()`/`sell()` base the fill on the live
    # quote, so a favourable tick between the candle close and the fill credits
    # that leg. When the credit exceeds the fees, net lands above gross.
    #
    # An earlier version of this report called such rows impossible and told
    # the operator the ledger was corrupt. It is not: the only real corruption
    # signature is a stored figure that disagrees with what the stored prices
    # produce, which `trade_audit` checks by re-driving `realise_pnl`.
    cost_credit = [t for t in trades
                   if float(t["net_pnl"]) > float(t["gross_pnl"])]
    by_reason: dict[str, dict] = {}
    for t in trades:
        k = str(t["exit_reason"])
        b = by_reason.setdefault(k, {"reason": k, "n": 0, "gross": 0.0,
                                     "net": 0.0, "gross_wins": 0,
                                     "net_wins": 0, "r_gross": []})
        b["n"] += 1
        b["gross"] += float(t["gross_pnl"])
        b["net"] += float(t["net_pnl"])
        b["gross_wins"] += 1 if float(t["gross_pnl"]) > 0 else 0
        b["net_wins"] += 1 if float(t["net_pnl"]) > 0 else 0
        r = initial_risk(t)
        if r:
            b["r_gross"].append(float(t["gross_pnl"]) / r)
    for b in by_reason.values():
        rs = b.pop("r_gross")
        b["avg_r_gross"] = _safe_div(sum(rs), len(rs))
        b["r_measurable"] = len(rs)

    rg, rn = r_multiples(trades, "gross_pnl"), r_multiples(trades, "net_pnl")
    return {
        "closed_trades": len(trades),
        "gross": basis_stats(gross, GROSS).as_dict(),
        "net": basis_stats(net, NET).as_dict(),
        "cost_flipped": len(flipped),
        "cost_flipped_gross": sum(float(t["gross_pnl"]) for t in flipped),
        "net_exceeds_gross": len(cost_credit),
        "net_exceeds_gross_ids": [t["id"] for t in cost_credit],
        "r_gross": basis_stats(rg, GROSS).as_dict(),
        "r_net": basis_stats(rn, NET).as_dict(),
        "r_measurable": len(rg),
        "r_unmeasurable": len(trades) - len(rg),
        "by_exit_reason": sorted(by_reason.values(), key=lambda b: -b["n"]),
        "warning": ("win rates, averages and profit factors are reported"
                    " separately per basis and must never be combined:"
                    " a trade can be a gross win and a net loss"),
    }


# ========================================================== fee scenarios
def tier_bps_table(custom_bps: float) -> dict[str, float]:
    """Every tier this analysis prices, including the configured one."""
    return {FEE_TIER_CUSTOM: float(custom_bps), **KRAKEN_SPOT_TAKER_BPS}


def refee_trade(t, bps_per_side: float) -> dict:
    """One trade re-priced at a different taker rate.

    Fills and slippage are held EXACTLY as recorded. That is the point of the
    exercise: a fee change moves what the venue takes, not where the order
    filled. Re-deriving the fill from the new fee would smuggle a second
    variable into a one-variable comparison, and the sizing feedback it
    implies (a higher fee affords slightly less quantity) is a portfolio
    effect that belongs in a replay, not in a re-pricing.
    """
    qty = float(t["qty"])
    two_leg = qty * float(t["entry_fill_price"]) + qty * float(t["exit_fill_price"])
    fees = two_leg * bps_per_side / 10_000.0
    gross = float(t["gross_pnl"])
    slip = float(t["slippage_cost"])
    fin = float(t["financing"])
    return {"fees": fees, "gross_pnl": gross, "slippage_cost": slip,
            "financing": fin, "net_pnl": gross - slip - fees - fin,
            "two_leg_notional": two_leg,
            "entry_notional": qty * float(t["entry_fill_price"])}


def fee_scenarios(repo, strategy: str, custom_bps: float,
                  tiers: dict | None = None) -> dict:
    """The same closed trades priced at every tier, fills unchanged."""
    trades = repo.get_trades(strategy)
    tiers = tiers if tiers is not None else tier_bps_table(custom_bps)
    turnover = sum(float(t["qty"]) * float(t["entry_fill_price"]) for t in trades)
    two_leg = sum(float(t["qty"]) * (float(t["entry_fill_price"])
                                     + float(t["exit_fill_price"]))
                  for t in trades)
    slippage = sum(float(t["slippage_cost"]) for t in trades)
    financing = sum(float(t["financing"]) for t in trades)
    gross = sum(float(t["gross_pnl"]) for t in trades)
    rows = []
    for name, bps in tiers.items():
        priced = [refee_trade(t, bps) for t in trades]
        net = [p["net_pnl"] for p in priced]
        st = basis_stats(net, NET)
        fees = sum(p["fees"] for p in priced)
        # What gross P&L would have to be for net to reach zero, holding
        # slippage and financing where they are.
        needed = fees + slippage + financing
        rows.append({
            "tier": name, "bps_per_side": bps,
            "total_fees": fees,
            "fee_bps_of_turnover": _safe_div(fees * 10_000.0, turnover),
            "gross_pnl": gross,
            "slippage": slippage, "financing": financing,
            "net_pnl": st.total,
            "expectancy_per_trade": st.expectancy,
            "expectancy_bps": _safe_div(st.total * 10_000.0, turnover),
            "win_rate_pct": st.win_rate_pct,
            "profit_factor": st.profit_factor,
            "avg_win": st.avg_win, "avg_loss": st.avg_loss,
            "breakeven_gross_required": needed,
            "breakeven_gross_bps": _safe_div(needed * 10_000.0, turnover),
            # Only meaningful against a POSITIVE gross edge. With gross <= 0
            # there is no edge to be a multiple of, and printing a negative
            # ratio would read as "nearly there" when the sign is the problem.
            "gross_shortfall_multiple": (_safe_div(needed, gross)
                                         if gross > 0 else None),
            "viable": st.total > 0,
        })
    return {
        "strategy": strategy, "closed_trades": len(trades),
        "entry_turnover": turnover, "two_leg_notional": two_leg,
        "gross_pnl": gross, "slippage": slippage, "financing": financing,
        "gross_bps": _safe_div(gross * 10_000.0, turnover),
        "rows": rows,
        "note": ("fills and slippage are held exactly as recorded; only the"
                 " taker rate moves. A higher fee would also have afforded"
                 " slightly less quantity -- that feedback is a portfolio"
                 " effect and is NOT modelled here"),
    }


# ========================================================== ATR economics
def atr_economics(a_cfg, x_cfg, atr_pcts=(0.25, 0.40, 0.55, 0.70, 1.00, 1.50),
                  tiers: dict | None = None) -> dict:
    """What each ATR band can pay for, at each fee tier. Pure arithmetic.

    Needs no trades at all, which is what makes it the one part of this
    analysis that is not sample-limited. The break-even win rate assumes a pure
    stop-or-target system; the live strategy has nine exits and will not match
    it trade for trade. It still bounds the problem, because every other exit
    resolves somewhere BETWEEN the stop and the target.
    """
    tiers = tiers if tiers is not None else tier_bps_table(
        x_cfg.effective_taker_bps())
    entry_slip = x_cfg.slippage_bps / 100.0        # bps -> percent
    exit_slip_win = x_cfg.slippage_bps / 100.0     # a target exit is quoted
    exit_slip_loss = x_cfg.stop_slippage_bps / 100.0
    slip_win = entry_slip + exit_slip_win
    slip_loss = entry_slip + exit_slip_loss

    rows = []
    for a in atr_pcts:
        stop_pct = a_cfg.stop_atr_mult * a
        target_pct = a_cfg.target_r * a_cfg.stop_atr_mult * a
        per_atr = []
        for name, bps in tiers.items():
            fee_rt = 2.0 * bps / 100.0             # both legs, percent
            win = target_pct - fee_rt - slip_win
            loss = stop_pct + fee_rt + slip_loss
            # Untradeable when a PERFECT trade -- target reached, no adverse
            # excursion -- still loses money. No win rate rescues that.
            untradeable = win <= 0.0
            per_atr.append({
                "tier": name, "bps_per_side": bps,
                "fee_round_trip_pct": fee_rt,
                "slippage_win_pct": slip_win, "slippage_loss_pct": slip_loss,
                "net_target_pct": win, "net_stop_pct": -loss,
                "breakeven_win_rate_pct": (None if untradeable
                                           else loss / (win + loss) * 100.0),
                "cost_share_of_target_pct": (fee_rt + slip_win) / target_pct * 100.0,
                "structurally_untradeable": untradeable,
            })
        rows.append({"atr_pct": a, "stop_pct": stop_pct,
                     "target_pct": target_pct, "tiers": per_atr})

    # The ATR at which the target exactly pays for a winning round trip. Below
    # it, nothing at that tier can work.
    floors = {name: (2.0 * bps / 100.0 + slip_win)
                    / (a_cfg.target_r * a_cfg.stop_atr_mult)
              for name, bps in tiers.items()}
    return {
        "stop_atr_mult": a_cfg.stop_atr_mult, "target_r": a_cfg.target_r,
        "min_atr_pct_configured": a_cfg.min_atr_pct,
        "zero_cost_breakeven_win_rate_pct": 100.0 / (1.0 + a_cfg.target_r),
        "rows": rows, "min_tradable_atr_pct": floors,
        "tiers": dict(tiers),
        "caveat": ("assumes a pure stop-or-target system. The live strategy"
                   " has nine exits, so a realised win rate is not a"
                   " target-hit rate -- this bounds the problem, it does not"
                   " predict a result"),
    }


def breakeven_win_rate(target_pct: float, stop_pct: float,
                       cost_win_pct: float, cost_loss_pct: float):
    """p such that p*(target - cost_win) = (1-p)*(stop + cost_loss)."""
    win = target_pct - cost_win_pct
    loss = stop_pct + cost_loss_pct
    if win <= 0 or (win + loss) <= 0:
        return None
    return loss / (win + loss) * 100.0


__all__ = ["exit_quality", "basis_stats", "BasisStats", "fee_scenarios",
           "refee_trade", "atr_economics", "breakeven_win_rate",
           "r_multiples", "initial_risk", "tier_bps_table", "GROSS", "NET"]


def ledger_provenance(cfg, repo, strategy: str) -> dict:
    """WHICH ledger a report just read, and whether it held anything.

    `db.connect` CREATES a database that is not there, so a research command
    pointed at the wrong working directory reports "0 closed trades" against a
    file it made a moment ago -- indistinguishable, in the output, from a real
    ledger of a strategy that has not traded. An operator reading a fee table
    has no way to tell those apart, and the second one is a finding while the
    first is a mistake.

    So every report states its source: the resolved path, whether the file was
    already populated, and the row counts behind the numbers above it.
    """
    import os

    path = cfg.engine.db_path
    try:
        size = os.path.getsize(path)
    except OSError:
        size = None
    trades = len(repo.get_trades(strategy))
    obs = len(repo.get_observations(strategy=strategy))
    return {
        "db_path": os.path.abspath(path),
        "db_bytes": size,
        "closed_trades": trades,
        "observations": obs,
        # A schema-only file is what `db.connect` leaves behind when it had to
        # create one. Empty of BOTH is the signature of a wrong path.
        "empty": trades == 0 and obs == 0,
    }


def provenance_lines(p: dict, strategy: str) -> list[str]:
    """The source block every report prints under its heading."""
    out = [f"  ledger: {p['db_path']}",
           f"  rows for {strategy}: {p['closed_trades']} closed trade(s), "
           f"{p['observations']} observation(s)"]
    if p["empty"]:
        out += [
            "  !! THIS LEDGER IS EMPTY FOR THIS STRATEGY.",
            "     `db.connect` creates a database that is not there, so this",
            "     may be a file this command just made rather than the one the",
            "     bot trades into. Check the path above, and run from the same",
            "     working directory as the bot (or pass --config).",
        ]
    return out


# ===================================================== per-trade forensics
# How far a recomputed component may sit from the stored one before the row is
# called inconsistent. The ledger stores float64 written through SQLite, so the
# only legitimate difference is representation noise; a cent is orders of
# magnitude above that and still far below any real accounting error.
RECON_TOL = 0.01


def audit_trade(t, tol: float = RECON_TOL) -> dict:
    """Re-derive one trade's P&L from its stored prices and compare.

    This is the test that actually distinguishes a corrupt ledger from a
    surprising-but-correct one. It re-drives the live `realise_pnl` -- the same
    function `close_position` called when the row was written -- rather than
    reimplementing the arithmetic, so a change to the cost model changes this
    audit with it.

    NEGATIVE SLIPPAGE IS NOT AN ERROR, AND THIS IS THE POINT
    --------------------------------------------------------
    `slippage_cost` is not a microstructure cost. It is the signed difference
    between the fill and the REFERENCE, and the reference is the signal
    candle's price while the fill comes from the live book a moment later:

        buy():  base = quote.ask         fill = base * (1 + slippage_bps)
                slip = (fill - ref) * qty
        sell(): base = quote.bid         fill = base * (1 - slippage_bps)
                slip = (ref - fill) * qty

    If the market ticked in the trade's favour between the candle close and the
    fill by more than `slippage_bps`, that leg's slippage is NEGATIVE -- the
    order filled better than the price the signal was measured at. When such a
    credit exceeds the fees, `net > gross`. That is the cost model working as
    designed, not a broken row.

    What is genuinely impossible is a stored value that disagrees with what the
    stored prices produce. That is what `reconciles` reports.
    """
    d = -1 if str(t["side"]) == "short" else 1
    qty = float(t["qty"])
    e_ref, e_fill = float(t["entry_ref_price"]), float(t["entry_fill_price"])
    x_ref, x_fill = float(t["exit_ref_price"]), float(t["exit_fill_price"])
    fees, fin = float(t["fees"]), float(t["financing"])

    # Re-driven, not reimplemented. Fees are stored already summed, so they go
    # in on one leg; `realise_pnl` only ever adds them together.
    from ..execution.paper_broker import realise_pnl
    rp = realise_pnl(e_ref, e_fill, x_ref, x_fill, qty, fees, 0.0, d)
    # `close_position` subtracts financing from net AFTER realise_pnl returns.
    recomputed_net = rp["net_pnl"] - fin

    entry_slip = (e_fill - e_ref) * qty * d
    exit_slip = (x_ref - x_fill) * qty * d
    deltas = {
        "gross_pnl": float(t["gross_pnl"]) - rp["gross_pnl"],
        "slippage_cost": float(t["slippage_cost"]) - rp["slippage_cost"],
        "net_pnl": float(t["net_pnl"]) - recomputed_net,
    }
    reconciles = all(abs(v) <= tol for v in deltas.values())

    legs = []
    if entry_slip < 0:
        legs.append("entry")
    if exit_slip < 0:
        legs.append("exit")
    net_gt_gross = float(t["net_pnl"]) > float(t["gross_pnl"])
    return {
        "id": t["id"], "symbol": t["symbol"], "side": t["side"],
        "entry_ms": int(t["entry_ms"]), "exit_ms": int(t["exit_ms"]),
        "exit_reason": t["exit_reason"], "qty": qty,
        "entry_ref_price": e_ref, "entry_fill_price": e_fill,
        "exit_ref_price": x_ref, "exit_fill_price": x_fill,
        "initial_stop": float(t["initial_stop"]),
        "final_stop": float(t["final_stop"]),
        "stored": {"gross_pnl": float(t["gross_pnl"]), "fees": fees,
                   "slippage_cost": float(t["slippage_cost"]),
                   "financing": fin, "net_pnl": float(t["net_pnl"])},
        "recomputed": {"gross_pnl": rp["gross_pnl"], "fees": rp["fees"],
                       "slippage_cost": rp["slippage_cost"],
                       "financing": fin, "net_pnl": recomputed_net},
        "deltas": deltas,
        "reconciles": reconciles,
        "entry_slippage": entry_slip,
        "exit_slippage": exit_slip,
        "entry_slippage_bps": (entry_slip / (qty * e_fill) * 10_000.0
                               if qty * e_fill else None),
        "exit_slippage_bps": (exit_slip / (qty * x_fill) * 10_000.0
                              if qty * x_fill else None),
        "net_exceeds_gross": net_gt_gross,
        "negative_slippage_legs": legs,
        "verdict": _verdict(reconciles, net_gt_gross, legs, entry_slip,
                            exit_slip, fees),
    }


def _verdict(reconciles, net_gt_gross, legs, entry_slip, exit_slip, fees):
    """Name the cause in the caller's own terms, or say it cannot be named."""
    if not reconciles:
        return ("LEDGER INCONSISTENT -- a stored figure disagrees with what "
                "the stored prices produce. This one IS a real accounting or "
                "schema fault; see the deltas.")
    if not net_gt_gross:
        return "consistent; net <= gross, nothing to explain"
    if not legs:
        return ("net > gross with NO negative slippage leg -- this should be "
                "unreachable and needs investigation")
    total = entry_slip + exit_slip
    where = " and ".join(legs)
    return (f"EXPECTED BEHAVIOUR: the {where} leg filled BETTER than its "
            f"reference, crediting {-total:+.4f} against {fees:.4f} of fees. "
            f"`slippage_cost` is a signed fill-vs-reference difference, not a "
            f"one-way cost, so a favourable tick larger than the fees puts net "
            f"above gross. Not a bug.")


def trade_audit(repo, strategy: str, tol: float = RECON_TOL) -> dict:
    """Audit every closed trade; surface the ones worth a human's attention."""
    trades = repo.get_trades(strategy)
    rows = [audit_trade(t, tol) for t in trades]
    return {
        "strategy": strategy, "closed_trades": len(rows), "tolerance": tol,
        "reconciled": sum(1 for r in rows if r["reconciles"]),
        "failed_reconciliation": [r for r in rows if not r["reconciles"]],
        "net_exceeds_gross": [r for r in rows if r["net_exceeds_gross"]],
        "negative_entry_leg": sum(1 for r in rows if r["entry_slippage"] < 0),
        "negative_exit_leg": sum(1 for r in rows if r["exit_slippage"] < 0),
        "rows": rows,
    }
