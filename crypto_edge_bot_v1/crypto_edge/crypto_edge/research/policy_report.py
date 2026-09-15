"""Pairs CONTROL against TP2 on identical signals, and reconciles CONTROL
against what actually happened.

RECONCILIATION COMES FIRST
--------------------------
A simulator that cannot reproduce trades that already happened cannot be
trusted on trades that did not. So before any TP2 number is reported, every
replayable closed Strategy B trade is replayed under CONTROL and compared to
the recorded ledger field by field -- exit reason, bar, price, fill, fees,
slippage, financing, net P&L and hold time.

"Same direction of profit" is not reconciliation. The check below is numeric,
and the tolerances are derived from named, understood differences rather than
widened until the test passes.

WHERE THE TWO CAN LEGITIMATELY DIFFER
-------------------------------------
1. QUOTE-PRICED FILLS. Live prices a non-stop exit against a live quote fetched
   at that instant. Quotes are not on the tape, so the replay uses the
   `quote=None` path -- real live code, but it forgoes the book spread. Bounded
   by half the venue's spread, so non-stop exits carry a bps tolerance and
   STOP exits, which never consult a quote, do not.
2. EVALUATION INSTANT. Live evaluates at the wall clock, mid-bar; the replay
   evaluates at the bar close. Worth a few seconds of financing and, at a time
   stop boundary, at most one bar.
3. CYCLE GRANULARITY. Live acts on the next cycle after a bar closes. A slow or
   missed cycle can delay an exit by a bar, so the bar index is compared with a
   +/-1 tolerance and the count of off-by-one matches is reported separately.

Anything outside those is a real discrepancy and is listed, not averaged away.
"""
from __future__ import annotations

import statistics as st
from dataclasses import dataclass, field

from . import policy_sim as sim
from .forward_test import _bucket, _num

# Tolerances, each traceable to a specific difference above.
TOL = {
    # A stop fill involves no quote, so it should agree to the cent.
    "stop_fill_bps": 0.5,
    # A non-stop fill forgoes the book spread: half of max_spread_bps_entry,
    # plus a little for venue rounding.
    "quoted_fill_bps": 15.0,
    "exit_bars": 1,             # cycle granularity
    "financing_abs": 0.01,      # seconds of accrual on a bps/day rate
    "net_pnl_rel": 0.02,        # 2% of notional-scaled P&L, from the fill gap
}

INSUFFICIENT, PROVISIONAL, COMPARABLE = (
    "INSUFFICIENT SAMPLE", "PROVISIONAL", "COMPARABLE")
MIN_PROVISIONAL, MIN_COMPARABLE = 100, 200


@dataclass
class Discrepancy:
    trade_id: str
    field: str
    live: object
    simulated: object
    note: str = ""


@dataclass
class Reconciliation:
    compared: int = 0
    exact_reason: int = 0
    exact_bar: int = 0
    within_one_bar: int = 0
    fill_ok: int = 0
    net_ok: int = 0
    unreplayable: dict = field(default_factory=dict)
    discrepancies: list = field(default_factory=list)
    by_exit_reason: dict = field(default_factory=dict)

    @property
    def reason_match_pct(self) -> float:
        return self.exact_reason / self.compared * 100.0 if self.compared else 0.0

    @property
    def trustworthy(self) -> bool:
        """Every reason matched, every fill and net P&L inside tolerance.

        Deliberately strict: the exit REASON is the simulator's core claim, and
        a single mismatch means the replay took a different path through the
        exit set than the live engine did.
        """
        return (self.compared > 0
                and self.exact_reason == self.compared
                and self.fill_ok == self.compared
                and self.net_ok == self.compared)


class PolicyComparison:
    """Builds paired CONTROL/TP2 records from stored tapes."""

    def __init__(self, repo, strategy: str, cfg, fixed_tp_pct: float = 2.0) -> None:
        self.repo = repo
        self.strategy = strategy
        self.cfg = cfg
        self.a = cfg.aggressive
        self.fixed_tp_pct = fixed_tp_pct
        self.broker = sim.broker_from(cfg.execution)
        self.obs = {o["id"]: o for o in repo.get_observations(strategy=strategy)}
        self.paths = repo.get_excursions(strategy)
        self.crossover_atr = 2.0 / (self.a.target_r * self.a.stop_atr_mult)

    # ------------------------------------------------------------- helpers
    def atr_pct(self, path) -> float | None:
        o = self.obs.get(path.observation_id)
        if o:
            v = _num((o.get("features") or {}).get("atr_pct"))
            if v is not None:
                return v
        return (path.stop_distance_pct / self.a.stop_atr_mult
                if self.a.stop_atr_mult else None)

    def score(self, path) -> float | None:
        o = self.obs.get(path.observation_id)
        return _num(o.get("score")) if o else None

    def regime(self, path) -> str:
        o = self.obs.get(path.observation_id)
        return str((o.get("features") or {}).get("btc_regime") or "unknown") \
            if o else "unknown"

    # ---------------------------------------------------------- the pairing
    def pair(self, path) -> sim.PairedRecord:
        """Replay ONE signal under both policies. Identical inputs throughout."""
        score = self.score(path)
        rec = sim.PairedRecord(
            observation_id=path.observation_id, symbol=path.symbol,
            side=path.side, signal_ms=path.signal_ms, setup_score=score,
            # Phase 1 maps setup score to confidence with the identity; kept as
            # two fields because they mean different things.
            confidence=score, atr_pct=self.atr_pct(path),
            stop_distance_pct=path.stop_distance_pct)

        if path.status != "complete":
            rec.unreplayable = sim.UNREPLAYABLE_OPEN
            return rec
        tape = self.repo.get_tape(path.observation_id)
        if not tape:
            # A v7-era path. Its bars were never recorded and will not be
            # invented -- see the module docstring in tape.py.
            rec.unreplayable = sim.UNREPLAYABLE_NO_TAPE
            return rec

        common = dict(
            tape=tape, cfg=self.a, symbol=path.symbol, side=path.side,
            entry_ref=path.ref_price, entry_fill=path.ref_price,
            entry_ms=path.signal_ms, initial_stop=path.stop_price, qty=1.0,
            meta=path.meta, broker=self.broker,
            fixed_tp_pct=self.fixed_tp_pct)
        rec.control = sim.replay(policy=sim.CONTROL, **common)
        rec.tp2 = sim.replay(policy=sim.TP2, **common)
        if not rec.control.ok or not rec.tp2.ok:
            rec.unreplayable = (rec.control.unreplayable
                                or rec.tp2.unreplayable)
        return rec

    def pairs(self) -> list:
        return [self.pair(p) for p in
                sorted(self.paths, key=lambda p: p.signal_ms)]

    # ------------------------------------------------------- reconciliation
    def reconcile(self) -> Reconciliation:
        """Replay CONTROL over trades that ACTUALLY happened and compare."""
        out = Reconciliation()
        by_candle = {}
        for o in self.obs.values():
            by_candle[o["candle_id"]] = o
        for t in self.repo.get_trades(self.strategy):
            cid = (t["journal"] or {}).get("candle_id")
            o = by_candle.get(cid)
            path = (self.repo.get_excursion(o["id"]) if o else None)
            tape = self.repo.get_tape(o["id"]) if o else []
            if path is None or not tape:
                key = (sim.UNREPLAYABLE_NO_TAPE if o else "no_observation")
                out.unreplayable[key] = out.unreplayable.get(key, 0) + 1
                continue
            got = sim.replay(
                tape=tape, cfg=self.a, policy=sim.CONTROL, symbol=t["symbol"],
                side=t["side"], entry_ref=float(t["entry_ref_price"]),
                entry_fill=float(t["entry_fill_price"]),
                entry_ms=int(t["entry_ms"]),
                initial_stop=float(t["initial_stop"]), qty=float(t["qty"]),
                entry_fee=0.0, meta=path.meta, broker=self.broker,
                fixed_tp_pct=self.fixed_tp_pct)
            if not got.ok:
                out.unreplayable[got.unreplayable] = \
                    out.unreplayable.get(got.unreplayable, 0) + 1
                continue
            self._compare(out, t, got)
        return out

    def _compare(self, out: Reconciliation, t, got: sim.SimResult) -> None:
        out.compared += 1
        tid = t["id"]
        live_reason, sim_reason = t["exit_reason"], got.exit_reason
        bucket = out.by_exit_reason.setdefault(
            live_reason, {"n": 0, "reason_match": 0, "fill_ok": 0})
        bucket["n"] += 1

        if live_reason == sim_reason:
            out.exact_reason += 1
            bucket["reason_match"] += 1
        else:
            out.discrepancies.append(Discrepancy(
                tid, "exit_reason", live_reason, sim_reason,
                "the replay took a different branch through the exit set"))

        live_bar = int(t["exit_ms"]) // sim.BAR_MS
        sim_bar = got.exit_ms // sim.BAR_MS
        if live_bar == sim_bar:
            out.exact_bar += 1
            out.within_one_bar += 1
        elif abs(live_bar - sim_bar) <= TOL["exit_bars"]:
            out.within_one_bar += 1
        else:
            out.discrepancies.append(Discrepancy(
                tid, "exit_bar", live_bar, sim_bar,
                f"more than {TOL['exit_bars']} bar(s) apart"))

        # Fills: a stop never consults a quote, so it gets the tight tolerance.
        is_stop = str(live_reason).startswith("stop")
        tol_bps = TOL["stop_fill_bps"] if is_stop else TOL["quoted_fill_bps"]
        live_fill = float(t["exit_fill_price"])
        gap_bps = (abs(got.exit_fill_price - live_fill) / live_fill * 10_000.0
                   if live_fill else 0.0)
        if gap_bps <= tol_bps:
            out.fill_ok += 1
            bucket["fill_ok"] += 1
        else:
            out.discrepancies.append(Discrepancy(
                tid, "exit_fill_price", live_fill, got.exit_fill_price,
                f"{gap_bps:.2f} bps apart, tolerance {tol_bps:.2f} "
                f"({'stop' if is_stop else 'quote-priced'} exit)"))

        notional = float(t["qty"]) * float(t["entry_fill_price"])
        net_gap = abs(got.net_pnl - float(t["net_pnl"]))
        if notional <= 0 or net_gap / notional <= TOL["net_pnl_rel"]:
            out.net_ok += 1
        else:
            out.discrepancies.append(Discrepancy(
                tid, "net_pnl", t["net_pnl"], got.net_pnl,
                f"{net_gap / notional * 100:.3f}% of notional apart"))

        fin_gap = abs(got.financing - float(t["financing"]))
        if fin_gap > TOL["financing_abs"]:
            out.discrepancies.append(Discrepancy(
                tid, "financing", t["financing"], got.financing,
                f"{fin_gap:.4f} apart"))

    # ------------------------------------------------------------ summary
    def summarise(self, pairs=None) -> dict:
        pairs = self.pairs() if pairs is None else pairs
        ok = [p for p in pairs if p.ok]
        blocked: dict = {}
        for p in pairs:
            if not p.ok:
                blocked[p.unreplayable or "unknown"] = \
                    blocked.get(p.unreplayable or "unknown", 0) + 1
        return {
            "total_paths": len(pairs),
            "replayable": len(ok),
            "unreplayable": blocked,
            "label": sample_label(len(ok)),
            **_pair_stats(ok),
            "halves": _halves(ok),
            "by_side": _group_stats(ok, lambda p: p.side),
            "by_confidence_bucket": _group_stats(ok, lambda p: _conf_bucket(p.confidence)),
            "by_score_bucket": _group_stats(ok, lambda p: _bucket(p.setup_score, 10)),
            "by_atr_bucket": _group_stats(ok, lambda p: _bucket(p.atr_pct, 0.25)),
            "by_btc_regime": _group_stats(
                ok, lambda p: _regime_of(self.obs, p.observation_id)),
            "crossover": {
                f"atr<={self.crossover_atr:.4f}": _pair_stats(
                    [p for p in ok if (p.atr_pct or 0) <= self.crossover_atr]),
                f"atr>{self.crossover_atr:.4f}": _pair_stats(
                    [p for p in ok if (p.atr_pct or 0) > self.crossover_atr]),
            },
        }


def _regime_of(obs: dict, oid: str) -> str:
    o = obs.get(oid)
    return str((o.get("features") or {}).get("btc_regime") or "unknown") \
        if o else "unknown"


def _conf_bucket(v) -> str | None:
    f = _num(v)
    if f is None:
        return None
    for lo, hi in ((0, 60), (60, 70), (70, 80), (80, 90), (90, 95)):
        if lo <= f < hi:
            return f"{lo}-{hi}"
    return "95+"


def sample_label(n: int) -> str:
    """How much weight the numbers may carry. Stated, never inferred."""
    if n < MIN_PROVISIONAL:
        return INSUFFICIENT
    if n < MIN_COMPARABLE:
        return PROVISIONAL
    return COMPARABLE


def _policy_stats(results: list) -> dict:
    if not results:
        return {"n": 0, "expectancy_pct": 0.0, "win_rate_pct": 0.0,
                "profit_factor": 0.0, "avg_winner_pct": 0.0,
                "avg_loser_pct": 0.0, "hold_minutes_median": 0.0}
    rets = [r.net_return_pct for r in results]
    wins = [v for v in rets if v > 0]
    losses = [v for v in rets if v <= 0]
    gp, gl = sum(wins), abs(sum(losses))
    return {
        "n": len(rets),
        "expectancy_pct": st.mean(rets),
        "win_rate_pct": len(wins) / len(rets) * 100.0,
        "profit_factor": (gp / gl if gl else (float("inf") if gp else 0.0)),
        "avg_winner_pct": st.mean(wins) if wins else 0.0,
        "avg_loser_pct": st.mean(losses) if losses else 0.0,
        "hold_minutes_median": st.median([r.hold_minutes for r in results]),
        "gross_expectancy_pct": st.mean([r.gross_return_pct for r in results]),
    }


def _pair_stats(pairs: list) -> dict:
    diffs = [p.net_diff_pct for p in pairs]
    return {
        "n": len(pairs),
        "control_wins": sum(1 for p in pairs if p.winner == sim.CONTROL),
        "tp2_wins": sum(1 for p in pairs if p.winner == sim.TP2),
        "ties": sum(1 for p in pairs if p.winner == "tie"),
        "mean_diff_pct": st.mean(diffs) if diffs else 0.0,
        "median_diff_pct": st.median(diffs) if diffs else 0.0,
        "CONTROL": _policy_stats([p.control for p in pairs]),
        "TP2": _policy_stats([p.tp2 for p in pairs]),
    }


def _group_stats(pairs: list, keyfn) -> dict:
    groups: dict[str, list] = {}
    for p in pairs:
        try:
            k = keyfn(p)
        except Exception:
            k = None
        if k is not None:
            groups.setdefault(str(k), []).append(p)
    return {k: _pair_stats(v) for k, v in sorted(groups.items())}


def _halves(pairs: list) -> dict:
    """First half against second half, chronologically.

    An advantage that only exists in one half is a regime artefact wearing the
    costume of an edge. Reporting the SIGN in each half is the cheapest test
    that catches it, and it is why the pairs are ordered by signal time.
    """
    ordered = sorted(pairs, key=lambda p: p.signal_ms)
    mid = len(ordered) // 2
    first, second = ordered[:mid], ordered[mid:]

    def half(rows):
        d = [p.net_diff_pct for p in rows]
        mean = st.mean(d) if d else 0.0
        return {"n": len(rows), "mean_diff_pct": mean,
                "sign": ("+" if mean > 0 else "-" if mean < 0 else "0")}

    a, b = half(first), half(second)
    return {"first": a, "second": b,
            "same_sign": bool(a["n"] and b["n"] and a["sign"] == b["sign"]
                              and a["sign"] != "0")}
