"""Short-funnel, gate-overlap and calibration diagnostics. Research only.

WHY THE RECORDED REJECTION LABELS CANNOT ANSWER THIS ON THEIR OWN
-----------------------------------------------------------------
`choose_side` records ONE reason per observation, and it is lossy in three ways
that matter enormously here:

  1. The label lists ALL of the reported side's blockers joined by "; ". So the
     recorded rows are MUTUALLY EXCLUSIVE, not overlapping -- a signal blocked
     by both momentum and 15m structure appears under a third, combined label
     and never inside either single-gate count. Read naively, every
     single-gate count is an UNDERSTATEMENT of that gate's true reach, and the
     rows look independent when they are nothing of the kind.
  2. When NEITHER side is clear it reports only the side that came CLOSEST --
     `min((LONG, SHORT), key=(len(blockers), name))`, so LONG wins ties. A
     `no short: ...` label therefore requires the short side to have had
     strictly FEWER blockers than the long side. Every case where the market
     was plainly long is missing from the short rows entirely.
  3. The gates AFTER side selection -- the bullish-BTC veto, breadth,
     exhaustion, trend quality, the score floor -- are never applied to a
     signal that a blocker already killed. So a rejection count for one of
     those gates is a count over SURVIVORS, not over signals.

This module therefore ignores the labels and recomputes every gate from the
FEATURES stored on each observation, using the strategy's own functions:
`momentum_votes`, `score_components` and `combine`. A change to a rule changes
this diagnostic with it, rather than leaving a second copy of the rule to
drift.

THE SCORE HAD TO BE RECOMPUTED, NOT READ
----------------------------------------
`observations.score` is direction-dependent: `build()` scores with the
direction `choose_side` already picked, and records 0.0 when it picked neither.
Reading that column as a short's score would be wrong on every long and
meaningless on every NO_TRADE. So the short score is rebuilt by re-driving
`score_components(features, -1)`, which is the same function the live path
calls. Where a stored feature vector is too incomplete for that, the score gate
is marked `unavailable` rather than guessed.

WHAT THIS CANNOT DO, AND WILL NOT PRETEND TO
--------------------------------------------
Counterfactual returns are raw price moves over a fixed horizon: no stop, no
target, no trail, no fees, no slippage, no financing. A rejected short showing
+0.45% did not necessarily survive to collect it -- the same path could have
hit its stop first, and a stop sits only 1.8 ATR away. Every counterfactual
figure here is therefore reported alongside the MEASURED cost drag, and never
as forgone P&L.

Where a v8 tape exists, `policy_sim` can replay the real exit engine on these
same signals and produce actual stop/target P&L. That is the only way to turn
any of this into a trade result, and each row reports how many of its
observations are `replayable` that way.
"""
from __future__ import annotations

import math
import statistics as st
from dataclasses import dataclass, field

from ..strategy import confidence as conf
from ..strategy.aggressive_momentum import (
    VOTING_WINDOWS, AggressiveMomentumStrategy, momentum_votes)
from .forward_test import _num

SHORT_DIR = -1
NAN = float("nan")

# ---------------------------------------------------------------- the gates
# Named and ORDERED as `build()` and `choose_side()` apply them to a short.
# Side-independent, before a direction is even considered:
GATE_ATR = "atr_pct_floor"
GATE_RELVOL = "rel_volume_floor"
GATE_STRUCT15_AVAIL = "struct15_available"
PRE_GATES = (GATE_ATR, GATE_RELVOL, GATE_STRUCT15_AVAIL)

# `_blockers(f, SHORT)` -- these decide whether a SHORT is considered at all,
# and they are the three that get joined into one lossy label.
GATE_MOMENTUM = "momentum_agreement"
GATE_STRUCT15 = "struct15_bearish"
GATE_STRUCT1H = "struct1h_not_hostile"
BLOCKER_GATES = (GATE_MOMENTUM, GATE_STRUCT15, GATE_STRUCT1H)

# Applied only AFTER the short side has been selected, so a signal killed by a
# blocker above was never measured against any of these.
GATE_REGIME = "btc_regime_not_bull"
GATE_REGIME_KNOWN = "btc_regime_known"
GATE_BREADTH = "breadth_not_too_strong"
GATE_EXHAUSTION = "move_1h_not_exhausted"
GATE_EXTENSION = "not_over_extended"
POST_GATES = (GATE_REGIME, GATE_REGIME_KNOWN, GATE_BREADTH, GATE_EXHAUSTION,
              GATE_EXTENSION)

# Scoring, last of all. The score floor lives in the strategy; the confidence
# floor lives in the runtime and is APPLIED SECOND, so with the identity
# transform in force the effective floor on a tradable setup is
# max(min_setup_score, min_confidence) -- currently 60, not 50. A diagnostic
# that stopped at min_setup_score would report candidates the bot refuses.
GATE_TREND_R2 = "trend_quality"
GATE_SCORE = "setup_score_floor"
GATE_CONFIDENCE = "confidence_floor"
SCORE_GATES = (GATE_TREND_R2, GATE_SCORE, GATE_CONFIDENCE)

ALL_GATES = PRE_GATES + BLOCKER_GATES + POST_GATES + SCORE_GATES

# Fallback only. `cost_bps_from_trades` measures this from the ledger whenever
# closed trades exist, and that measurement is what the reports use.
DEFAULT_COST_BPS = 44.6


@dataclass
class GateResult:
    """One gate's verdict on one observation, plus the value that decided it."""
    passed: bool
    value: float | None = None
    threshold: float | None = None
    unavailable: bool = False


@dataclass
class ShortCandidate:
    observation_id: str
    symbol: str
    signal_ms: int
    score: float | None            # the SHORT's score, recomputed
    recorded_side: str = "long"
    recorded_score: float | None = None
    gates: dict = field(default_factory=dict)
    cf_return: float | None = None      # signed toward the SHORT
    has_tape: bool = False

    def blocked_by(self) -> list[str]:
        return [g for g in ALL_GATES
                if g in self.gates and not self.gates[g].passed]

    def clears(self, ignoring: tuple = ()) -> bool:
        return all(r.passed for g, r in self.gates.items() if g not in ignoring)


def _finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def rehydrate(obj):
    """Put NaN back where the journal wrote JSON null.

    `_jsonable` maps every non-finite float to None on the way in. The live
    scoring functions guard with `np.isfinite`, which raises on None, so a
    stored feature vector cannot be fed back to them untouched. Restoring NaN
    is what makes re-driving the real functions possible at all -- the
    alternative is a second, drifting copy of the scoring rules.
    """
    if isinstance(obj, dict):
        return {k: rehydrate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rehydrate(v) for v in obj]
    if obj is None:
        return NAN
    return obj


def short_score(features: dict, strategy: AggressiveMomentumStrategy):
    """The score this signal would have carried AS A SHORT.

    Re-drives `score_components(f, -1)` and `combine`, the same two functions
    `build()` calls. Returns None when the stored features cannot support them.
    """
    try:
        comps = strategy.score_components(features, SHORT_DIR)
        return float(strategy.combine(comps)), comps
    except (TypeError, ValueError, KeyError, ZeroDivisionError):
        return None, None


def evaluate_short_gates(features: dict, score: float | None, cfg) -> dict:
    """Re-run every short-side gate, in live order, from stored features.

    A gate whose input was never recorded is marked `unavailable`. Whether it
    then PASSES follows the live code case by case rather than one blanket
    rule: `_blockers` treats a missing 15m structure as blocking and a missing
    1h structure as not hostile, and the exhaustion checks only fire on a
    finite value. Collapsing those into "missing fails" would invent
    rejections; collapsing them into "missing passes" would invent candidates.
    """
    out: dict[str, GateResult] = {}
    c = cfg

    # --- side-independent, before any direction ---------------------------
    atr_pct = _num(features.get("atr_pct"))
    out[GATE_ATR] = GateResult(
        passed=bool(_finite(atr_pct) and atr_pct >= c.min_atr_pct),
        value=atr_pct, threshold=c.min_atr_pct, unavailable=not _finite(atr_pct))

    relvol = _num(features.get("rel_volume"))
    # Live: `if np.isfinite(rel_volume) and rel_volume < min` -- a NON-finite
    # relative volume does NOT reject.
    out[GATE_RELVOL] = GateResult(
        passed=not (_finite(relvol) and relvol < c.min_rel_volume),
        value=relvol, threshold=c.min_rel_volume, unavailable=not _finite(relvol))

    s15 = _num(features.get("ema_struct_15m"))
    out[GATE_STRUCT15_AVAIL] = GateResult(
        passed=_finite(s15), value=s15, threshold=None,
        unavailable=not _finite(s15))

    # --- _blockers(f, SHORT) ---------------------------------------------
    m_atr = features.get("momentum_atr") or {}
    votes = (momentum_votes(m_atr, SHORT_DIR, c.min_vote_atr)
             if isinstance(m_atr, dict) and m_atr else None)
    out[GATE_MOMENTUM] = GateResult(
        passed=bool(votes is not None and votes >= c.min_momentum_agree),
        value=votes, threshold=c.min_momentum_agree, unavailable=votes is None)

    # A short needs struct15 * -1 >= min_ema_struct_15m, i.e. struct15 <= -0.5
    out[GATE_STRUCT15] = GateResult(
        passed=bool(_finite(s15) and s15 * SHORT_DIR >= c.min_ema_struct_15m),
        value=s15, threshold=-c.min_ema_struct_15m, unavailable=not _finite(s15))

    s1h = _num(features.get("ema_struct_1h"))
    # Hostile when struct1h * -1 <= max_hostile_ema_1h, i.e. struct1h >= +0.5.
    # Live: an unavailable 1h structure is NOT hostile, so it passes here too.
    hostile = _finite(s1h) and s1h * SHORT_DIR <= c.max_hostile_ema_1h
    out[GATE_STRUCT1H] = GateResult(
        passed=not hostile, value=s1h, threshold=-c.max_hostile_ema_1h,
        unavailable=not _finite(s1h))

    # --- after the side has been selected --------------------------------
    regime = str(features.get("btc_regime") or "unknown")
    # Two separate refusals in the live code, kept separate here: the veto is a
    # directional OPINION and is what the ablation switches off; the unknown
    # check is a fail-closed safety rule that blocks both sides, and dropping
    # it would not be an ablation of the short rules at all.
    out[GATE_REGIME] = GateResult(
        passed=not (bool(c.veto_shorts_in_btc_bull) and regime == "bull"),
        value=None, threshold=None, unavailable=False)
    out[GATE_REGIME_KNOWN] = GateResult(
        passed=regime != "unknown", value=None, threshold=None,
        unavailable=regime == "unknown")

    # Live: `f.get("breadth_pct", 50.0)` -- an ABSENT key defaults to 50.
    breadth = _num(features.get("breadth_pct"))
    if breadth is None:
        breadth = 50.0
    out[GATE_BREADTH] = GateResult(
        passed=breadth <= c.max_breadth_for_short, value=breadth,
        threshold=c.max_breadth_for_short,
        unavailable=_num(features.get("breadth_pct")) is None)

    move_1h = _num(m_atr.get("1h")) if isinstance(m_atr, dict) else None
    out[GATE_EXHAUSTION] = GateResult(
        passed=not (_finite(move_1h)
                    and move_1h * SHORT_DIR > c.max_move_atr_1h),
        value=move_1h, threshold=-c.max_move_atr_1h,
        unavailable=not _finite(move_1h))

    # A short heads DOWN, so its room is the distance to the swing LOW.
    behind = _num(features.get("dist_to_swing_low_atr"))
    out[GATE_EXTENSION] = GateResult(
        passed=not (_finite(behind) and behind < -c.max_extension_atr),
        value=behind, threshold=-c.max_extension_atr,
        unavailable=not _finite(behind))

    # --- scoring ----------------------------------------------------------
    r2s = [v for v in (_num(features.get("trend_r2_15m")),
                       _num(features.get("trend_r2_1h"))) if _finite(v)]
    r2 = max(r2s) if r2s else None
    out[GATE_TREND_R2] = GateResult(
        passed=not (_finite(r2) and r2 < c.min_trend_r2),
        value=r2, threshold=c.min_trend_r2, unavailable=r2 is None)

    out[GATE_SCORE] = GateResult(
        passed=bool(_finite(score) and score >= c.min_setup_score),
        value=score, threshold=c.min_setup_score, unavailable=not _finite(score))

    # Re-drives the runtime's own two calls rather than assuming the identity:
    # if `confidence_is_identity` is ever turned off, this gate follows.
    confidence = None
    if _finite(score):
        try:
            confidence = conf.to_confidence(score, c.confidence_is_identity)
        except NotImplementedError:
            confidence = None
    out[GATE_CONFIDENCE] = GateResult(
        passed=bool(confidence is not None
                    and conf.is_tradable(confidence, c.min_confidence,
                                         c.confidence_buckets)),
        value=confidence, threshold=c.min_confidence,
        unavailable=confidence is None)
    return out


def gate_threshold_text(a_cfg) -> dict[str, str]:
    """The rule each gate applies, as the config actually sets it."""
    r = a_cfg
    return {
        GATE_ATR: f"atr_pct >= {r.min_atr_pct:g}%",
        GATE_RELVOL: f"rel_volume >= {r.min_rel_volume:g} (or not finite)",
        GATE_STRUCT15_AVAIL: "ema_struct_15m is finite",
        GATE_MOMENTUM: (f"{r.min_momentum_agree}/{len(VOTING_WINDOWS)} windows"
                        f" <= -{r.min_vote_atr:g} ATR"),
        GATE_STRUCT15: f"ema_struct_15m <= {-r.min_ema_struct_15m:g}",
        GATE_STRUCT1H: f"ema_struct_1h < {-r.max_hostile_ema_1h:g}",
        GATE_REGIME: f"btc_regime != bull (veto={r.veto_shorts_in_btc_bull})",
        GATE_REGIME_KNOWN: "btc_regime != unknown",
        GATE_BREADTH: f"breadth <= {r.max_breadth_for_short:g}%",
        GATE_EXHAUSTION: f"1h move > {-r.max_move_atr_1h:g} ATR",
        GATE_EXTENSION: f"dist_to_swing_low >= {-r.max_extension_atr:g} ATR",
        GATE_TREND_R2: f"max trend R2 >= {r.min_trend_r2:g}",
        GATE_SCORE: f"short setup_score >= {r.min_setup_score:g}",
        GATE_CONFIDENCE: (f"confidence >= {r.min_confidence:g} with a"
                          " non-zero bucket multiplier"),
    }


class ShortFunnel:
    """Stage counts, gate overlap, and offline ablations for the short side.

    Every observation is asked the same question -- "could this have been a
    SHORT, and what happened next if it had been" -- regardless of the side the
    live code ended up recording. That is the only framing in which a funnel
    means anything: a gate's reach is how many signals it stopped, not how many
    survived long enough to be labelled with its name.
    """

    def __init__(self, repo, strategy: str, cfg, cost_bps: float | None = None,
                 horizon_h: int | None = None) -> None:
        self.repo = repo
        self.strategy = strategy
        self.a = cfg.aggressive
        self.x = cfg.execution
        self.horizon_h = horizon_h
        if cost_bps is not None:
            self.cost_bps, self.measured_cost = float(cost_bps), False
        else:
            self.cost_bps = cost_bps_from_trades(repo, strategy)
            # True only when there was turnover to measure it FROM. Reporting a
            # hardcoded default as a measurement is how a placeholder becomes a
            # finding.
            self.measured_cost = any(
                float(t["qty"]) * float(t["entry_fill_price"])
                for t in repo.get_trades(strategy))
        self._strat = AggressiveMomentumStrategy(cfg.aggressive)
        self.obs = repo.get_observations(strategy=strategy)
        self._cf = None
        self.candidates = [self._candidate(o) for o in self.obs]

    # ------------------------------------------------------------ helpers
    def counterfactuals(self) -> dict:
        """Raw forward moves keyed by observation, one list per observation."""
        if self._cf is None:
            out: dict[str, list[float]] = {}
            q = ("SELECT observation_id, horizon_h, return_pct FROM"
                 " counterfactuals WHERE return_pct IS NOT NULL")
            args: tuple = ()
            if self.horizon_h is not None:
                q += " AND horizon_h = ?"
                args = (int(self.horizon_h),)
            for r in self.repo.conn.execute(q, args):
                out.setdefault(r["observation_id"], []).append(
                    float(r["return_pct"]))
            self._cf = out
        return self._cf

    def _candidate(self, o) -> ShortCandidate:
        f = rehydrate(o.get("features") or {})
        score, _ = short_score(f, self._strat)
        # Signed toward the SHORT: a fall is a win for a short, so the raw
        # move is negated. Averaging raw moves would make a directional gate
        # look wrong exactly when it was working.
        raw = self.counterfactuals().get(o["id"])
        cf = (-st.mean(raw) if raw else None)
        return ShortCandidate(
            observation_id=o["id"], symbol=o["symbol"],
            signal_ms=int(o["ts_ms"]), score=score,
            recorded_side=(o.get("side") or "long"),
            recorded_score=_num(o.get("score")),
            gates=evaluate_short_gates(f, score, self.a),
            cf_return=cf, has_tape=self.repo.tape_bar_count(o["id"]) > 0)

    def _stats(self, rows: list) -> dict:
        cf = [c.cf_return for c in rows if c.cf_return is not None]
        taped = sum(1 for c in rows if c.has_tape)
        mean = st.mean(cf) if cf else None
        return {
            "n": len(rows), "with_outcome": len(cf),
            "mean_cf_pct": mean,
            "median_cf_pct": st.median(cf) if cf else None,
            "mean_after_cost_pct": (mean - self.cost_bps / 100.0
                                    if mean is not None else None),
            "share_positive": (sum(1 for v in cf if v > 0) / len(cf) * 100.0
                               if cf else None),
            "replayable": taped,
            "pnl_reproducible": bool(rows) and taped == len(rows),
        }

    # ----------------------------------------------------------- 1. funnel
    def funnel(self) -> list[dict]:
        """Sequential funnel in the live order. Each row is what SURVIVED.

        `rejected` is what this gate killed OUT OF WHAT REACHED IT -- the
        number the live pipeline actually attributes to it. `overlap()` gives
        the other, larger view: how many that gate would have stopped on its
        own, whatever came before.
        """
        thresholds = gate_threshold_text(self.a)
        rows, surviving = [], list(self.candidates)
        rows.append({"stage": "observations evaluated", "gate": "-",
                     "reached": len(surviving), "rejected": 0,
                     "rejected_stats": self._stats([]),
                     **self._stats(surviving)})
        for gate in ALL_GATES:
            before = surviving
            surviving = [c for c in before if c.gates[gate].passed]
            killed = [c for c in before if not c.gates[gate].passed]
            rows.append({
                "stage": gate, "gate": thresholds[gate],
                "reached": len(before), "rejected": len(killed),
                "unavailable": sum(1 for c in killed
                                   if c.gates[gate].unavailable),
                "rejected_stats": self._stats(killed),
                **self._stats(surviving)})
        return rows

    def side_contest(self) -> dict:
        """Whether the long/short tie-break ever actually cost a short.

        A short is taken only when the short side is clear AND the long side is
        not. The gates are mirror images through `* d`, so a clear short
        implies a blocked long -- but that is a property of the current
        thresholds, not a guarantee, and the diagnostic states it as a measured
        count rather than an assumption.
        """
        both = sum(1 for c in self.candidates
                   if all(c.gates[g].passed for g in BLOCKER_GATES)
                   and _long_blockers_empty(c))
        return {"short_clear": sum(1 for c in self.candidates
                                   if all(c.gates[g].passed
                                          for g in BLOCKER_GATES)),
                "short_and_long_both_clear": both,
                "note": ("both-clear would be recorded as 'contradictory long"
                         " and short conditions', not as a short")}

    # -------------------------------------------- 2. overlap / redundancy
    def overlap(self, gates=None) -> dict:
        """Co-occurrence of short-side gates, and each gate's UNIQUE reach.

        `only_this_gate` counts what a gate rejects that NOTHING else would
        have -- the only figure that says what relaxing it ALONE would admit. A
        gate with a large `total` and a tiny `only_this_gate` is not the
        binding constraint however bad its label looks, and this is the number
        the recorded rejection rows cannot produce.
        """
        gates = tuple(gates or ALL_GATES)
        matrix, unique, totals = {}, {}, {}
        for g in gates:
            failed_g = [c for c in self.candidates if not c.gates[g].passed]
            totals[g] = len(failed_g)
            unique[g] = sum(1 for c in failed_g
                            if [x for x in c.blocked_by() if x in gates] == [g])
            matrix[g] = {h: sum(1 for c in failed_g if not c.gates[h].passed)
                         for h in gates}
        return {"totals": totals, "only_this_gate": unique, "matrix": matrix,
                "gates": list(gates),
                "n_blockers_histogram": _histogram(
                    [len([x for x in c.blocked_by() if x in gates])
                     for c in self.candidates]),
                "n_observations": len(self.candidates)}

    # ------------------------------------------------- 3. gate ablations
    def ablations(self) -> list[dict]:
        """What each relaxation would ADMIT. Offline, never applied.

        `ignoring` drops a gate ENTIRELY rather than nudging a threshold: a
        nudge needs a number nobody has justified yet, and the point here is to
        bound the upside, not to pick a new value. So every `extra_admitted`
        count is a CEILING on what relaxing that gate could add, and every
        `mean_cf_pct` beside it is a raw price move, not a trade result.
        """
        variants = [
            ("current short rules", ()),
            ("drop the bullish-BTC short veto", (GATE_REGIME,)),
            ("relax short momentum agreement", (GATE_MOMENTUM,)),
            ("relax 15m bearish-structure", (GATE_STRUCT15,)),
            ("relax momentum + 15m together", (GATE_MOMENTUM, GATE_STRUCT15)),
            ("relax 1h hostility", (GATE_STRUCT1H,)),
            ("drop veto AND relax momentum + 15m",
             (GATE_REGIME, GATE_MOMENTUM, GATE_STRUCT15)),
        ]
        # Two readings of every variant, because they answer different
        # questions. DIRECTIONAL ignores the score and confidence floors, so it
        # isolates what the structural gate itself was holding back. FULL keeps
        # them, so it says what would actually have been traded. When the two
        # diverge, the structural gate was never the binding constraint -- the
        # score floor was, and relaxing the structural gate changes nothing.
        directional = tuple(g for g in ALL_GATES
                            if g not in (GATE_SCORE, GATE_CONFIDENCE))

        def clears_dir(c, ignoring=()):
            return all(c.gates[g].passed for g in directional
                       if g not in ignoring)

        base = [c for c in self.candidates if c.clears()]
        base_dir = [c for c in self.candidates if clears_dir(c)]
        out = []
        for label, ignoring in variants:
            admitted = [c for c in self.candidates if c.clears(ignoring)]
            extra = [c for c in admitted if not c.clears()]
            adm_dir = [c for c in self.candidates if clears_dir(c, ignoring)]
            extra_dir = [c for c in adm_dir if not clears_dir(c)]
            out.append({
                "variant": label, "ignoring": list(ignoring),
                "admitted": len(admitted), "baseline": len(base),
                "extra_admitted": len(extra),
                "admitted_directional": len(adm_dir),
                "baseline_directional": len(base_dir),
                "extra_directional": len(extra_dir),
                "extra_stats": self._stats(extra),
                "extra_directional_stats": self._stats(extra_dir),
                "all_stats": self._stats(admitted),
            })
        return out

    # ------------------------------------------------ 4. score thresholds
    def score_floors(self, floors=(60.0, 70.0, 75.0, 80.0)) -> dict:
        """Realised and hypothetical, side by side but NEVER summed together.

        The realised column is a SHADOW ledger: it re-totals the trades a
        higher floor would still have taken. It is not what the account would
        have shown, because skipping the low-score trades leaves more cash and
        a different ladder slot for the survivors -- see `caveats`.
        """
        trades = self.repo.get_trades(self.strategy)
        scored = [(t, _num((t["journal"] or {}).get("setup_score")))
                  for t in trades]
        realised, hypo = [], []
        for f in floors:
            kept = [t for t, s in scored if s is not None and s >= f]
            wins = [t for t in kept if t["net_pnl"] > 0]
            turnover = sum(float(t["qty"]) * float(t["entry_fill_price"])
                           for t in kept)
            net = sum(t["net_pnl"] for t in kept)
            realised.append({
                "floor": f, "closed_trades": len(kept),
                "net_pnl": net,
                "gross_pnl": sum(t["gross_pnl"] for t in kept),
                "fees": sum(t["fees"] for t in kept),
                "slippage": sum(t["slippage_cost"] for t in kept),
                "financing": sum(t["financing"] for t in kept),
                "turnover": turnover,
                "net_bps": (net / turnover * 10_000.0 if turnover else None),
                "win_rate_pct": (len(wins) / len(kept) * 100.0 if kept else 0.0),
                "dropped": len(trades) - len(kept),
            })
            above = [c for c in self.candidates
                     if c.score is not None and c.score >= f]
            hypo.append({"floor": f, "observations": len(above),
                         **self._stats(above)})
        missing = sum(1 for _, s in scored if s is None)
        shorts = [c.score for c in self.candidates if c.score is not None]
        return {
            "realised": realised, "hypothetical": hypo,
            "short_score_computable": len(shorts),
            "short_score_missing": len(self.candidates) - len(shorts),
            "short_score_histogram": _score_histogram(shorts),
            "trades_without_recorded_score": missing,
            "total_closed_trades": len(trades),
            "warning": ("realised and hypothetical are reported separately and"
                        " must not be added: one is P&L after real fills and"
                        " costs, the other a raw price move with no stop"),
            "caveats": [
                "path dependence: dropping the low-score trades frees cash and"
                " ladder slots, so the survivors would have been SIZED"
                " DIFFERENTLY. Only a portfolio replay settles that.",
                "a floor above every recorded score yields n=0, which is not"
                " an improvement -- read closed_trades before net_pnl.",
            ],
        }

    # ------------------------------------------------ 5. cost attribution
    def cost_split(self) -> dict:
        """Exact entry/exit fee and slippage split, plus the GAP component."""
        return cost_split(self.repo, self.strategy, self.x)

    def score_bucket_costs(self, width: float = 10.0) -> list[dict]:
        """Per-score-bucket P&L expressed in bps of the notional it risked.

        Dollar P&L per bucket confounds quality with size: the ladder gives a
        higher-confidence setup more notional, so a bucket can lose more
        dollars while losing less per dollar. Normalising by turnover separates
        them, and adding the measured cost drag back recovers the GROSS edge --
        which is what distinguishes a bucket that has no edge from one whose
        edge is real but smaller than its costs. Those need opposite fixes.
        """
        out: dict[str, dict] = {}
        for t in self.repo.get_trades(self.strategy):
            s = _num((t["journal"] or {}).get("setup_score"))
            if s is None:
                continue
            lo = math.floor(s / width) * width
            key = f"{lo:g}-{lo + width:g}"
            b = out.setdefault(key, {
                "bucket": key, "n": 0, "wins": 0, "turnover": 0.0,
                "gross_pnl": 0.0, "net_pnl": 0.0, "fees": 0.0,
                "slippage": 0.0, "financing": 0.0})
            b["n"] += 1
            b["wins"] += 1 if t["net_pnl"] > 0 else 0
            b["turnover"] += float(t["qty"]) * float(t["entry_fill_price"])
            for k, col in (("gross_pnl", "gross_pnl"), ("net_pnl", "net_pnl"),
                           ("fees", "fees"), ("slippage", "slippage_cost"),
                           ("financing", "financing")):
                b[k] += float(t[col])
        rows = []
        for b in out.values():
            tn = b["turnover"]
            b["avg_notional"] = tn / b["n"] if b["n"] else 0.0
            b["win_rate_pct"] = b["wins"] / b["n"] * 100.0 if b["n"] else 0.0
            b["net_bps"] = b["net_pnl"] / tn * 10_000.0 if tn else None
            b["gross_bps"] = b["gross_pnl"] / tn * 10_000.0 if tn else None
            b["cost_bps"] = (((b["fees"] + b["slippage"] + b["financing"]) / tn
                              * 10_000.0) if tn else None)
            rows.append(b)
        return sorted(rows, key=lambda r: float(r["bucket"].split("-")[0]))


def _long_blockers_empty(c: ShortCandidate) -> bool:
    """A mirror of `_blockers(f, LONG)` being empty, from the short's values.

    The long gates read the SAME two structure numbers with the opposite sign,
    so they can be inverted exactly: a long is unblocked when the 15m structure
    is bullish enough and the 1h structure is not hostile to a long. Momentum
    cannot be inverted from the short's vote count -- a window that failed to
    vote short need not vote long -- so this is deliberately conservative and
    reports only the structure part.
    """
    s15 = c.gates[GATE_STRUCT15].value
    s1h = c.gates[GATE_STRUCT1H].value
    thr15 = -(c.gates[GATE_STRUCT15].threshold or 0.0)
    thr1h = -(c.gates[GATE_STRUCT1H].threshold or 0.0)
    if not _finite(s15):
        return False
    long15 = s15 >= thr15
    long1h_ok = not (_finite(s1h) and s1h <= thr1h)
    return bool(long15 and long1h_ok)


def _score_histogram(scores: list, width: float = 10.0) -> dict:
    """Where the RECOMPUTED short scores actually sit.

    Printed beside the floor table because a floor that admits nothing looks
    identical to a floor that is too high, and the distribution is what tells
    the two apart.
    """
    out: dict[str, int] = {}
    for v in scores:
        lo = math.floor(v / width) * width
        key = f"{lo:g}-{lo + width:g}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: float(kv[0].split("-")[0])))


def _histogram(values: list) -> dict:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def cost_split(repo, strategy: str, exec_cfg) -> dict:
    """Decompose every closed trade's costs into its real components.

    The ledger stores `fees` and `slippage_cost` already summed across both
    legs, which is exactly the granularity that hides the answer. These are
    rebuilt from the four stored prices instead, so the entry leg, the exit
    leg, and the part of the exit that is a GAP THROUGH THE STOP rather than a
    bps assumption can be told apart. That distinction decides whether the
    slippage line is a modelling choice that could be argued with or a market
    fact that cannot.
    """
    trades = repo.get_trades(strategy)
    stop_bps = float(exec_cfg.stop_slippage_bps)
    tot = {"entry_fee": 0.0, "exit_fee": 0.0, "entry_slippage": 0.0,
           "exit_slippage": 0.0, "exit_slippage_stops": 0.0,
           "stop_slippage_modelled": 0.0,
           "gap_component": 0.0, "financing": 0.0,
           "gross_pnl": 0.0, "net_pnl": 0.0, "turnover": 0.0,
           "fees_ledger": 0.0, "slippage_ledger": 0.0}
    gapped = stops = 0
    for t in trades:
        qty = float(t["qty"])
        d = -1 if t["side"] == "short" else 1
        e_ref, e_fill = float(t["entry_ref_price"]), float(t["entry_fill_price"])
        x_ref, x_fill = float(t["exit_ref_price"]), float(t["exit_fill_price"])
        e_not, x_not = qty * e_fill, qty * x_fill
        # Both legs pay the same taker rate, so the stored total splits by
        # notional exactly -- no assumption about the rate is needed.
        total_fee = float(t["fees"])
        share = e_not / (e_not + x_not) if (e_not + x_not) else 0.5
        tot["entry_fee"] += total_fee * share
        tot["exit_fee"] += total_fee * (1.0 - share)
        tot["fees_ledger"] += total_fee
        tot["slippage_ledger"] += float(t["slippage_cost"])
        tot["entry_slippage"] += (e_fill - e_ref) * qty * d
        exit_slip = (x_ref - x_fill) * qty * d
        tot["exit_slippage"] += exit_slip
        tot["financing"] += float(t["financing"])
        tot["gross_pnl"] += float(t["gross_pnl"])
        tot["net_pnl"] += float(t["net_pnl"])
        tot["turnover"] += e_not
        if str(t["exit_reason"]).startswith("stop"):
            stops += 1
            tot["exit_slippage_stops"] += exit_slip
            # The broker fills at `base * (1 - stop_bps * d)` where `base` is
            # the stop itself, or the bar's OPEN when the bar gapped through
            # it. Recovering `base` from the fill splits the exit slippage into
            # the bps the model charges and the gap it did not choose.
            denom = 1.0 - stop_bps / 10_000.0 * d
            base = x_fill / denom if denom else x_fill
            modelled = base * stop_bps / 10_000.0 * qty
            gap = (x_ref - base) * qty * d
            if gap > 0:
                gapped += 1
            else:
                gap, modelled = 0.0, exit_slip
            tot["stop_slippage_modelled"] += modelled
            tot["gap_component"] += gap
    tot["n_trades"] = len(trades)
    tot["stop_exits"] = stops
    tot["gapped_exits"] = gapped
    # Identity checks, reported rather than assumed: a non-zero residual means
    # the reconstruction is wrong and nothing built on it is safe to read.
    tot["fee_residual"] = tot["fees_ledger"] - (tot["entry_fee"] + tot["exit_fee"])
    tot["slippage_residual"] = tot["slippage_ledger"] - (
        tot["entry_slippage"] + tot["exit_slippage"])
    # Scoped to the STOP exits: the split only claims to explain those, and
    # measuring it against every exit's slippage would leave the non-stop legs
    # sitting in the residual and read as a broken reconstruction.
    tot["exit_slippage_non_stop"] = (tot["exit_slippage"]
                                     - tot["exit_slippage_stops"])
    tot["gap_residual"] = (tot["exit_slippage_stops"]
                           - tot["stop_slippage_modelled"]
                           - tot["gap_component"])
    if tot["turnover"]:
        tot["bps"] = {k: v / tot["turnover"] * 10_000.0
                      for k, v in tot.items()
                      if isinstance(v, float) and k != "turnover"}
    return tot


def cost_bps_from_trades(repo, strategy: str,
                         default: float = DEFAULT_COST_BPS) -> float:
    """Round-trip cost drag in bps of turnover, MEASURED from the ledger.

    Falls back to `default` only when there is no turnover to measure, and the
    caller records which of the two it got via `ShortFunnel.measured_cost`.
    """
    trades = repo.get_trades(strategy)
    turnover = sum(float(t["qty"]) * float(t["entry_fill_price"]) for t in trades)
    if not turnover:
        return default
    cost = sum(float(t["fees"]) + float(t["slippage_cost"])
               + float(t["financing"]) for t in trades)
    return cost / turnover * 10_000.0
