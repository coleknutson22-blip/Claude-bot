"""Short-funnel and calibration diagnostics.

WHAT THESE TESTS ARE REALLY DEFENDING
-------------------------------------
This module exists because the recorded rejection labels cannot answer "why no
shorts". That makes it a SECOND implementation of rules the live code already
owns, and a second implementation is exactly how a diagnostic starts confidently
describing a strategy that no longer behaves that way. Four failure modes would
each produce a plausible, wrong answer:

  1. GATE DRIFT. If `evaluate_short_gates` disagrees with `_blockers` by one
     boundary, the funnel reports rejections the bot never made. The fidelity
     tests re-drive the live function on the same features and require the two
     verdicts to match, including at the exact threshold.

  2. A GUESSED SCORE. `observations.score` is direction-dependent and is 0.0 on
     every NO_TRADE, so reading it as a short's score would be wrong on every
     row. The score is recomputed through the strategy's own
     `score_components(f, -1)`, and that is asserted, not assumed.

  3. COUNTERFACTUALS READ WITH THE WRONG SIGN. A short was right when price
     FELL. Averaging raw moves would make a working directional gate look
     broken -- and would do so most strongly exactly where it worked best.

  4. A COST SPLIT THAT DOES NOT ADD UP. The whole point of rebuilding costs
     from the four stored prices is to separate the bps the model chooses from
     the gap it does not. If the split fails to reconcile against the ledger,
     the conclusion "slippage is mostly gap" is unfounded.

All four are tested directly and mutation-tested.
"""
import math
import unittest

import helpers  # noqa: F401  -- silences the engine's log handlers
from crypto_edge.config import Config
from crypto_edge.models import ClosedTrade
from crypto_edge.research import short_funnel as sf
from crypto_edge.strategy.aggressive_momentum import (
    SHORT, AggressiveMomentumStrategy)
from helpers import temp_repo

B = "aggressive_momentum_v2"
T0 = 1_700_000_000_000

DOWN = {"30m": -2.2, "1h": -2.4, "2h": -2.3, "3h": -2.1, "6h": -2.0}
UP = {"30m": 2.2, "1h": 2.4, "2h": 2.3, "3h": 2.1, "6h": 2.0}
# Only two of five windows point down, so momentum agreement fails at 3.
WEAK_DOWN = {"30m": -2.2, "1h": -2.4, "2h": 0.2, "3h": 0.1, "6h": 0.05}

# The recomputed short score `feats()` produces. Asserted below rather than
# only relied on: a fixture that quietly slipped under the 60 confidence floor
# would make half these tests pass for the wrong reason -- every gate would
# look redundant because the floor was blocking the row anyway.
CLEAN_SHORT_SCORE = 90.13


def feats(**kw) -> dict:
    """A feature vector that clears EVERY short gate, before overrides.

    Including the score and confidence floors -- see `CLEAN_SHORT_SCORE`.
    """
    f = {
        "price": 100.0, "atr": 1.0, "atr_pct": 0.8, "rel_volume": 2.0,
        "ema_struct_5m": -1.0, "ema_struct_15m": -1.0, "ema_struct_1h": -0.9,
        "momentum_atr": dict(DOWN), "breadth_pct": 20.0, "btc_regime": "bear",
        "trend_r2_15m": 0.7, "trend_r2_1h": 0.7,
        "dist_to_swing_low_atr": 2.5, "dist_to_swing_high_atr": 2.5,
        "rel_strength": {"1h": -5.0, "6h": -5.0}, "atr_expansion": 1.8,
    }
    f.update(kw)
    return f


def low_score_feats(**kw) -> dict:
    """Clears every DIRECTIONAL gate and fails the score and confidence floors.

    This is the shape that separates the two readings of an ablation: relaxing
    a structural gate lets this row through the direction check and the bot
    still refuses it. A diagnostic that reported only one number would claim a
    gate change buys trades it cannot.
    """
    f = feats(momentum_atr={"30m": -0.2, "1h": -0.2, "2h": -0.2, "3h": -0.2,
                            "6h": -0.2},
              ema_struct_5m=-0.5, ema_struct_15m=-0.5, rel_volume=0.9,
              atr_expansion=0.7, trend_r2_15m=0.12, trend_r2_1h=0.12,
              dist_to_swing_low_atr=0.25,
              rel_strength={"1h": 0.0, "6h": 0.0}, breadth_pct=75.0)
    f.update(kw)
    return f


# The gates that decide DIRECTION, with the score and confidence floors left
# out. Relaxing a structural gate can only be measured against this set: with
# the floors in scope a low-scoring row is blocked either way, and every
# structural gate reads as redundant.
DIRECTIONAL = tuple(g for g in sf.ALL_GATES
                    if g not in (sf.GATE_SCORE, sf.GATE_CONFIDENCE))


class Base(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.a = self.cfg.aggressive
        self.strat = AggressiveMomentumStrategy(self.a)
        self.repo, _ = temp_repo()
        self.i = 0

    # ------------------------------------------------------------- fixtures
    def obs(self, features=None, *, cf=None, side="long", score=0.0,
            horizons=(1, 4, 24)):
        self.i += 1
        oid = self.repo.add_observation({
            "ts_ms": T0 + self.i * 300_000, "symbol": f"S{self.i}/USD",
            "candle_id": f"c{self.i}", "strategy": B, "strategy_version": "v2",
            "decision": "REJECTED_STRATEGY", "reject_reason": "x",
            "side": side, "score": score, "rank": None, "price": 100.0,
            "features": _jsonify(features if features is not None else feats())})
        if cf is not None:
            for h in horizons:
                self.repo.add_counterfactual(oid, h, 100.0,
                                             100.0 * (1 + cf / 100.0), cf,
                                             T0 + h * 3_600_000)
        return oid

    def trade(self, *, score=70.0, qty=25.0, entry_ref=100.0, entry_fill=100.06,
              exit_ref=98.2, exit_fill=98.05, reason="stop", side="long",
              financing=0.0, journal_score=True):
        self.i += 1
        d = -1 if side == "short" else 1
        gross = (exit_ref - entry_ref) * qty * d
        fees = (qty * entry_fill + qty * exit_fill) * 0.00075
        slip = ((entry_fill - entry_ref) * qty * d
                + (exit_ref - exit_fill) * qty * d)
        net = gross - fees - slip - financing
        self.repo.add_trade(ClosedTrade(
            id=f"trd{self.i}", position_id=f"pos{self.i}",
            symbol=f"S{self.i}/USD", strategy=B, strategy_version="v2",
            side=side, qty=qty, entry_ref_price=entry_ref,
            entry_fill_price=entry_fill, entry_ms=T0 + self.i * 600_000,
            exit_ref_price=exit_ref, exit_fill_price=exit_fill,
            exit_ms=T0 + self.i * 600_000 + 3_600_000, exit_reason=reason,
            initial_stop=98.2, final_stop=98.2, gross_pnl=gross, fees=fees,
            slippage_cost=slip, net_pnl=net, financing=financing,
            return_pct=0.0, account_return_pct=0.0, mfe=1.0, mae=-1.0,
            duration_s=3600.0, equity_after=10_000.0,
            journal=({"setup_score": score, "confidence": score}
                     if journal_score else {})))
        return net

    def funnel(self, **kw):
        return sf.ShortFunnel(self.repo, B, self.cfg, **kw)

    def gates(self, features=None, score=None):
        return sf.evaluate_short_gates(
            features if features is not None else feats(), score, self.a)


def _jsonify(f):
    """What the journal would have stored: every non-finite float becomes null."""
    def conv(v):
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v
    return conv(f)


# ================================================== 1. fidelity to the live code
class TestGateFidelity(Base):
    """The diagnostic must agree with `_blockers`, not merely resemble it."""

    def assert_agrees(self, features):
        live_blocked = bool(self.strat._blockers(features, SHORT))
        mine = self.gates(features)
        diag_blocked = not all(mine[g].passed for g in sf.BLOCKER_GATES)
        self.assertEqual(
            live_blocked, diag_blocked,
            f"live _blockers and evaluate_short_gates disagree on {features}")

    def test_agrees_on_a_clean_short(self):
        self.assert_agrees(feats())

    def test_agrees_when_momentum_is_short(self):
        self.assert_agrees(feats(momentum_atr=dict(WEAK_DOWN)))

    def test_agrees_when_15m_structure_is_not_bearish(self):
        self.assert_agrees(feats(ema_struct_15m=-0.2))

    def test_agrees_when_1h_structure_is_hostile(self):
        self.assert_agrees(feats(ema_struct_1h=0.9))

    def test_agrees_on_a_long_shaped_market(self):
        self.assert_agrees(feats(ema_struct_5m=1.0, ema_struct_15m=1.0,
                                 ema_struct_1h=0.9, momentum_atr=dict(UP)))

    def test_agrees_at_every_15m_structure_boundary(self):
        # The threshold itself is where an off-by-one lives.
        for s15 in (-0.51, -0.5, -0.49, 0.0, 0.49, 0.5, 0.51):
            with self.subTest(s15=s15):
                self.assert_agrees(feats(ema_struct_15m=s15))

    def test_agrees_at_every_1h_hostility_boundary(self):
        for s1h in (-0.51, -0.5, -0.49, 0.49, 0.5, 0.51):
            with self.subTest(s1h=s1h):
                self.assert_agrees(feats(ema_struct_1h=s1h))

    def test_agrees_at_the_momentum_vote_boundary(self):
        # Exactly `min_vote_atr` counts as a vote; a hair under does not.
        eps = 1e-9
        for v in (self.a.min_vote_atr, self.a.min_vote_atr - eps,
                  self.a.min_vote_atr + eps):
            m = {"30m": -v, "1h": -v, "2h": -v, "3h": 0.0, "6h": 0.0}
            with self.subTest(v=v):
                self.assert_agrees(feats(momentum_atr=m))

    def test_missing_1h_structure_is_not_hostile_in_either(self):
        # Live: `if np.isfinite(struct1h) and ...` -- absent means NOT hostile.
        f = feats()
        f.pop("ema_struct_1h")
        self.assert_agrees(f)
        self.assertTrue(self.gates(f)[sf.GATE_STRUCT1H].passed)
        self.assertTrue(self.gates(f)[sf.GATE_STRUCT1H].unavailable)

    def test_missing_15m_structure_blocks_in_both(self):
        f = feats()
        f.pop("ema_struct_15m")
        self.assert_agrees(f)
        self.assertFalse(self.gates(f)[sf.GATE_STRUCT15].passed)
        self.assertFalse(self.gates(f)[sf.GATE_STRUCT15_AVAIL].passed)

    def test_non_finite_rel_volume_does_not_reject(self):
        # Live: `if np.isfinite(rel_volume) and rel_volume < min`. Treating a
        # missing value as a rejection would invent a whole funnel stage.
        g = self.gates(feats(rel_volume=None))
        self.assertTrue(g[sf.GATE_RELVOL].passed)
        self.assertTrue(g[sf.GATE_RELVOL].unavailable)

    def test_absent_breadth_defaults_to_fifty_as_live_does(self):
        f = feats()
        f.pop("breadth_pct")
        g = self.gates(f)
        self.assertEqual(g[sf.GATE_BREADTH].value, 50.0)
        self.assertTrue(g[sf.GATE_BREADTH].passed)


# ================================================== 2. the individual gates
class TestGateThresholds(Base):
    def test_atr_floor_is_inclusive(self):
        self.assertTrue(self.gates(feats(atr_pct=self.a.min_atr_pct))[
            sf.GATE_ATR].passed)
        self.assertFalse(self.gates(feats(atr_pct=self.a.min_atr_pct - 1e-9))[
            sf.GATE_ATR].passed)

    def test_rel_volume_floor_is_inclusive(self):
        self.assertTrue(self.gates(feats(rel_volume=self.a.min_rel_volume))[
            sf.GATE_RELVOL].passed)
        self.assertFalse(self.gates(feats(rel_volume=self.a.min_rel_volume - 1e-9))[
            sf.GATE_RELVOL].passed)

    def test_bull_regime_vetoes_a_fully_qualified_short(self):
        g = self.gates(feats(btc_regime="bull"))
        self.assertFalse(g[sf.GATE_REGIME].passed)
        # Everything upstream passed: this is a short the rules otherwise want.
        for gate in sf.BLOCKER_GATES:
            self.assertTrue(g[gate].passed, gate)

    def test_the_veto_follows_its_config_flag(self):
        cfg = Config()
        cfg.aggressive.veto_shorts_in_btc_bull = False
        g = sf.evaluate_short_gates(feats(btc_regime="bull"), 80.0,
                                    cfg.aggressive)
        self.assertTrue(g[sf.GATE_REGIME].passed)

    def test_unknown_regime_is_a_separate_gate_from_the_veto(self):
        # They are different rules: one is a directional opinion, the other a
        # fail-closed refusal. Folding them together would make the ablation
        # "drop the veto" silently also drop a safety check.
        g = self.gates(feats(btc_regime="unknown"))
        self.assertTrue(g[sf.GATE_REGIME].passed)
        self.assertFalse(g[sf.GATE_REGIME_KNOWN].passed)

    def test_breadth_ceiling_is_inclusive(self):
        m = self.a.max_breadth_for_short
        self.assertTrue(self.gates(feats(breadth_pct=m))[sf.GATE_BREADTH].passed)
        self.assertFalse(self.gates(feats(breadth_pct=m + 1e-9))[
            sf.GATE_BREADTH].passed)

    def test_exhaustion_uses_the_shorts_own_direction(self):
        # A short is exhausted by a large move DOWN, not up.
        big_down = dict(DOWN, **{"1h": -(self.a.max_move_atr_1h + 1.0)})
        self.assertFalse(self.gates(feats(momentum_atr=big_down))[
            sf.GATE_EXHAUSTION].passed)
        big_up = dict(DOWN, **{"1h": self.a.max_move_atr_1h + 1.0})
        self.assertTrue(self.gates(feats(momentum_atr=big_up))[
            sf.GATE_EXHAUSTION].passed)

    def test_extension_reads_the_swing_low_not_the_high(self):
        # A short heads down, so its room is the distance to the swing LOW.
        beyond = -(self.a.max_extension_atr + 1.0)
        self.assertFalse(self.gates(feats(dist_to_swing_low_atr=beyond))[
            sf.GATE_EXTENSION].passed)
        self.assertTrue(self.gates(feats(dist_to_swing_high_atr=beyond))[
            sf.GATE_EXTENSION].passed)

    def test_trend_quality_takes_the_max_of_both_frames(self):
        low = self.a.min_trend_r2 - 0.05
        self.assertTrue(self.gates(feats(trend_r2_15m=low, trend_r2_1h=0.9))[
            sf.GATE_TREND_R2].passed)
        self.assertFalse(self.gates(feats(trend_r2_15m=low, trend_r2_1h=low))[
            sf.GATE_TREND_R2].passed)

    def test_score_and_confidence_are_separate_floors(self):
        # min_setup_score is 50 and min_confidence is 60. A score of 55 clears
        # the strategy and is refused by the runtime, so a diagnostic that
        # stopped at min_setup_score would report a candidate the bot rejects.
        between = (self.a.min_setup_score + self.a.min_confidence) / 2.0
        self.assertGreater(self.a.min_confidence, self.a.min_setup_score)
        g = self.gates(score=between)
        self.assertTrue(g[sf.GATE_SCORE].passed)
        self.assertFalse(g[sf.GATE_CONFIDENCE].passed)

    def test_confidence_floor_is_inclusive(self):
        g = self.gates(score=self.a.min_confidence)
        self.assertTrue(g[sf.GATE_CONFIDENCE].passed)
        g = self.gates(score=self.a.min_confidence - 1e-9)
        self.assertFalse(g[sf.GATE_CONFIDENCE].passed)

    def test_setup_score_floor_is_inclusive(self):
        # Exactly at the floor must PASS, as `score < min_setup_score` does.
        g = self.gates(score=self.a.min_setup_score)
        self.assertTrue(g[sf.GATE_SCORE].passed)
        g = self.gates(score=self.a.min_setup_score - 1e-9)
        self.assertFalse(g[sf.GATE_SCORE].passed)

    def test_the_confidence_gate_reads_min_confidence_not_the_score_floor(self):
        # By default `min_confidence` coincides with the lowest bucket edge, so
        # a gate that read `min_setup_score` instead would still behave
        # correctly and the bug would ship. Moving the floor off that edge is
        # what makes the parameter observable.
        cfg = Config()
        cfg.aggressive.min_confidence = 75.0
        g = sf.evaluate_short_gates(feats(), 70.0, cfg.aggressive)
        self.assertTrue(g[sf.GATE_SCORE].passed)
        self.assertFalse(g[sf.GATE_CONFIDENCE].passed)
        g = sf.evaluate_short_gates(feats(), 75.0, cfg.aggressive)
        self.assertTrue(g[sf.GATE_CONFIDENCE].passed)

    def test_no_fitted_confidence_transform_means_unavailable_not_assumed(self):
        # `to_confidence` raises until a calibrated mapping exists. Quietly
        # substituting the score would report candidates under a transform
        # nobody has fitted -- the exact thing `confidence.py` refuses to do.
        cfg = Config()
        cfg.aggressive.confidence_is_identity = False
        g = sf.evaluate_short_gates(feats(), 90.0, cfg.aggressive)
        self.assertTrue(g[sf.GATE_SCORE].passed)
        self.assertFalse(g[sf.GATE_CONFIDENCE].passed)
        self.assertTrue(g[sf.GATE_CONFIDENCE].unavailable)

    def test_a_zero_multiplier_bucket_is_refused_even_above_the_floor(self):
        # `is_tradable` requires BOTH the floor and a non-zero multiplier.
        cfg = Config()
        cfg.aggressive.min_confidence = 10.0
        cfg.aggressive.confidence_buckets = [{"min": 90.0, "mult": 1.0}]
        g = sf.evaluate_short_gates(feats(), 50.0, cfg.aggressive)
        self.assertFalse(g[sf.GATE_CONFIDENCE].passed)

    def test_a_missing_score_fails_rather_than_passes(self):
        g = self.gates(score=None)
        self.assertFalse(g[sf.GATE_SCORE].passed)
        self.assertTrue(g[sf.GATE_SCORE].unavailable)
        self.assertFalse(g[sf.GATE_CONFIDENCE].passed)


# ============================================ 3. the recomputed short score
class TestShortScore(Base):
    def test_rehydrate_restores_nan_through_nesting(self):
        out = sf.rehydrate({"a": None, "b": {"c": None, "d": 1.0},
                            "e": [None, 2.0], "f": "bear"})
        self.assertTrue(math.isnan(out["a"]))
        self.assertTrue(math.isnan(out["b"]["c"]))
        self.assertEqual(out["b"]["d"], 1.0)
        self.assertTrue(math.isnan(out["e"][0]))
        self.assertEqual(out["f"], "bear")

    def test_the_clean_fixture_clears_every_gate_including_the_floors(self):
        got, _ = sf.short_score(feats(), self.strat)
        self.assertAlmostEqual(got, CLEAN_SHORT_SCORE, places=2)
        g = self.gates(score=got)
        for gate in sf.ALL_GATES:
            self.assertTrue(g[gate].passed, gate)

    def test_score_is_the_strategys_own_function(self):
        f = feats()
        got, comps = sf.short_score(f, self.strat)
        want = self.strat.combine(self.strat.score_components(f, -1))
        self.assertAlmostEqual(got, want, places=12)
        self.assertEqual(set(comps), set(self.strat.score_components(f, -1)))

    def test_the_short_score_is_not_the_stored_long_score(self):
        # `observations.score` is whatever direction `choose_side` picked, and
        # 0.0 on every NO_TRADE. Reading it as a short's score is wrong twice.
        f = feats(ema_struct_5m=1.0, ema_struct_15m=1.0, ema_struct_1h=0.9,
                  momentum_atr=dict(UP), rel_strength={"1h": 3.0, "6h": 3.0})
        long_score = self.strat.combine(self.strat.score_components(f, 1))
        short, _ = sf.short_score(f, self.strat)
        self.assertGreater(long_score, short)

    def test_a_stored_feature_vector_survives_rehydration(self):
        # Straight from the DB, with nulls where NaN used to be.
        self.obs(feats(rel_volume=float("nan"), atr_expansion=float("nan")))
        f = self.funnel()
        self.assertIsNotNone(f.candidates[0].score)

    def test_candidates_carry_both_scores_so_the_difference_stays_visible(self):
        self.obs(feats(), side="long", score=88.0)
        c = self.funnel().candidates[0]
        self.assertEqual(c.recorded_side, "long")
        self.assertEqual(c.recorded_score, 88.0)
        self.assertNotAlmostEqual(c.score, 88.0, places=6)


# ==================================================== 4. counterfactual sign
class TestCounterfactualSign(Base):
    def test_a_fall_is_a_win_for_a_short(self):
        self.obs(feats(), cf=-2.0)
        c = self.funnel().candidates[0]
        self.assertAlmostEqual(c.cf_return, +2.0, places=9)

    def test_a_rise_is_a_loss_for_a_short(self):
        self.obs(feats(), cf=+3.0)
        self.assertAlmostEqual(self.funnel().candidates[0].cf_return, -3.0,
                               places=9)

    def test_the_sign_does_not_follow_the_recorded_side(self):
        # Every observation is asked the SHORT question, whatever the live code
        # recorded. Signing by the stored side would flip half the rows.
        self.obs(feats(), cf=-2.0, side="long")
        self.obs(feats(), cf=-2.0, side="short")
        for c in self.funnel().candidates:
            self.assertAlmostEqual(c.cf_return, +2.0, places=9)

    def test_no_counterfactual_means_none_not_zero(self):
        self.obs(feats())
        self.assertIsNone(self.funnel().candidates[0].cf_return)
        self.assertEqual(self.funnel()._stats(self.funnel().candidates)[
            "with_outcome"], 0)

    def test_horizon_filter_selects_one_horizon(self):
        oid = self.obs(feats(), cf=-1.0, horizons=(1,))
        self.repo.add_counterfactual(oid, 24, 100.0, 95.0, -5.0, T0)
        self.assertAlmostEqual(self.funnel().candidates[0].cf_return, 3.0,
                               places=9)
        self.assertAlmostEqual(
            self.funnel(horizon_h=24).candidates[0].cf_return, 5.0, places=9)
        self.assertAlmostEqual(
            self.funnel(horizon_h=1).candidates[0].cf_return, 1.0, places=9)

    def test_after_cost_subtracts_the_measured_drag(self):
        self.obs(feats(), cf=-2.0)
        f = self.funnel(cost_bps=50.0)
        s = f._stats(f.candidates)
        self.assertAlmostEqual(s["mean_cf_pct"], 2.0, places=9)
        self.assertAlmostEqual(s["mean_after_cost_pct"], 1.5, places=9)


# ============================================================= 5. the funnel
class TestFunnel(Base):
    def test_stages_follow_the_live_order(self):
        self.obs(feats())
        stages = [r["stage"] for r in self.funnel().funnel()][1:]
        self.assertEqual(stages, list(sf.ALL_GATES))
        # The blockers must precede the gates that only run after a side is
        # picked; a reordering would attribute rejections to the wrong rule.
        for pre in sf.BLOCKER_GATES:
            for post in sf.POST_GATES + sf.SCORE_GATES:
                self.assertLess(stages.index(pre), stages.index(post))

    def test_each_stage_reaches_what_the_previous_one_passed(self):
        for _ in range(3):
            self.obs(feats(btc_regime="bull"))
        for _ in range(2):
            self.obs(feats(momentum_atr=dict(WEAK_DOWN)))
        rows = self.funnel().funnel()
        for prev, nxt in zip(rows, rows[1:]):
            self.assertEqual(nxt["reached"], prev["reached"] - prev["rejected"])

    def test_a_gate_upstream_of_another_takes_the_rejection(self):
        # Blocked on BOTH momentum and 15m structure. Momentum runs first, so
        # the sequential funnel charges it and 15m shows zero -- which is
        # exactly the artefact the overlap view exists to correct.
        self.obs(feats(momentum_atr=dict(WEAK_DOWN), ema_struct_15m=-0.2))
        rows = {r["stage"]: r for r in self.funnel().funnel()}
        self.assertEqual(rows[sf.GATE_MOMENTUM]["rejected"], 1)
        self.assertEqual(rows[sf.GATE_STRUCT15]["rejected"], 0)

    def test_the_veto_row_counts_only_fully_qualified_shorts(self):
        self.obs(feats(btc_regime="bull"))                       # qualified
        self.obs(feats(btc_regime="bull", ema_struct_15m=0.9,
                       momentum_atr=dict(UP)))                   # not
        rows = {r["stage"]: r for r in self.funnel().funnel()}
        self.assertEqual(rows[sf.GATE_REGIME]["rejected"], 1)

    def test_rejected_rows_carry_their_own_counterfactual(self):
        self.obs(feats(btc_regime="bull"), cf=-1.0)
        self.obs(feats(), cf=+9.0)
        rows = {r["stage"]: r for r in self.funnel().funnel()}
        self.assertAlmostEqual(
            rows[sf.GATE_REGIME]["rejected_stats"]["mean_cf_pct"], 1.0,
            places=9)

    def test_unavailable_inputs_are_counted_not_hidden(self):
        f = feats()
        f.pop("ema_struct_15m")
        self.obs(f)
        rows = {r["stage"]: r for r in self.funnel().funnel()}
        self.assertEqual(rows[sf.GATE_STRUCT15_AVAIL]["unavailable"], 1)

    def test_an_empty_database_produces_a_funnel_not_a_crash(self):
        rows = self.funnel().funnel()
        self.assertEqual(rows[0]["reached"], 0)
        self.assertTrue(all(r["reached"] == 0 for r in rows))

    def test_side_contest_reports_the_tie_break(self):
        for _ in range(4):
            self.obs(feats())
        c = self.funnel().side_contest()
        self.assertEqual(c["short_clear"], 4)
        # A clear short implies a blocked long under the current mirrored
        # thresholds, so nothing should land in the contradictory bucket.
        self.assertEqual(c["short_and_long_both_clear"], 0)


# ============================================================ 6. the overlap
class TestOverlap(Base):
    def test_totals_ignore_order_where_the_funnel_cannot(self):
        self.obs(feats(momentum_atr=dict(WEAK_DOWN), ema_struct_15m=-0.2))
        o = self.funnel().overlap()
        self.assertEqual(o["totals"][sf.GATE_MOMENTUM], 1)
        self.assertEqual(o["totals"][sf.GATE_STRUCT15], 1)
        self.assertEqual(o["matrix"][sf.GATE_MOMENTUM][sf.GATE_STRUCT15], 1)

    def test_only_this_gate_excludes_the_doubly_blocked(self):
        self.obs(feats(momentum_atr=dict(WEAK_DOWN), ema_struct_15m=-0.2))
        self.obs(feats(momentum_atr=dict(WEAK_DOWN)))
        o = self.funnel().overlap(DIRECTIONAL)
        self.assertEqual(o["totals"][sf.GATE_MOMENTUM], 2)
        self.assertEqual(o["only_this_gate"][sf.GATE_MOMENTUM], 1)
        self.assertEqual(o["only_this_gate"][sf.GATE_STRUCT15], 0)

    def test_a_fully_redundant_gate_has_no_unique_reach(self):
        # 1h hostility only ever fires on bullish-structured symbols, which the
        # 15m gate already stops. Relaxing it alone would admit nothing, and
        # the recorded label cannot say so.
        for _ in range(5):
            self.obs(feats(ema_struct_5m=1.0, ema_struct_15m=1.0,
                           ema_struct_1h=0.9, momentum_atr=dict(UP)))
        o = self.funnel().overlap(DIRECTIONAL)
        self.assertEqual(o["totals"][sf.GATE_STRUCT1H], 5)
        self.assertEqual(o["only_this_gate"][sf.GATE_STRUCT1H], 0)

    def test_restricting_the_gate_set_changes_only_this_gate(self):
        # With the score floor in scope a low-scoring row's structural gate
        # looks redundant, because the floor blocks it anyway. Excluding the
        # floors is what makes the structural question answerable.
        self.obs(low_score_feats(momentum_atr=dict(WEAK_DOWN)))
        allg = self.funnel().overlap()
        dirg = self.funnel().overlap(DIRECTIONAL)
        self.assertEqual(allg["only_this_gate"][sf.GATE_MOMENTUM], 0)
        self.assertEqual(dirg["only_this_gate"][sf.GATE_MOMENTUM], 1)

    def test_histogram_counts_gates_per_observation(self):
        self.obs(feats())                                          # 0 blockers
        self.obs(feats(momentum_atr=dict(WEAK_DOWN)))               # 1
        self.obs(feats(momentum_atr=dict(WEAK_DOWN), ema_struct_15m=-0.2))
        h = self.funnel().overlap(DIRECTIONAL)["n_blockers_histogram"]
        self.assertEqual(h, {"0": 1, "1": 1, "2": 1})


# ========================================================== 7. the ablations
class TestAblations(Base):
    def rows(self):
        return {a["variant"]: a for a in self.funnel().ablations()}

    def test_dropping_the_veto_admits_the_vetoed_shorts(self):
        for _ in range(7):
            self.obs(feats(btc_regime="bull"), cf=-1.0)
        r = self.rows()["drop the bullish-BTC short veto"]
        self.assertEqual(r["extra_directional"], 7)
        self.assertAlmostEqual(r["extra_directional_stats"]["mean_cf_pct"],
                               1.0, places=9)

    def test_dropping_a_gate_that_blocks_nothing_admits_nothing(self):
        for _ in range(3):
            self.obs(feats())
        for variant, row in self.rows().items():
            self.assertEqual(row["extra_directional"], 0, variant)

    def test_the_combined_variant_is_at_least_the_sum_of_its_parts(self):
        self.obs(feats(momentum_atr=dict(WEAK_DOWN)))
        self.obs(feats(ema_struct_15m=-0.2))
        self.obs(feats(momentum_atr=dict(WEAK_DOWN), ema_struct_15m=-0.2))
        r = self.rows()
        self.assertEqual(r["relax short momentum agreement"]["extra_directional"], 1)
        self.assertEqual(r["relax 15m bearish-structure"]["extra_directional"], 1)
        # The doubly-blocked row only appears when BOTH are relaxed.
        self.assertEqual(r["relax momentum + 15m together"]["extra_directional"], 3)

    def test_the_directional_and_full_readings_are_reported_separately(self):
        # Score 0 fails the floor, so relaxing a structural gate admits a
        # candidate directionally and none at all in practice. Collapsing the
        # two would claim a gate change buys trades it cannot.
        for _ in range(4):
            self.obs(low_score_feats(btc_regime="bull"))
        r = self.rows()["drop the bullish-BTC short veto"]
        self.assertEqual(r["extra_directional"], 4)
        self.assertEqual(r["extra_admitted"], 0)

    def test_a_low_scoring_fixture_really_only_fails_the_floors(self):
        # Guards the fixture itself: if it started failing a structural gate
        # too, the test above would pass for the wrong reason.
        g = sf.evaluate_short_gates(
            low_score_feats(),
            sf.short_score(low_score_feats(), self.strat)[0], self.a)
        for gate in DIRECTIONAL:
            self.assertTrue(g[gate].passed, gate)
        self.assertFalse(g[sf.GATE_SCORE].passed)
        self.assertFalse(g[sf.GATE_CONFIDENCE].passed)

    def test_the_baseline_variant_admits_no_extras(self):
        for _ in range(3):
            self.obs(feats(btc_regime="bull"))
        self.obs(feats())
        r = self.rows()["current short rules"]
        self.assertEqual(r["extra_directional"], 0)
        self.assertEqual(r["extra_admitted"], 0)
        self.assertEqual(r["baseline"], 1)
        self.assertEqual(r["baseline_directional"], 1)

    def test_dropping_the_veto_admits_shorts_that_clear_every_other_gate(self):
        # The full reading, not just the directional one: these are shorts the
        # bot would actually have taken. That is what makes the veto the
        # answer to "why zero shorts" rather than one contributor among many.
        for _ in range(5):
            self.obs(feats(btc_regime="bull"), cf=-1.0)
        r = self.rows()["drop the bullish-BTC short veto"]
        self.assertEqual(r["extra_directional"], 5)
        self.assertEqual(r["extra_admitted"], 5)

    def test_replayability_is_reported_per_variant(self):
        for _ in range(3):
            self.obs(feats(btc_regime="bull"), cf=-1.0)
        r = self.rows()["drop the bullish-BTC short veto"]
        # No tape was stored, so stop/TP P&L is not reproducible for any of
        # them -- the warning the report prints depends on this being honest.
        self.assertEqual(r["extra_directional_stats"]["replayable"], 0)
        self.assertFalse(r["extra_directional_stats"]["pnl_reproducible"])


# ====================================================== 8. the score floors
class TestScoreFloors(Base):
    def test_realised_totals_only_the_trades_a_floor_keeps(self):
        self.trade(score=65.0)
        self.trade(score=85.0)
        out = self.funnel().score_floors((60.0, 80.0))
        by = {r["floor"]: r for r in out["realised"]}
        self.assertEqual(by[60.0]["closed_trades"], 2)
        self.assertEqual(by[80.0]["closed_trades"], 1)
        self.assertEqual(by[80.0]["dropped"], 1)

    def test_a_floor_is_inclusive_of_its_own_value(self):
        self.trade(score=70.0)
        by = {r["floor"]: r for r in
              self.funnel().score_floors((70.0, 70.1))["realised"]}
        self.assertEqual(by[70.0]["closed_trades"], 1)
        self.assertEqual(by[70.1]["closed_trades"], 0)

    def test_trades_without_a_recorded_score_are_counted_not_dropped_silently(self):
        self.trade(score=90.0)
        self.trade(journal_score=False)
        out = self.funnel().score_floors((60.0,))
        self.assertEqual(out["total_closed_trades"], 2)
        self.assertEqual(out["trades_without_recorded_score"], 1)
        self.assertEqual(out["realised"][0]["closed_trades"], 1)

    def test_realised_and_hypothetical_stay_in_separate_structures(self):
        self.trade(score=90.0)
        self.obs(feats(), cf=-1.0)
        out = self.funnel().score_floors((60.0,))
        self.assertIn("net_pnl", out["realised"][0])
        self.assertNotIn("net_pnl", out["hypothetical"][0])
        self.assertIn("mean_cf_pct", out["hypothetical"][0])
        self.assertNotIn("mean_cf_pct", out["realised"][0])
        self.assertTrue(out["warning"])
        self.assertTrue(out["caveats"])

    def test_net_bps_normalises_the_shadow_ledger_by_turnover(self):
        self.trade(score=90.0, qty=25.0)
        r = self.funnel().score_floors((60.0,))["realised"][0]
        self.assertAlmostEqual(
            r["net_bps"], r["net_pnl"] / r["turnover"] * 10_000.0, places=9)

    def test_the_short_score_distribution_is_reported(self):
        self.obs(feats())
        out = self.funnel().score_floors((60.0,))
        self.assertEqual(out["short_score_computable"], 1)
        self.assertEqual(out["short_score_missing"], 0)
        self.assertEqual(sum(out["short_score_histogram"].values()), 1)


# ======================================================== 9. the cost split
class TestCostSplit(Base):
    def split(self):
        return sf.cost_split(self.repo, B, self.cfg.execution)

    def test_the_fee_split_follows_the_notional_not_a_half(self):
        # Both legs pay the same taker rate, so the stored total splits by
        # notional EXACTLY. Halving it would misreport the exit leg on every
        # trade that moved, which is every trade.
        qty, entry_fill, exit_fill = 25.0, 100.0, 80.0
        self.trade(qty=qty, entry_ref=entry_fill, entry_fill=entry_fill,
                   exit_ref=exit_fill, exit_fill=exit_fill, reason="time")
        c = self.split()
        e_not, x_not = qty * entry_fill, qty * exit_fill
        self.assertAlmostEqual(c["entry_fee"],
                               c["fees_ledger"] * e_not / (e_not + x_not),
                               places=9)
        self.assertGreater(c["entry_fee"], c["exit_fee"])
        self.assertNotAlmostEqual(c["entry_fee"], c["fees_ledger"] / 2.0,
                                  places=4)

    def test_the_fee_split_reconciles_against_the_ledger(self):
        self.trade()
        self.trade(side="short", entry_ref=100.0, entry_fill=99.94,
                   exit_ref=101.8, exit_fill=101.95, reason="stop")
        c = self.split()
        self.assertAlmostEqual(c["fee_residual"], 0.0, places=9)
        self.assertAlmostEqual(c["entry_fee"] + c["exit_fee"],
                               c["fees_ledger"], places=9)

    def test_the_slippage_split_reconciles_against_the_ledger(self):
        self.trade()
        self.trade(side="short", entry_ref=100.0, entry_fill=99.94,
                   exit_ref=101.8, exit_fill=101.95)
        c = self.split()
        self.assertAlmostEqual(c["slippage_residual"], 0.0, places=9)

    def test_entry_slippage_is_a_cost_on_both_sides(self):
        self.trade(entry_ref=100.0, entry_fill=100.06)              # long
        self.trade(side="short", entry_ref=100.0, entry_fill=99.94,
                   exit_ref=101.8, exit_fill=101.8)
        # A long fills above its reference and a short below; both are costs.
        self.assertGreater(self.split()["entry_slippage"], 0.0)

    def test_a_gapped_stop_is_separated_from_the_modelled_bps(self):
        bps = self.cfg.execution.stop_slippage_bps
        stop = 98.2
        clean = stop * (1.0 - bps / 10_000.0)
        self.trade(exit_ref=stop, exit_fill=clean, reason="stop")
        c = self.split()
        self.assertEqual(c["gapped_exits"], 0)
        self.assertAlmostEqual(c["gap_component"], 0.0, places=9)
        self.assertAlmostEqual(c["stop_slippage_modelled"],
                               c["exit_slippage_stops"], places=9)

    def test_a_gap_through_the_stop_is_attributed_to_the_gap(self):
        bps = self.cfg.execution.stop_slippage_bps
        stop, gap_open = 98.2, 97.0
        self.trade(exit_ref=stop, exit_fill=gap_open * (1.0 - bps / 10_000.0),
                   reason="stop_gap", qty=25.0)
        c = self.split()
        self.assertEqual(c["gapped_exits"], 1)
        # The gap is the distance the bar OPENED beyond the stop, not a bps
        # assumption -- so it is the stop-to-open distance times the size.
        self.assertAlmostEqual(c["gap_component"], (stop - gap_open) * 25.0,
                               places=6)
        self.assertAlmostEqual(c["gap_residual"], 0.0, places=6)

    def test_a_short_gapping_upward_is_also_a_gap(self):
        bps = self.cfg.execution.stop_slippage_bps
        stop, gap_open = 101.8, 103.0
        self.trade(side="short", entry_ref=100.0, entry_fill=100.0,
                   exit_ref=stop, exit_fill=gap_open * (1.0 + bps / 10_000.0),
                   reason="stop_gap", qty=25.0)
        c = self.split()
        self.assertEqual(c["gapped_exits"], 1)
        self.assertAlmostEqual(c["gap_component"], (gap_open - stop) * 25.0,
                               places=6)
        self.assertAlmostEqual(c["gap_residual"], 0.0, places=6)

    def test_non_stop_exits_are_held_out_of_the_gap_identity(self):
        # A target exit has slippage too, and letting it fall into the residual
        # would make a correct reconstruction read as a broken one.
        self.trade(exit_ref=102.0, exit_fill=101.9, reason="target")
        c = self.split()
        self.assertEqual(c["stop_exits"], 0)
        self.assertAlmostEqual(c["gap_residual"], 0.0, places=9)
        self.assertAlmostEqual(c["exit_slippage_non_stop"], c["exit_slippage"],
                               places=9)

    def test_the_split_reads_stop_slippage_from_the_execution_config(self):
        # It lives on ExecutionCfg, not AggressiveCfg. Reading the wrong object
        # silently returns a default and the gap attribution goes with it.
        cfg = Config()
        cfg.execution.stop_slippage_bps = 40.0
        stop = 98.2
        fill = stop * (1.0 - 40.0 / 10_000.0)
        self.trade(exit_ref=stop, exit_fill=fill, reason="stop")
        c = sf.cost_split(self.repo, B, cfg.execution)
        self.assertEqual(c["gapped_exits"], 0)
        # With the DEFAULT 15 bps the same fill looks like a gap it is not.
        wrong = sf.cost_split(self.repo, B, Config().execution)
        self.assertEqual(wrong["gapped_exits"], 1)

    def test_financing_is_carried_separately(self):
        self.trade(side="short", entry_ref=100.0, entry_fill=100.0,
                   exit_ref=99.0, exit_fill=99.0, financing=3.25)
        self.assertAlmostEqual(self.split()["financing"], 3.25, places=9)

    def test_bps_are_expressed_against_entry_turnover(self):
        self.trade(qty=25.0, entry_fill=100.0)
        c = self.split()
        self.assertAlmostEqual(c["turnover"], 2500.0, places=6)
        self.assertAlmostEqual(c["bps"]["net_pnl"],
                               c["net_pnl"] / 2500.0 * 10_000.0, places=9)

    def test_an_empty_ledger_reports_no_bps_rather_than_dividing_by_zero(self):
        c = self.split()
        self.assertEqual(c["n_trades"], 0)
        self.assertNotIn("bps", c)


# ================================================ 10. the measured cost drag
class TestMeasuredCost(Base):
    def test_drag_is_measured_from_the_ledger_when_trades_exist(self):
        self.trade(qty=25.0, entry_fill=100.0)
        f = self.funnel()
        self.assertTrue(f.measured_cost)
        c = f.cost_split()
        self.assertAlmostEqual(
            f.cost_bps,
            (c["fees_ledger"] + c["slippage_ledger"] + c["financing"])
            / c["turnover"] * 10_000.0, places=9)

    def test_the_default_is_flagged_as_a_default_not_a_measurement(self):
        f = self.funnel()
        self.assertFalse(f.measured_cost)
        self.assertEqual(f.cost_bps, sf.DEFAULT_COST_BPS)

    def test_an_explicit_override_is_not_reported_as_measured(self):
        self.trade()
        f = self.funnel(cost_bps=12.5)
        self.assertFalse(f.measured_cost)
        self.assertEqual(f.cost_bps, 12.5)


# =============================================== 11. size versus quality
class TestScoreBucketCosts(Base):
    def test_bps_separates_quality_from_position_size(self):
        # Same dollar loss, four times the notional. Dollars say they are
        # equally bad; bps say the larger one is four times better per dollar
        # risked. Only the second reading answers "is the SCORE working".
        self.trade(score=65.0, qty=25.0, exit_ref=99.0, exit_fill=99.0,
                   entry_fill=100.0, reason="time")
        self.trade(score=85.0, qty=100.0, exit_ref=99.75, exit_fill=99.75,
                   entry_fill=100.0, reason="time")
        by = {b["bucket"]: b for b in self.funnel().score_bucket_costs()}
        lo, hi = by["60-70"], by["80-90"]
        self.assertAlmostEqual(lo["gross_pnl"], hi["gross_pnl"], places=6)
        self.assertLess(hi["turnover"] / 3.0, hi["turnover"])
        self.assertGreater(hi["gross_bps"], lo["gross_bps"])

    def test_gross_bps_is_net_plus_the_cost_it_paid(self):
        self.trade(score=75.0)
        b = self.funnel().score_bucket_costs()[0]
        self.assertAlmostEqual(b["gross_bps"], b["net_bps"] + b["cost_bps"],
                               places=6)

    def test_buckets_sort_numerically_not_lexically(self):
        for s in (65.0, 75.0, 85.0, 95.0, 105.0):
            self.trade(score=s)
        got = [b["bucket"] for b in self.funnel().score_bucket_costs()]
        self.assertEqual(got, ["60-70", "70-80", "80-90", "90-100", "100-110"])

    def test_trades_without_a_score_are_excluded_from_the_buckets(self):
        self.trade(score=75.0)
        self.trade(journal_score=False)
        self.assertEqual(sum(b["n"] for b in self.funnel().score_bucket_costs()),
                         1)


if __name__ == "__main__":
    unittest.main()
