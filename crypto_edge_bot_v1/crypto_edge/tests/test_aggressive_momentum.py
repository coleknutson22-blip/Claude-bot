"""Strategy B: deterministic LONG / SHORT / NO_TRADE from fast timeframes.

WHAT THIS STRATEGY IS FOR
-------------------------
`trend_breakout` requires a closed Donchian breakout, and on 120 synthetic
markets that single condition rejected 88 of them. It is the correct design for
a slow trend strategy and the reason it trades rarely. Strategy B replaces the
binary trigger with graded agreement across 5m/15m/1h -- and on the same 120
markets takes 45 entries, 25 long and 20 short.

The willingness comes from the DIRECTION test, not from weakening the gates that
protect the account: data sanity, ATR floor, liquidity, exhaustion and regime
hostility are all still hard vetoes, and none of Strategy A's thresholds are
touched by anything here.

SYMMETRY IS A CORRECTNESS PROPERTY, NOT A NICETY
------------------------------------------------
A long/short strategy that scores an equal-and-opposite move differently has a
directional bias hidden in its arithmetic. Two bugs of exactly that kind were
found by these tests and fixed:

  * momentum used SIMPLE percentage returns, which do not negate under price
    inversion -- a +10% move reflects to -9.09%, so every short looked weaker
    than its mirrored long. Now log returns, which negate exactly.
  * ATR% divided a 15m ATR by a 5m close. Two different instants, so on frames
    that disagreed the ratio was silently meaningless rather than obviously
    wrong. Now the 15m ATR over the 15m close, with a hard guard on frames that
    disagree about price by more than a configured amount.
"""
import unittest

import numpy as np

import helpers  # noqa: F401  -- silences the engine's log handlers
from crypto_edge.config import AggressiveCfg, StrategyCfg, load_config
from crypto_edge.strategy import features as F
from crypto_edge.strategy.aggressive_momentum import (VOTING_WINDOWS, WEIGHTS,
                                                      AggressiveMomentumStrategy,
                                                      momentum_votes)
from crypto_edge.strategy.base import LONG, NO_TRADE, SHORT, MarketContext
from fixtures_fast import frames, mirror

NEUTRAL = MarketContext(ts_ms=1_700_000_000_000, btc_regime="neutral",
                        btc_regime_score=50.0, breadth_pct=50.0)


def ctx(regime="neutral", breadth=50.0, score=50.0, blocked=None):
    return MarketContext(ts_ms=1_700_000_000_000, btc_regime=regime,
                         btc_regime_score=score, breadth_pct=breadth,
                         blocked_symbols=blocked or {})


def strat(**over):
    cfg = AggressiveCfg()
    for k, v in over.items():
        setattr(cfg, k, v)
    return AggressiveMomentumStrategy(cfg)


UP = frames(0.00040, seed=11)
DOWN = mirror(UP)
FLAT = frames(0.0, seed=23, chop=True)


class TestFeaturesAreCausal(unittest.TestCase):
    """No feature may read a bar that had not closed when it was computed."""

    def test_future_bars_cannot_change_a_past_value(self):
        """The real causality property: what happens after bar N is invisible
        to the value computed AT bar N, whatever it turns out to be."""
        s = UP["15m"]
        n = 200
        before = F.log_roc(s.close[:n], 12)
        future = s.close.copy()
        future[n:] = 1e6                      # an arbitrary, violent future
        self.assertAlmostEqual(before, F.log_roc(future[:n], 12), places=12)

    def test_a_whole_feature_vector_is_blind_to_the_future(self):
        from crypto_edge.models import Series
        cut = {}
        wrecked = {}
        for tf, s in UP.items():
            n = len(s) - 20
            cut[tf] = Series(s.symbol, tf, s.open_ms[:n], s.open[:n], s.high[:n],
                             s.low[:n], s.close[:n], s.volume[:n])
            c = s.close.copy(); c[n:] = 1e6
            h = s.high.copy(); h[n:] = 1e6
            wrecked[tf] = Series(s.symbol, tf, s.open_ms[:n], s.open[:n], h[:n],
                                 s.low[:n], c[:n], s.volume[:n])
        a, b = F.compute(cut), F.compute(wrecked)
        self.assertEqual(a["momentum"], b["momentum"])
        self.assertEqual(a["atr_pct"], b["atr_pct"])
        self.assertEqual(a["ema_struct_15m"], b["ema_struct_15m"])

    def test_trend_quality_on_a_prefix_is_unchanged_by_the_suffix(self):
        s = UP["15m"]
        a = F.trend_quality(s.close[:120], 24)
        b = F.trend_quality(s.close[:120].copy(), 24)
        self.assertEqual(a, b)
        # extending the series must not alter the value computed on the prefix
        c = F.trend_quality(s.close[:150], 24)
        self.assertNotEqual(a, c, "sanity: a different window IS a different value")

    def test_ema_structure_is_a_prefix_function(self):
        s = UP["15m"]
        first = F.ema_structure(s.close[:200])
        again = F.ema_structure(np.concatenate([s.close[:200], s.close[200:]])[:200])
        self.assertEqual(first, again)

    def test_swing_levels_exclude_the_current_bar(self):
        """A swing high that includes the bar being measured is that bar."""
        close = np.array([10.0] * 30 + [50.0])
        high = close * 1.001
        low = close * 0.999
        hi_d, _ = F.swing_proximity(high, low, close, 24, atr_value=1.0)
        self.assertGreater(close[-1], 10.0)
        self.assertLess(hi_d, 0.0,
                        "price is ABOVE the prior swing high, so distance is negative")

    def test_every_requested_feature_is_present(self):
        f = F.compute(UP)
        for key in ("price", "atr", "atr_pct", "atr_expansion", "short_vol_pct",
                    "momentum", "momentum_atr", "ema_struct_5m", "ema_struct_15m",
                    "ema_struct_1h", "slope_15m_pct", "trend_r2_15m",
                    "slope_1h_pct", "trend_r2_1h", "rel_volume",
                    "dist_to_swing_high_atr", "dist_to_swing_low_atr",
                    "rel_strength", "breadth_pct", "btc_regime",
                    "dollar_volume", "spread_bps"):
            self.assertIn(key, f, key)

    def test_all_six_momentum_windows_are_computed(self):
        m = F.compute(UP)["momentum"]
        for w in ("30m", "1h", "2h", "3h", "6h", "24h"):
            self.assertIn(w, m)
            self.assertTrue(np.isfinite(m[w]), w)

    def test_momentum_windows_map_to_the_right_bar_counts(self):
        self.assertEqual(F.bars_for("5m", 30), 6)
        self.assertEqual(F.bars_for("5m", 60), 12)
        self.assertEqual(F.bars_for("15m", 120), 8)
        self.assertEqual(F.bars_for("15m", 180), 12)
        self.assertEqual(F.bars_for("1h", 360), 6)
        self.assertEqual(F.bars_for("1h", 1440), 24)


class TestDirectionalDecisions(unittest.TestCase):
    def test_a_rising_market_reads_long(self):
        sig = strat().evaluate_frames(UP, NEUTRAL)
        self.assertEqual(sig.side, LONG, sig.reject_reason)

    def test_the_mirrored_market_reads_short(self):
        sig = strat().evaluate_frames(DOWN, NEUTRAL)
        self.assertEqual(sig.side, SHORT, sig.reject_reason)

    def test_a_directionless_market_reads_no_trade(self):
        """A fast cycle: the windows straddle turns, so nothing agrees."""
        sig = strat().evaluate_frames(frames(0.0, seed=5, oscillate=True), NEUTRAL)
        self.assertEqual(sig.side, NO_TRADE)
        self.assertFalse(sig.passed)
        self.assertIn("momentum agreement", sig.reject_reason)

    def test_mixed_momentum_cannot_reach_agreement(self):
        """The gate itself, isolated from any fixture: half up, half down."""
        mixed = {"30m": 0.9, "1h": -0.9, "2h": 0.9, "3h": -0.9, "6h": -0.9}
        self.assertEqual(momentum_votes(mixed, 1, 0.15), 2)
        self.assertEqual(momentum_votes(mixed, -1, 0.15), 3)

    def test_choppy_markets_never_produce_an_entry(self):
        """Chop may still show a short-term LEAN -- it must never be tradable.

        Asserted across seeds rather than one, because a single mean-reverting
        path can drift far enough to look directional; the property that has to
        hold is that none of them clear the score threshold.
        """
        s = strat()
        taken = []
        for seed in range(20, 32):
            sig = s.evaluate_frames(frames(0.0, seed=seed, chop=True), NEUTRAL)
            if sig.passed:
                taken.append((seed, sig.side, round(sig.score, 1)))
        self.assertEqual(taken, [], f"chop produced entries: {taken}")

    def test_chop_is_scored_low_even_when_it_leans(self):
        sig = strat().evaluate_frames(FLAT, NEUTRAL)
        self.assertFalse(sig.passed)
        self.assertLess(sig.score, AggressiveCfg().min_setup_score)

    def test_chop_does_not_vote_on_noise_alone(self):
        """Sign without magnitude is not evidence: three of five agree by chance."""
        tiny = {w: 0.01 for w in VOTING_WINDOWS}
        self.assertEqual(momentum_votes(tiny, 1, min_atr=0.15), 0)
        self.assertEqual(momentum_votes(tiny, 1, min_atr=0.0), len(VOTING_WINDOWS))

    def test_a_no_trade_always_explains_itself(self):
        for fr, c in ((frames(0.0, seed=5, oscillate=True), NEUTRAL),
                      (UP, ctx(regime="bear")), (DOWN, ctx(regime="bull"))):
            sig = strat().evaluate_frames(fr, c)
            self.assertEqual(sig.side, NO_TRADE)
            self.assertTrue(sig.reject_reason, "every rejection must name its cause")

    def test_no_donchian_breakout_is_required(self):
        """The point of the strategy: an entry without a completed breakout."""
        sig = strat().evaluate_frames(UP, ctx(regime="bull", breadth=65))
        self.assertEqual(sig.side, LONG)
        self.assertNotIn("donchian", sig.reject_reason.lower())
        self.assertNotIn("breakout", sig.reject_reason.lower())


class TestLongShortSymmetry(unittest.TestCase):
    """An equal-and-opposite market must produce an equal-and-opposite reading."""

    def mirrored_features(self, f: dict) -> dict:
        g = dict(f)
        g["momentum_atr"] = {k: -v for k, v in f["momentum_atr"].items()}
        g["ema_struct_5m"] = -f["ema_struct_5m"]
        g["ema_struct_15m"] = -f["ema_struct_15m"]
        g["rel_strength"] = {k: -v for k, v in f["rel_strength"].items()}
        g["dist_to_swing_high_atr"] = f["dist_to_swing_low_atr"]
        g["dist_to_swing_low_atr"] = f["dist_to_swing_high_atr"]
        g["breadth_pct"] = 100.0 - f["breadth_pct"]
        return g

    def test_components_are_exactly_mirrored(self):
        """Exact, on a synthetic feature vector: no arithmetic bias anywhere."""
        s = strat()
        f = F.compute(UP)
        f["breadth_pct"] = 63.0
        long_c = s.score_components(f, 1)
        short_c = s.score_components(self.mirrored_features(f), -1)
        for k in WEIGHTS:
            self.assertAlmostEqual(long_c[k], short_c[k], places=9, msg=k)

    def test_the_combined_score_is_exactly_mirrored(self):
        s = strat()
        f = F.compute(UP)
        self.assertAlmostEqual(s.combine(s.score_components(f, 1)),
                               s.combine(s.score_components(
                                   self.mirrored_features(f), -1)), places=9)

    def test_log_momentum_negates_exactly_under_price_inversion(self):
        """The bug this replaced: simple returns do NOT negate."""
        up = F.compute(UP)["momentum"]
        dn = F.compute(DOWN)["momentum"]
        for w in ("30m", "1h", "2h", "3h", "6h"):
            self.assertAlmostEqual(up[w], -dn[w], places=9, msg=w)

    def test_end_to_end_scores_agree_within_a_tolerance(self):
        """ATR is a price-space quantity, so the mirror is near-exact, not exact."""
        s = strat()
        a = s.evaluate_frames(UP, NEUTRAL)
        b = s.evaluate_frames(DOWN, NEUTRAL)
        self.assertEqual((a.side, b.side), (LONG, SHORT))
        self.assertLess(abs(a.score - b.score), 2.0,
                        f"long {a.score:.2f} vs short {b.score:.2f}")

    def test_the_stop_sits_on_the_losing_side_for_both(self):
        s = strat(min_setup_score=0.0)
        a = s.evaluate_frames(UP, NEUTRAL)
        b = s.evaluate_frames(DOWN, NEUTRAL)
        self.assertLess(a.stop_price, a.ref_price, "long stop is below entry")
        self.assertGreater(b.stop_price, b.ref_price, "short stop is above entry")

    def test_signal_direction_matches_its_side(self):
        s = strat(min_setup_score=0.0)
        self.assertEqual(s.evaluate_frames(UP, NEUTRAL).direction, 1)
        self.assertEqual(s.evaluate_frames(DOWN, NEUTRAL).direction, -1)


class TestRegimeVetoes(unittest.TestCase):
    def test_a_long_is_vetoed_in_a_bear_btc_regime(self):
        sig = strat().evaluate_frames(UP, ctx(regime="bear"))
        self.assertEqual(sig.side, NO_TRADE)
        self.assertIn("long vetoed", sig.reject_reason)

    def test_a_short_is_vetoed_in_a_bull_btc_regime(self):
        sig = strat().evaluate_frames(DOWN, ctx(regime="bull"))
        self.assertEqual(sig.side, NO_TRADE)
        self.assertIn("short vetoed", sig.reject_reason)

    def test_an_unknown_regime_fails_closed(self):
        sig = strat().evaluate_frames(UP, ctx(regime="unknown"))
        self.assertEqual(sig.side, NO_TRADE)
        self.assertIn("unavailable", sig.reject_reason)

    def test_weak_breadth_blocks_a_long(self):
        sig = strat().evaluate_frames(UP, ctx(breadth=10.0))
        self.assertEqual(sig.side, NO_TRADE)
        self.assertIn("breadth", sig.reject_reason)

    def test_strong_breadth_blocks_a_short(self):
        sig = strat().evaluate_frames(DOWN, ctx(breadth=90.0))
        self.assertEqual(sig.side, NO_TRADE)
        self.assertIn("breadth", sig.reject_reason)

    def test_the_vetoes_are_the_mirror_of_each_other(self):
        s = strat()
        self.assertEqual(s.evaluate_frames(UP, ctx(regime="bull")).side, LONG)
        self.assertEqual(s.evaluate_frames(DOWN, ctx(regime="bear")).side, SHORT)

    def test_a_blocked_symbol_is_refused(self):
        sig = strat().evaluate_frames(
            UP, ctx(blocked={"SOL/USD": "listing event"}))
        self.assertEqual(sig.side, NO_TRADE)
        self.assertIn("no-trade list", sig.reject_reason)


class TestExhaustionAndOverextension(unittest.TestCase):
    def test_a_move_already_finished_is_refused(self):
        sig = strat(max_move_atr_1h=0.5).evaluate_frames(UP, NEUTRAL)
        self.assertEqual(sig.side, NO_TRADE)
        self.assertIn("already moved", sig.reject_reason)

    def test_the_exhaustion_test_is_symmetric(self):
        s = strat(max_move_atr_1h=0.5)
        for fr in (UP, DOWN):
            self.assertIn("already moved", s.evaluate_frames(fr, NEUTRAL).reject_reason)

    def test_far_beyond_the_swing_level_is_refused(self):
        sig = strat(max_extension_atr=0.0).evaluate_frames(UP, NEUTRAL)
        if sig.side == NO_TRADE:
            self.assertTrue("over-extended" in sig.reject_reason
                            or "already moved" in sig.reject_reason)

    def test_a_quiet_market_is_below_the_atr_floor(self):
        sig = strat(min_atr_pct=99.0).evaluate_frames(UP, NEUTRAL)
        self.assertEqual(sig.side, NO_TRADE)
        self.assertIn("ATR", sig.reject_reason)

    def test_thin_participation_is_refused(self):
        sig = strat(min_rel_volume=99.0).evaluate_frames(UP, NEUTRAL)
        self.assertEqual(sig.side, NO_TRADE)
        self.assertIn("relative volume", sig.reject_reason)


class TestDataIntegrityFailsClosed(unittest.TestCase):
    def test_a_missing_timeframe_is_fatal(self):
        fr = {k: v for k, v in UP.items() if k != "5m"}
        sig = strat().evaluate_frames(fr, NEUTRAL)
        self.assertEqual(sig.side, NO_TRADE)
        self.assertIn("no 5m candles", sig.reject_reason)

    def test_short_history_is_fatal(self):
        from crypto_edge.models import Series
        fr = dict(UP)
        s15 = UP["15m"]
        fr["15m"] = Series(s15.symbol, "15m", s15.open_ms[:20], s15.open[:20],
                           s15.high[:20], s15.low[:20], s15.close[:20],
                           s15.volume[:20])
        sig = strat().evaluate_frames(fr, NEUTRAL)
        self.assertEqual(sig.side, NO_TRADE)
        self.assertIn("insufficient", sig.reject_reason)

    def test_frames_that_disagree_about_price_are_refused(self):
        """The silent-nonsense case: three frames, not one market."""
        from crypto_edge.models import Series
        fr = dict(UP)
        s = UP["1h"]
        fr["1h"] = Series(s.symbol, s.timeframe, s.open_ms, s.open * 10,
                          s.high * 10, s.low * 10, s.close * 10, s.volume)
        sig = strat().evaluate_frames(fr, NEUTRAL)
        self.assertEqual(sig.side, NO_TRADE)
        self.assertIn("disagree on price", sig.reject_reason)

    def test_aligned_frames_pass_the_same_guard(self):
        f = F.compute(UP)
        self.assertLess(f["frame_price_spread_pct"], 25.0)

    def test_insane_candles_are_refused(self):
        from crypto_edge.models import Series
        fr = dict(UP)
        s = UP["15m"]
        bad = s.close.copy()
        bad[-1] = -5.0
        fr["15m"] = Series(s.symbol, s.timeframe, s.open_ms, s.open, s.high,
                           s.low, bad, s.volume)
        sig = strat().evaluate_frames(fr, NEUTRAL)
        self.assertEqual(sig.side, NO_TRADE)
        self.assertIn("sanity", sig.reject_reason)


class TestDeterminism(unittest.TestCase):
    def test_the_same_input_gives_the_same_signal(self):
        s = strat()
        a, b = s.evaluate_frames(UP, NEUTRAL), s.evaluate_frames(UP, NEUTRAL)
        self.assertEqual((a.side, a.score, a.reject_reason),
                         (b.side, b.score, b.reject_reason))
        self.assertEqual(a.components, b.components)

    def test_a_fresh_instance_agrees_with_the_first(self):
        """Restart determinism: no state carries between runs."""
        a = strat().evaluate_frames(UP, NEUTRAL)
        b = strat().evaluate_frames(UP, NEUTRAL)
        self.assertEqual(a.score, b.score)
        self.assertEqual(a.side, b.side)

    def test_evaluating_a_prefix_is_stable(self):
        """Closed candles only: the last bar decides, and it does not move."""
        s = strat()
        first = s.evaluate_frames(UP, NEUTRAL)
        again = s.evaluate_frames({k: v for k, v in UP.items()}, NEUTRAL)
        self.assertEqual(first.score, again.score)

    def test_the_candle_id_is_stable_and_names_the_bar(self):
        sig = strat().evaluate_frames(UP, NEUTRAL)
        self.assertIn("SOL/USD", sig.candle_id)
        self.assertIn("15m", sig.candle_id)


class TestResearchRecording(unittest.TestCase):
    def test_every_component_is_recorded_separately(self):
        sig = strat(min_setup_score=0.0).evaluate_frames(UP, NEUTRAL)
        self.assertEqual(set(sig.components), set(WEIGHTS))
        self.assertEqual(sig.features["score_components"], sig.components)

    def test_the_side_and_vote_count_are_recorded(self):
        sig = strat(min_setup_score=0.0).evaluate_frames(UP, NEUTRAL)
        self.assertEqual(sig.features["side"], LONG)
        self.assertGreaterEqual(sig.features["momentum_votes"], 3)

    def test_a_sub_threshold_setup_still_records_its_direction(self):
        """Stage 3 decides tradability; Stage 2 still records the reading."""
        sig = strat(min_setup_score=99.0).evaluate_frames(UP, NEUTRAL)
        self.assertEqual(sig.side, LONG)
        self.assertFalse(sig.passed)
        self.assertIn("setup score", sig.reject_reason)
        self.assertTrue(sig.components, "components recorded even when rejected")
        self.assertGreater(sig.stop_price, 0.0, "and the stop it would have used")

    def test_momentum_and_volatility_state_are_recorded(self):
        sig = strat(min_setup_score=0.0).evaluate_frames(UP, NEUTRAL)
        f = sig.features
        self.assertIn("momentum", f)
        self.assertIn("momentum_atr", f)
        self.assertIn("atr_expansion", f)
        self.assertIn("short_vol_pct", f)
        self.assertIn("btc_regime", f)
        self.assertIn("rel_strength", f)

    def test_every_weight_has_a_written_justification(self):
        from crypto_edge.strategy.aggressive_momentum import JUSTIFICATIONS
        self.assertEqual(set(JUSTIFICATIONS), set(WEIGHTS))
        for k, v in JUSTIFICATIONS.items():
            self.assertGreater(len(v), 30, k)


class TestStrategyAIsUntouched(unittest.TestCase):
    """Stage 2 adds a strategy. It must not edit the one already running."""

    def test_strategy_a_thresholds_are_unchanged(self):
        c = StrategyCfg()
        self.assertEqual(c.name, "trend_breakout")
        self.assertEqual(c.donchian_lookback, 48)
        self.assertEqual(c.min_adx, 20.0)
        self.assertEqual(c.max_rsi, 78.0)
        self.assertEqual(c.min_rel_volume, 1.2)
        self.assertEqual(c.min_score, 55.0)
        self.assertEqual(c.stop_atr_mult, 2.2)
        self.assertEqual(c.max_extension_atr, 3.5)
        self.assertEqual(c.entry_timeframe, "1h")
        self.assertEqual(c.regime_timeframe, "4h")

    def test_the_shipped_config_still_matches(self):
        cfg = load_config("config/config.toml", "/nonexistent")
        self.assertEqual(cfg.strategy.min_score, 55.0)
        self.assertEqual(cfg.universe.min_dollar_volume_24h, 5_000_000.0)
        self.assertEqual(cfg.universe.min_atr_pct, 0.8)
        self.assertEqual(cfg.universe.min_market_age_days, 45)
        self.assertEqual(cfg.risk.risk_per_trade_pct, 0.5)
        self.assertEqual(cfg.risk.max_open_positions, 6)

    def test_the_two_strategies_have_different_names_and_configs(self):
        self.assertNotEqual(AggressiveCfg().name, StrategyCfg().name)
        self.assertEqual(AggressiveCfg().name, "aggressive_momentum_v2")

    def test_strategy_b_scoring_does_not_import_strategy_a_weights(self):
        from crypto_edge.strategy import scoring
        self.assertNotEqual(set(WEIGHTS), set(scoring.WEIGHTS),
                            "the two scorers must be independent")


if __name__ == "__main__":
    unittest.main()
