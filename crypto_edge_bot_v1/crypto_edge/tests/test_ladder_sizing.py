"""The capital ladder, confidence buckets, and the risk cap that overrules them.

WHAT THE LADDER IS
------------------
Each slot may use a percentage of the cash still FREE when it opens: 50% of the
balance, then 75% of what remains, then all of it. At full confidence that
deploys the whole account across three positions.

WHAT THE RISK CAP IS FOR
------------------------
A percentage of cash says nothing about how much can be LOST -- the same
allocation is a 0.5% or a 5% account risk depending on where the stop sits. So
the ladder only ever proposes a ceiling, and the smallest of four independent
limits wins. Which one bound is recorded on every trade, because "is the risk
cap doing anything, or is the ladder always the binding one?" is a question the
research database has to be able to answer in one query.
"""
import unittest

import helpers  # noqa: F401  -- silences the engine's log handlers
from crypto_edge.config import AggressiveCfg
from crypto_edge.portfolio import ladder as L
from crypto_edge.strategy import confidence as C

CFG = AggressiveCfg()
BUCKETS = CFG.confidence_buckets


def size(free_cash=10_000.0, equity=10_000.0, n_open=0, confidence=84.7,
         stop_pct=1.5, direction=1, exposure=0.0, buffer=10_000.0, **over):
    entry = 100.0
    stop = entry * (1 - stop_pct / 100 * direction)
    kw = dict(free_cash=free_cash, equity=equity, n_open=n_open,
              confidence=confidence,
              multiplier=C.multiplier(confidence, BUCKETS),
              entry_price=entry, stop_price=stop, direction=direction,
              exposure=exposure, daily_buffer_remaining=buffer,
              ceilings_pct=CFG.ladder_ceilings_pct,
              max_loss_pct=CFG.max_loss_pct,
              daily_buffer_fraction=CFG.daily_buffer_fraction,
              leverage=CFG.leverage, max_exposure_pct=CFG.max_exposure_pct,
              max_positions=CFG.max_open_positions)
    kw.update(over)
    return L.plan(**kw)


def walk(confidence, cash=10_000.0, stop_pct=1.5, buffer=10_000.0):
    """Fill all three slots at one confidence, returning each notional."""
    free, out = cash, []
    for n in range(CFG.max_open_positions):
        p = size(free_cash=free, n_open=n, confidence=confidence,
                 stop_pct=stop_pct, exposure=cash - free, buffer=buffer)
        out.append(p)
        free -= p.notional
    return out, free


class TestLadderArithmetic(unittest.TestCase):
    def test_the_ceilings_are_fifty_seventyfive_and_one_hundred(self):
        self.assertEqual(CFG.ladder_ceilings_pct, [50.0, 75.0, 100.0])
        self.assertEqual(CFG.max_open_positions, 3)

    def test_full_confidence_deploys_the_whole_balance(self):
        plans, left = walk(97.0)
        self.assertAlmostEqual(plans[0].notional, 5_000.0, places=6)
        self.assertAlmostEqual(plans[1].notional, 3_750.0, places=6)
        self.assertAlmostEqual(plans[2].notional, 1_250.0, places=6)
        self.assertAlmostEqual(left, 0.0, places=6)

    def test_the_worked_example_from_the_design(self):
        """84.7 -> the 80-89 bucket -> 80% of a 50% ceiling = 40% of cash."""
        p = size(confidence=84.7)
        self.assertEqual(p.slot, 1)
        self.assertAlmostEqual(p.ceiling_cash, 5_000.0)
        self.assertAlmostEqual(p.multiplier, 0.80)
        self.assertAlmostEqual(p.notional, 4_000.0, places=6)

    def test_eighty_percent_confidence_across_three_slots(self):
        plans, left = walk(84.7)
        got = [round(p.notional, 2) for p in plans]
        self.assertEqual(got, [4_000.0, 3_600.0, 1_920.0])
        self.assertAlmostEqual(left, 480.0, places=2)

    def test_a_low_confidence_run_leaves_cash_idle(self):
        plans, left = walk(65.0)
        got = [round(p.notional, 2) for p in plans]
        self.assertEqual(got, [2_000.0, 2_400.0, 2_240.0])
        self.assertAlmostEqual(left, 3_360.0, places=2)

    def test_a_later_slot_can_exceed_an_earlier_one(self):
        """A property of the design, not a bug: later ceilings are computed on
        a base the small early slots barely touched."""
        plans, _ = walk(65.0)
        self.assertGreater(plans[2].notional, plans[0].notional)

    def test_the_ceiling_never_exceeds_the_cash_on_hand(self):
        for n, pct in enumerate(CFG.ladder_ceilings_pct):
            slot, ceiling = L.ceiling_for_slot(1_000.0, n, CFG.ladder_ceilings_pct)
            self.assertEqual(slot, n + 1)
            self.assertLessEqual(ceiling, 1_000.0)

    def test_zero_cash_offers_nothing(self):
        self.assertEqual(L.ceiling_for_slot(0.0, 0, CFG.ladder_ceilings_pct),
                         (1, 0.0))

    def test_a_fourth_position_has_no_slot(self):
        slot, ceiling = L.ceiling_for_slot(10_000.0, 3, CFG.ladder_ceilings_pct)
        self.assertEqual((slot, ceiling), (0, 0.0))

    def test_the_three_position_maximum_is_enforced(self):
        p = size(n_open=3)
        self.assertFalse(p.tradable)
        self.assertIn("slots are in use", p.reason)

    def test_slot_numbering_is_one_based_and_follows_the_open_count(self):
        for n in range(3):
            self.assertEqual(size(n_open=n).slot, n + 1)


class TestConfidenceBuckets(unittest.TestCase):
    """Every boundary, because an off-by-one here mis-sizes every trade."""

    def test_the_mapping_is_exactly_as_specified(self):
        for confidence, want in ((60.0, 0.40), (65.0, 0.40), (69.9, 0.40),
                                 (70.0, 0.60), (79.9, 0.60),
                                 (80.0, 0.80), (89.9, 0.80),
                                 (90.0, 0.90), (94.9, 0.90),
                                 (95.0, 1.00), (100.0, 1.00)):
            self.assertAlmostEqual(C.multiplier(confidence, BUCKETS), want,
                                   msg=f"confidence {confidence}")

    def test_below_sixty_is_no_trade_not_a_small_trade(self):
        for confidence in (0.0, 42.0, 59.0, 59.99):
            self.assertEqual(C.multiplier(confidence, BUCKETS), 0.0)
            self.assertEqual(C.bucket_for(confidence, BUCKETS)[0],
                             C.NO_TRADE_BUCKET)
            self.assertFalse(C.is_tradable(confidence, 60.0, BUCKETS))

    def test_exactly_sixty_is_tradable(self):
        self.assertTrue(C.is_tradable(60.0, 60.0, BUCKETS))
        self.assertAlmostEqual(C.multiplier(60.0, BUCKETS), 0.40)

    def test_bucket_labels_are_groupable(self):
        self.assertEqual(C.bucket_for(84.7, BUCKETS)[0], "80-89")
        self.assertEqual(C.bucket_for(65.0, BUCKETS)[0], "60-69")
        self.assertEqual(C.bucket_for(99.0, BUCKETS)[0], "95+")

    def test_a_sub_floor_confidence_produces_no_size(self):
        p = size(confidence=55.0)
        self.assertFalse(p.tradable)
        self.assertIn("below the trading floor", p.reason)

    def test_confidence_is_the_identity_of_setup_score_in_phase_one(self):
        for score in (0.0, 34.0, 50.0, 84.7, 100.0):
            self.assertEqual(C.to_confidence(score, identity=True), score)

    def test_a_non_identity_transform_refuses_rather_than_inventing_one(self):
        """No calibration has been fitted; pretending otherwise would be worse."""
        with self.assertRaises(NotImplementedError):
            C.to_confidence(84.7, identity=False)

    def test_the_multipliers_are_configurable(self):
        custom = [{"min": 60.0, "mult": 0.10}, {"min": 90.0, "mult": 1.00}]
        self.assertAlmostEqual(C.multiplier(75.0, custom), 0.10)
        self.assertAlmostEqual(C.multiplier(95.0, custom), 1.00)

    def test_an_empty_bucket_list_trades_nothing(self):
        self.assertEqual(C.multiplier(99.0, []), 0.0)


class TestTheRiskCapOverrulesTheLadder(unittest.TestCase):
    def test_a_tight_stop_leaves_the_ladder_binding(self):
        p = size(stop_pct=1.5)
        self.assertEqual(p.binding_constraint, L.LADDER)
        self.assertAlmostEqual(p.notional, 4_000.0)

    def test_a_wide_stop_makes_the_risk_cap_bind(self):
        p = size(stop_pct=4.0)
        self.assertEqual(p.binding_constraint, L.RISK_CAP)
        self.assertAlmostEqual(p.notional, 2_500.0, places=6)

    def test_the_loss_at_the_stop_is_pinned_at_the_cap(self):
        """However wide the stop, the account risk lands on the ceiling."""
        for stop_pct in (2.5, 4.0, 6.0, 10.0):
            p = size(stop_pct=stop_pct)
            loss = p.notional * stop_pct / 100.0
            self.assertLessEqual(loss, p.max_loss_cash + 1e-6, f"stop {stop_pct}%")
            self.assertAlmostEqual(loss, 100.0, places=4, msg=f"stop {stop_pct}%")

    def test_the_cap_is_one_percent_of_equity_by_default(self):
        self.assertAlmostEqual(size().max_loss_cash, 100.0)
        self.assertAlmostEqual(size(equity=25_000.0).max_loss_cash, 250.0)

    def test_the_risk_cap_never_loses_to_the_ladder(self):
        p = size(stop_pct=8.0)
        self.assertLess(p.notional, p.target_notional,
                        "the ladder wanted more and did not get it")

    def test_exposure_room_can_bind(self):
        p = size(exposure=9_000.0, stop_pct=0.5)
        self.assertEqual(p.binding_constraint, L.EXPOSURE)
        self.assertAlmostEqual(p.notional, 1_000.0, places=6)

    def test_at_one_times_the_ladder_can_never_outrun_the_cash(self):
        """Cash is a backstop, not a binding limit, and that is structural.

        A ceiling is at most 100% of free cash and a multiplier at most 1.0, so
        at 1x leverage the ladder's ask is always <= the balance. The limit is
        kept because it stops being redundant the moment leverage is raised or
        the ceilings are edited -- but it should never be what binds today, and
        a test that expected it to bind was testing an impossible state.
        """
        for cash in (250.0, 500.0, 3_000.0, 10_000.0):
            for n in range(3):
                for confidence in (60.0, 84.7, 97.0):
                    p = size(free_cash=cash, n_open=n, confidence=confidence,
                             stop_pct=0.2)
                    self.assertLessEqual(p.target_notional, p.affordable + 1e-9)
                    self.assertNotEqual(p.binding_constraint, L.CASH)

    def test_cash_binds_once_leverage_decouples_it(self):
        """With leverage the two stop moving together, and the guard earns its keep."""
        p = size(free_cash=1_000.0, equity=100_000.0, confidence=97.0,
                 stop_pct=0.05, leverage=2.0, n_open=2)
        self.assertGreater(p.affordable, 0.0)
        self.assertLessEqual(p.notional, p.affordable + 1e-9)

    def test_a_ladder_cash_tie_reports_the_ladder(self):
        """Slot 3's ceiling IS the remaining cash; that is design, not scarcity."""
        p = size(n_open=2, free_cash=1_250.0, confidence=97.0, stop_pct=0.2)
        self.assertEqual(p.binding_constraint, L.LADDER)

    def test_every_limit_is_recorded_for_research(self):
        p = size(stop_pct=4.0)
        for key in (L.LADDER, L.RISK_CAP, L.EXPOSURE, L.CASH):
            self.assertIn(key, p.detail)
        self.assertIn("stop_pct", p.detail)

    def test_a_size_below_the_venue_minimum_is_refused_not_rounded_up(self):
        p = size(free_cash=20.0, stop_pct=0.5, min_notional=50.0)
        self.assertFalse(p.tradable)
        self.assertEqual(p.binding_constraint, L.MIN_NOTIONAL)


class TestTheDailyBufferTaper(unittest.TestCase):
    """The cap tightens as the day's losses accumulate.

    A flat 1% of equity is the same size on a flat day and on a day already
    2.5% into a 3% limit. Scaling to what is LEFT means the last trade before a
    daily halt cannot be the one that causes it.
    """

    def test_a_full_buffer_leaves_the_equity_term_binding(self):
        self.assertAlmostEqual(size(buffer=10_000.0).max_loss_cash, 100.0)

    def test_a_shrinking_buffer_shrinks_the_cap(self):
        seen = [size(buffer=b).max_loss_cash for b in (1000, 300, 150, 60, 20)]
        self.assertEqual(seen, sorted(seen, reverse=True))
        self.assertAlmostEqual(seen[-1], 12.0, places=6)   # 60% of 20

    def test_the_taper_shrinks_the_position_too(self):
        big = size(stop_pct=2.5, buffer=10_000.0).notional
        small = size(stop_pct=2.5, buffer=60.0).notional
        self.assertLess(small, big / 2)

    def test_an_exhausted_buffer_permits_nothing(self):
        p = size(buffer=0.0)
        self.assertFalse(p.tradable)
        self.assertEqual(p.max_loss_cash, 0.0)

    def test_the_tighter_of_the_two_terms_always_wins(self):
        for equity, buffer in ((10_000, 10_000), (10_000, 100), (500, 10_000)):
            got = L.loss_ceiling(equity, buffer, max_loss_pct=1.0,
                                 daily_buffer_fraction=0.6)
            self.assertAlmostEqual(got, min(equity * 0.01, buffer * 0.6))


class TestLongShortSizingSymmetry(unittest.TestCase):
    def test_a_mirrored_short_is_sized_identically(self):
        long_p = size(direction=1, stop_pct=2.0)
        entry = 100.0
        short_p = size(direction=-1, stop_price=entry * 1.02, stop_pct=2.0)
        self.assertAlmostEqual(long_p.notional, short_p.notional, places=6)
        self.assertEqual(long_p.binding_constraint, short_p.binding_constraint)

    def test_the_risk_cap_binds_a_short_the_same_way(self):
        entry = 100.0
        p = size(direction=-1, stop_price=entry * 1.04)
        self.assertEqual(p.binding_constraint, L.RISK_CAP)
        self.assertAlmostEqual(p.notional * 0.04, 100.0, places=4)

    def test_a_stop_below_entry_is_refused_for_a_short(self):
        p = size(direction=-1, stop_price=98.0)
        self.assertFalse(p.tradable)
        self.assertIn("wrong side", p.reason)

    def test_a_stop_above_entry_is_refused_for_a_long(self):
        p = size(direction=1, stop_price=102.0)
        self.assertFalse(p.tradable)
        self.assertIn("wrong side", p.reason)


class TestLeverageIsSeparateFromConfidence(unittest.TestCase):
    def test_phase_one_is_one_times(self):
        self.assertEqual(CFG.leverage, 1.0)

    def test_leverage_scales_notional_without_touching_confidence(self):
        base = size(stop_pct=0.5, leverage=1.0).target_notional
        levered = size(stop_pct=0.5, leverage=3.0).target_notional
        self.assertAlmostEqual(levered, base * 3.0, places=6)
        self.assertAlmostEqual(size(leverage=3.0).multiplier,
                               size(leverage=1.0).multiplier)

    def test_the_risk_cap_still_binds_under_leverage(self):
        """Leverage must not become a way around the loss ceiling."""
        p = size(stop_pct=4.0, leverage=5.0)
        self.assertEqual(p.binding_constraint, L.RISK_CAP)
        self.assertAlmostEqual(p.notional * 0.04, 100.0, places=4)


if __name__ == "__main__":
    unittest.main()
