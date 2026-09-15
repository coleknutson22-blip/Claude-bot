"""Offline exit-policy replay: CONTROL vs a fixed +2% target, on one signal.

WHAT THESE TESTS ARE REALLY DEFENDING
-------------------------------------
The simulator exists to answer a question the operator already has an opinion
about, which makes a plausible-but-wrong answer far more dangerous than a
crash. Three failure modes would each produce one:

  1. A DIFFERENT EXIT ORDER. The live engine resolves the stop intrabar first,
     then `check_exit` on the close, and only then ratchets the stop -- so a
     new stop never applies on the bar that set it. Any other order changes
     which trades survive, and the comparison would measure the reordering.

  2. TP2 DIFFERING IN MORE THAN THE TARGET. The whole design is one variable.
     If TP2 also trailed differently, the result would be uninterpretable.

  3. A SIMULATOR THAT CANNOT REPRODUCE REALITY. Reconciliation against the
     recorded ledger is the only thing standing between "a result" and "a
     number the code happened to produce".

All three are tested directly and mutation-tested.
"""
import unittest

import helpers  # noqa: F401  -- silences the engine's log handlers
from crypto_edge.config import Config
from crypto_edge.research import policy_report as rep
from crypto_edge.research import policy_sim as sim
from helpers import temp_repo

B = "aggressive_momentum_v2"
T0, BAR = 1_700_000_000_000, 300_000


def tape(rows, atr=1.0, struct=1.0, regime="bull", start=1):
    """rows = [(open, high, low, close), ...] as consecutive 5m bars."""
    return [{"open_ms": T0 + (start + i) * BAR,
             "elapsed_min": (start + i) * 5.0,
             "open": o, "high": h, "low": l, "close": c,
             "atr": atr, "atr_tf": "15m", "ema_struct_15m": struct,
             "btc_regime": regime} for i, (o, h, l, c) in enumerate(rows)]


class SimCase(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.a = self.cfg.aggressive
        self.broker = sim.broker_from(self.cfg.execution)

    def run_sim(self, rows, policy=sim.CONTROL, side="long", entry=100.0,
                stop=None, **kw):
        stop = stop if stop is not None else (98.2 if side == "long" else 101.8)
        return sim.replay(tape=rows, cfg=self.a, policy=policy, symbol="X/USD",
                          side=side, entry_ref=entry, entry_fill=entry,
                          entry_ms=T0, initial_stop=stop, qty=1.0,
                          broker=self.broker, meta={"price_precision": 6}, **kw)


# ============================================ CONTROL replays each exit
class TestControlReplaysEveryExit(SimCase):
    def test_the_initial_hard_stop(self):
        r = self.run_sim(tape([(100.0, 100.2, 98.0, 98.1)]))
        self.assertEqual(r.exit_reason, "stop")
        self.assertEqual(r.exit_bar_index, 0)

    def test_a_gap_through_the_stop_fills_at_the_open(self):
        # The classic backtest lie is pretending a gap fills at the stop.
        r = self.run_sim(tape([(97.0, 97.5, 96.0, 96.5)]))
        self.assertEqual(r.exit_reason, "stop_gap")
        self.assertLess(r.exit_fill_price, 98.2)
        self.assertLess(r.exit_fill_price, 97.0)   # worse than the open

    def test_the_2r_target_fires_on_the_close(self):
        r = self.run_sim(tape([(100.0, 103.7, 100.0, 103.7)]))
        self.assertEqual(r.exit_reason, "target")

    def test_a_wick_through_2r_that_closes_back_does_not_exit(self):
        # Live evaluates the target on the CLOSE. A touch model would exit here
        # and would overstate every target's hit rate.
        r = self.run_sim(tape([(100.0, 104.0, 99.9, 100.1)]))
        self.assertNotEqual(r.exit_reason, "target")

    def test_momentum_invalidation(self):
        r = self.run_sim(tape([(100.0, 100.5, 99.9, 100.2)], struct=-1.0))
        self.assertEqual(r.exit_reason, "momentum_invalidation")

    def test_the_hostile_regime_exit(self):
        r = self.run_sim(tape([(100.0, 100.5, 99.9, 100.2)], regime="bear"))
        self.assertEqual(r.exit_reason, "hostile_regime")

    def test_a_short_is_closed_by_a_bull_regime(self):
        r = self.run_sim(tape([(100.0, 100.1, 99.5, 99.8)], regime="bull",
                              struct=-1.0), side="short")
        self.assertEqual(r.exit_reason, "hostile_regime")

    def test_the_eight_hour_time_stop(self):
        # Held above 0.5R so the early stop never fires, and each bar OPENS
        # above the ratcheted stop -- otherwise the trail gaps it out first,
        # which is correct behaviour but a different test.
        n = int(self.a.time_stop_hours * 12) + 2
        rows = tape([(101.5, 101.9, 101.4, 101.8)] * n, atr=0.0)
        r = self.run_sim(rows)
        self.assertEqual(r.exit_reason, "time_stop")
        self.assertGreaterEqual(r.hold_minutes, self.a.time_stop_hours * 60)

    def test_a_ratcheted_stop_can_itself_be_gapped_through(self):
        # Once breakeven is set, a bar that OPENS below it fills at the open,
        # not at the stop. The trail does not exempt a trade from gap risk.
        rows = tape([(101.5, 102.2, 101.4, 101.9),      # sets breakeven
                     (99.0, 99.5, 98.5, 98.7)], atr=0.0)
        r = self.run_sim(rows)
        self.assertEqual(r.exit_reason, "stop_gap")
        self.assertLess(r.exit_fill_price, 99.0)

    def test_the_four_hour_no_progress_stop(self):
        n = int(self.a.time_stop_early_hours * 12) + 2
        # flat: never reaches time_stop_min_r, never stops out
        r = self.run_sim(tape([(100.0, 100.2, 99.9, 100.05)] * n))
        self.assertEqual(r.exit_reason, "time_stop_no_progress")
        self.assertGreaterEqual(r.hold_minutes,
                                self.a.time_stop_early_hours * 60)

    def test_a_trade_making_progress_survives_the_four_hour_stop(self):
        n = int(self.a.time_stop_early_hours * 12) + 2
        rows = tape([(100.0, 101.9, 99.9, 101.8)] * n)   # > 0.5R, under 2R
        r = self.run_sim(rows)
        self.assertNotEqual(r.exit_reason, "time_stop_no_progress")

    def test_the_forced_short_collateral_close(self):
        # A short has no floor: price doubling eats the collateral at 1x.
        r = self.run_sim(tape([(100.0, 185.0, 100.0, 184.0)] * 2,
                              struct=-1.0, regime="bear"),
                         side="short", stop=500.0)
        self.assertEqual(r.exit_reason, "forced_close_collateral")

    def test_the_breakeven_ratchet_converts_a_loser_into_a_scratch(self):
        # Runs to 1.2R, then falls back to where the INITIAL stop was. Live
        # exits at breakeven; a frozen-stop model would score a full -1R loss.
        rows = tape([(100.0, 102.2, 99.9, 102.2), (102.2, 102.3, 98.0, 98.1)])
        r = self.run_sim(rows)
        self.assertEqual(r.exit_reason, "stop")
        self.assertGreater(r.exit_fill_price, 100.0,
                           "the stop never moved to breakeven")

    def test_the_chandelier_trail_takes_over_above_its_threshold(self):
        rows = tape([(100.0, 103.3, 99.9, 103.2), (103.2, 103.3, 99.0, 99.1)],
                    atr=0.5)
        r = self.run_sim(rows)
        self.assertEqual(r.exit_reason, "stop")
        # 2.5 ATR behind the 103.3 extreme is well above breakeven
        self.assertGreater(r.exit_fill_price, 101.0)

    def test_a_path_with_no_exit_is_reported_not_guessed(self):
        r = self.run_sim(tape([(100.0, 100.2, 99.9, 100.05)] * 2))
        self.assertFalse(r.ok)
        self.assertEqual(r.unreplayable, sim.NO_EXIT_IN_WINDOW)


# ====================================================== exit priority
class TestCompetingExitsFollowTheLiveOrder(SimCase):
    def test_stop_and_target_on_one_bar_resolve_to_the_stop(self):
        # Live resolves the stop intrabar BEFORE looking at the close.
        r = self.run_sim(tape([(100.0, 104.0, 98.0, 103.8)]))
        self.assertEqual(r.exit_reason, "stop")

    def test_a_new_stop_never_applies_on_the_bar_that_set_it(self):
        # The ratchet runs AFTER the exit checks, so breakeven set on bar 1
        # can only fire from bar 2. Reversing that would stop trades out on
        # the very bar that moved their stop.
        # The ratchet reads the CLOSE, so the close must clear 1R (101.80).
        rows = tape([(100.0, 102.0, 99.99, 101.9),      # sets breakeven @100.18
                     (101.9, 102.0, 99.0, 99.1)],       # now it fires
                    atr=0.0)
        r = self.run_sim(rows)
        self.assertEqual(r.exit_bar_index, 1)
        self.assertEqual(r.exit_reason, "stop")
        self.assertGreater(r.exit_fill_price, 100.0)

    def test_target_beats_momentum_invalidation(self):
        r = self.run_sim(tape([(100.0, 103.7, 100.0, 103.7)], struct=-1.0))
        self.assertEqual(r.exit_reason, "target")

    def test_target_beats_the_hostile_regime(self):
        r = self.run_sim(tape([(100.0, 103.7, 100.0, 103.7)], regime="bear"))
        self.assertEqual(r.exit_reason, "target")

    def test_target_beats_both_time_stops(self):
        n = int(self.a.time_stop_hours * 12) + 2
        rows = tape([(100.0, 100.1, 99.9, 100.05)] * (n - 1)
                    + [(100.0, 103.7, 100.0, 103.7)])
        # the early time stop fires long before the target bar is reached
        r = self.run_sim(rows)
        self.assertEqual(r.exit_reason, "time_stop_no_progress")

    def test_momentum_beats_the_regime_exit(self):
        r = self.run_sim(tape([(100.0, 100.5, 99.9, 100.2)], struct=-1.0,
                              regime="bear"))
        self.assertEqual(r.exit_reason, "momentum_invalidation")

    def test_the_regime_exit_beats_the_time_stops(self):
        n = int(self.a.time_stop_hours * 12) + 2
        r = self.run_sim(tape([(100.0, 100.5, 99.9, 100.2)] * n,
                              regime="bear"))
        self.assertEqual(r.exit_reason, "hostile_regime")

    def test_the_forced_short_close_outranks_everything(self):
        r = self.run_sim(tape([(100.0, 185.0, 100.0, 184.0)],
                              struct=1.0, regime="bear"),
                         side="short", stop=500.0)
        self.assertEqual(r.exit_reason, "forced_close_collateral")

    def test_the_chandelier_and_the_target_compete_correctly(self):
        # Trail set on bar 1; bar 2 closes above 2R but its LOW takes the
        # trail out first, and the stop is resolved first.
        rows = tape([(100.0, 103.3, 99.9, 103.2),
                     (103.2, 103.8, 101.0, 103.7)], atr=0.5)
        r = self.run_sim(rows)
        self.assertEqual(r.exit_reason, "stop")

    def test_the_documented_order_matches_the_live_source(self):
        # A rename in the live module must not silently pass here.
        import inspect

        from crypto_edge.portfolio import aggressive_exits
        src = inspect.getsource(aggressive_exits.check_exit)
        order = [src.index(tok) for tok in
                 ("FORCED_SHORT", "return TARGET", "return MOMENTUM",
                  "return REGIME", "return TIME\n", "return TIME_EARLY")]
        self.assertEqual(order, sorted(order),
                         "check_exit's branch order changed; the simulator "
                         "reproduces it by calling it, but this test pins the "
                         "order the tests above assume")


# ====================================================== the TP2 policy
class TestTP2(SimCase):
    def test_a_long_target_is_entry_times_1_02(self):
        self.assertAlmostEqual(sim.fixed_target_price(100.0, 1), 102.0)

    def test_a_short_target_is_entry_times_0_98(self):
        self.assertAlmostEqual(sim.fixed_target_price(100.0, -1), 98.0)

    def test_tp2_fires_where_control_would_not(self):
        rows = tape([(100.0, 102.1, 99.9, 102.05)])
        self.assertEqual(self.run_sim(rows, sim.TP2).exit_reason, "target")
        self.assertNotEqual(self.run_sim(rows, sim.CONTROL).exit_reason,
                            "target")

    def test_tp2_is_close_based_exactly_like_control(self):
        rows = tape([(100.0, 102.5, 99.9, 100.1)])     # wick only
        self.assertNotEqual(self.run_sim(rows, sim.TP2).exit_reason, "target")

    def test_the_short_side_mirrors(self):
        rows = tape([(100.0, 100.1, 97.9, 97.95)], struct=-1.0, regime="bear")
        r = self.run_sim(rows, sim.TP2, side="short")
        self.assertEqual(r.exit_reason, "target")
        self.assertGreater(r.net_return_pct, 0)

    def test_tp2_changes_only_the_target(self):
        # Every other configured number must be untouched, or the experiment
        # is no longer one variable.
        a2 = sim.tp2_cfg(self.a, 1.8, 2.0)
        for fieldname in ("stop_atr_mult", "breakeven_at_r",
                          "breakeven_offset_r", "trail_start_r",
                          "trail_atr_mult", "time_stop_hours",
                          "time_stop_early_hours", "time_stop_min_r",
                          "min_ema_struct_15m", "exit_on_hostile_regime",
                          "short_borrow_bps_per_day",
                          "short_force_close_at_loss_pct", "leverage",
                          "max_open_positions", "min_setup_score"):
            self.assertEqual(getattr(a2, fieldname), getattr(self.a, fieldname),
                             fieldname)
        self.assertNotEqual(a2.target_r, self.a.target_r)

    def test_the_r_equivalent_lands_on_exactly_two_percent(self):
        for stop_pct in (0.45, 1.0, 1.8, 3.6):
            a2 = sim.tp2_cfg(self.a, stop_pct, 2.0)
            self.assertAlmostEqual(a2.target_r * stop_pct, 2.0, places=9)

    def test_the_live_config_object_is_never_mutated(self):
        before = self.a.target_r
        sim.tp2_cfg(self.a, 1.8, 2.0)
        self.assertEqual(self.a.target_r, before)

    def test_below_the_crossover_control_is_the_looser_rule(self):
        # atr 0.3% -> stop 0.54% -> 2R needs only +1.08%
        rows = tape([(100.0, 101.2, 99.9, 101.15)])
        r_c = self.run_sim(rows, sim.CONTROL, stop=99.46)
        r_t = self.run_sim(rows, sim.TP2, stop=99.46)
        self.assertEqual(r_c.exit_reason, "target")
        self.assertNotEqual(r_t.exit_reason, "target")


# ============================================================ cost model
class TestCosts(SimCase):
    def test_both_policies_share_one_broker(self):
        rows = tape([(100.0, 103.7, 100.0, 103.7)])
        c = self.run_sim(rows, sim.CONTROL)
        t = self.run_sim(rows, sim.TP2)
        self.assertAlmostEqual(c.fees, t.fees, places=9)

    def test_fees_are_charged_on_the_exit(self):
        r = self.run_sim(tape([(100.0, 103.7, 100.0, 103.7)]))
        self.assertGreater(r.fees, 0)
        self.assertLess(r.net_return_pct, r.gross_return_pct)

    def test_slippage_is_a_cost_on_both_sides(self):
        long_r = self.run_sim(tape([(100.0, 103.7, 100.0, 103.7)]))
        short_r = self.run_sim(tape([(100.0, 100.0, 96.3, 96.3)], struct=-1.0,
                                    regime="bear"), side="short", stop=101.8)
        self.assertGreater(long_r.slippage_cost, 0)
        self.assertGreater(short_r.slippage_cost, 0)

    def test_a_stop_uses_the_worse_stop_slippage(self):
        stop_r = self.run_sim(tape([(100.0, 100.2, 98.0, 98.1)]))
        tgt_r = self.run_sim(tape([(100.0, 103.7, 100.0, 103.7)]))
        self.assertGreater(stop_r.slippage_cost / 1.0, 0)
        self.assertGreater(tgt_r.slippage_cost, 0)

    def test_short_financing_accrues_and_longs_pay_none(self):
        n = int(self.a.time_stop_hours * 12) + 2
        short_r = self.run_sim(tape([(100.0, 100.2, 99.8, 100.0)] * n,
                                    struct=-1.0),
                               side="short", stop=101.8)
        long_r = self.run_sim(tape([(100.0, 100.2, 99.8, 100.05)] * n))
        self.assertGreater(short_r.financing, 0)
        self.assertEqual(long_r.financing, 0.0)

    def test_financing_scales_with_hold_time(self):
        # Both legs must exit for the SAME reason, or this compares two
        # different trades. Progress is held above 0.5R so only the 8h stop
        # can end them, and the shorter leg is ended by its tape running out.
        # A short is invalidated by a BULLISH 15m stack and by a BULL regime,
        # so both are set neutral -- otherwise this measures a momentum exit.
        n = int(self.a.time_stop_hours * 12) + 2
        rows = tape([(98.5, 98.6, 98.1, 98.2)] * n, struct=0.0,
                    regime="neutral")

        def run(entry_ms):
            return sim.replay(tape=rows, cfg=self.a, policy=sim.CONTROL,
                              symbol="X/USD", side="short", entry_ref=100.0,
                              entry_fill=100.0, entry_ms=entry_ms,
                              initial_stop=101.8, qty=1.0, broker=self.broker,
                              meta={"price_precision": 6})

        full = run(T0)
        self.assertEqual(full.exit_reason, "time_stop")
        self.assertGreater(full.financing, 0)
        half = run(T0 + (full.exit_ms - T0) // 2)     # half the hold
        self.assertLess(half.financing, full.financing)

    def test_venue_precision_is_applied_to_the_fill(self):
        r = sim.replay(tape=tape([(100.0, 103.7, 100.0, 103.7)]), cfg=self.a,
                       policy=sim.CONTROL, symbol="X/USD", side="long",
                       entry_ref=100.0, entry_fill=100.0, entry_ms=T0,
                       initial_stop=98.2, qty=1.0, broker=self.broker,
                       meta={"price_precision": 2})
        self.assertAlmostEqual(r.exit_fill_price,
                               round(r.exit_fill_price, 2), places=9)

    def test_gross_and_net_are_both_reported(self):
        r = self.run_sim(tape([(100.0, 103.7, 100.0, 103.7)]))
        self.assertGreater(r.gross_return_pct, 0)
        self.assertLess(r.net_return_pct, r.gross_return_pct)

    def test_the_broker_must_come_from_the_live_execution_config(self):
        with self.assertRaises(ValueError):
            sim.replay(tape=tape([(100.0, 101.0, 99.0, 100.0)]), cfg=self.a,
                       policy=sim.CONTROL, symbol="X/USD", side="long",
                       entry_ref=100.0, entry_fill=100.0, entry_ms=T0,
                       initial_stop=98.2, broker=None)


# ======================================================== paired records
class PairCase(unittest.TestCase):
    def setUp(self):
        from crypto_edge.research.excursion_recorder import ExcursionRecorder
        from crypto_edge.research import excursion as X
        self.X = X
        self.repo, self.db = temp_repo()
        self.cfg = Config()
        self.rec = ExcursionRecorder(self.repo, B)

    def add(self, oid, rows, *, side="long", entry=100.0, stop=None,
            score=70.0, atr_pct=1.0, regime="bull", complete=True,
            with_tape=True, ts=None, struct=1.0):
        stop = stop if stop is not None else (98.2 if side == "long" else 101.8)
        ts = ts if ts is not None else T0
        self.repo.add_observation({
            "id": oid, "ts_ms": ts, "symbol": f"{oid}/USD", "candle_id": oid,
            "strategy": B, "strategy_version": "1", "decision": "ENTERED",
            "reject_reason": "", "side": side, "score": score, "rank": 1,
            "price": entry,
            "features": {"atr_pct": atr_pct, "btc_regime": regime}})
        p = self.X.build(observation_id=oid, symbol=f"{oid}/USD", strategy=B,
                         side=side, direction=-1 if side == "short" else 1,
                         ref_price=entry, stop_price=stop, signal_ms=ts,
                         meta={"price_precision": 6})
        p.status = self.X.COMPLETE if complete else self.X.OPEN
        self.repo.upsert_excursion(p)
        if with_tape:
            from crypto_edge.research.tape import TapeBar
            self.repo.add_tape_bars(oid, [
                TapeBar(open_ms=ts + (i + 1) * BAR, elapsed_min=(i + 1) * 5.0,
                        open=o, high=h, low=l, close=c, atr=1.0, atr_tf="15m",
                        ema_struct_15m=struct, btc_regime=regime)
                for i, (o, h, l, c) in enumerate(rows)])
        return p

    def comparison(self):
        return rep.PolicyComparison(self.repo, B, self.cfg)


class TestPairedComparison(PairCase):
    def test_a_pair_carries_both_policies_and_the_difference(self):
        self.add("a1", [(100.0, 102.6, 100.0, 102.5), (102.5, 102.6, 98.0, 98.1)])
        p = self.comparison().pairs()[0]
        self.assertTrue(p.ok)
        self.assertEqual(p.control.policy, sim.CONTROL)
        self.assertEqual(p.tp2.policy, sim.TP2)
        self.assertAlmostEqual(p.net_diff_pct,
                               p.tp2.net_return_pct - p.control.net_return_pct)

    def test_the_winner_is_the_higher_net_return(self):
        self.add("a1", [(100.0, 102.6, 100.0, 102.5), (102.5, 102.6, 98.0, 98.1)])
        p = self.comparison().pairs()[0]
        self.assertEqual(p.winner, sim.TP2)
        self.assertGreater(p.net_diff_pct, 0)

    def test_an_identical_outcome_is_a_tie(self):
        # Both policies stop out on the same bar: nothing to choose between.
        self.add("a1", [(100.0, 100.2, 98.0, 98.1)])
        p = self.comparison().pairs()[0]
        self.assertEqual(p.winner, "tie")
        self.assertAlmostEqual(p.net_diff_pct, 0.0)

    def test_the_paired_record_carries_the_signal_context(self):
        self.add("a1", [(100.0, 103.7, 100.0, 103.7)], score=84.7, atr_pct=1.0)
        d = self.comparison().pairs()[0].as_dict()
        for key in ("observation_id", "symbol", "side", "signal_ms",
                    "setup_score", "confidence", "atr_pct",
                    "stop_distance_pct", "CONTROL", "TP2", "DIFFERENCE"):
            self.assertIn(key, d)
        self.assertEqual(d["setup_score"], 84.7)
        self.assertEqual(d["DIFFERENCE"]["winner"], d["DIFFERENCE"]["winner"])

    def test_both_policies_see_identical_inputs(self):
        self.add("a1", [(100.0, 103.7, 100.0, 103.7)])
        p = self.comparison().pairs()[0]
        # same entry, same stop, same tape -- only the target differs
        self.assertEqual(p.control.exit_bar_index, p.tp2.exit_bar_index)
        self.assertEqual(p.control.exit_reason, p.tp2.exit_reason)

    def test_pairs_are_ordered_by_signal_time(self):
        self.add("a2", [(100.0, 103.7, 100.0, 103.7)], ts=T0 + 10 * BAR)
        self.add("a1", [(100.0, 103.7, 100.0, 103.7)], ts=T0)
        ids = [p.observation_id for p in self.comparison().pairs()]
        self.assertEqual(ids, ["a1", "a2"])


class TestExclusions(PairCase):
    def test_a_v7_path_without_tape_is_excluded_not_reconstructed(self):
        self.add("a1", [], with_tape=False)
        p = self.comparison().pairs()[0]
        self.assertFalse(p.ok)
        self.assertEqual(p.unreplayable, sim.UNREPLAYABLE_NO_TAPE)
        self.assertIsNone(p.control)

    def test_an_open_path_is_excluded(self):
        self.add("a1", [(100.0, 103.7, 100.0, 103.7)], complete=False)
        p = self.comparison().pairs()[0]
        self.assertEqual(p.unreplayable, sim.UNREPLAYABLE_OPEN)

    def test_exclusions_are_counted_by_reason(self):
        self.add("a1", [], with_tape=False)
        self.add("a2", [(100.0, 103.7, 100.0, 103.7)], complete=False)
        self.add("a3", [(100.0, 103.7, 100.0, 103.7)])
        s = self.comparison().summarise()
        self.assertEqual(s["total_paths"], 3)
        self.assertEqual(s["replayable"], 1)
        self.assertEqual(s["unreplayable"][sim.UNREPLAYABLE_NO_TAPE], 1)
        self.assertEqual(s["unreplayable"][sim.UNREPLAYABLE_OPEN], 1)

    def test_an_excluded_path_contributes_nothing_to_the_statistics(self):
        self.add("a1", [], with_tape=False)
        s = self.comparison().summarise()
        self.assertEqual(s["n"], 0)
        self.assertEqual(s["control_wins"], 0)
        self.assertEqual(s["tp2_wins"], 0)

    def test_a_tape_that_never_exits_is_excluded(self):
        self.add("a1", [(100.0, 100.2, 99.9, 100.05)] * 2)
        p = self.comparison().pairs()[0]
        self.assertEqual(p.unreplayable, sim.NO_EXIT_IN_WINDOW)

    def test_one_arm_failing_drops_the_WHOLE_pair(self):
        # A 0.54% stop: CONTROL's 2R is +1.08% and fires, TP2's +2% never does
        # on this short tape. Keeping CONTROL's result while dropping TP2's
        # would count a win for CONTROL precisely where TP2 had no chance --
        # a selection effect pointing at whichever policy has the nearer
        # target. Both arms are excluded together or neither is.
        self.add("a1", [(100.0, 101.2, 100.0, 101.15)], stop=99.46,
                 atr_pct=0.3)
        p = self.comparison().pairs()[0]
        self.assertEqual(p.control.exit_reason, "target")
        self.assertFalse(p.tp2.ok)
        self.assertFalse(p.ok, "a half-replayed pair leaked into the sample")
        s = self.comparison().summarise()
        self.assertEqual(s["replayable"], 0)
        self.assertEqual(s["control_wins"], 0)


class TestSampleSizeGuards(unittest.TestCase):
    def test_under_one_hundred_is_insufficient(self):
        for n in (0, 1, 47, 99):
            self.assertEqual(rep.sample_label(n), rep.INSUFFICIENT)

    def test_one_hundred_to_one_ninety_nine_is_provisional(self):
        for n in (100, 150, 199):
            self.assertEqual(rep.sample_label(n), rep.PROVISIONAL)

    def test_two_hundred_and_up_allows_a_comparison(self):
        for n in (200, 1000):
            self.assertEqual(rep.sample_label(n), rep.COMPARABLE)

    def test_the_thresholds_are_the_pre_registered_ones(self):
        self.assertEqual(rep.MIN_PROVISIONAL, 100)
        self.assertEqual(rep.MIN_COMPARABLE, 200)


class TestChronologicalHalves(PairCase):
    def make(self, n, winner_is_tp2, start_i):
        for i in range(n):
            rows = ([(100.0, 102.6, 100.0, 102.5), (102.5, 102.6, 98.0, 98.1)]
                    if winner_is_tp2 else [(100.0, 100.2, 98.0, 98.1)])
            self.add(f"p{start_i + i:03d}", rows, ts=T0 + (start_i + i) * 10 * BAR)

    def test_the_split_is_chronological(self):
        self.make(4, True, 0)
        self.make(4, False, 10)
        h = self.comparison().summarise()["halves"]
        self.assertEqual(h["first"]["n"], 4)
        self.assertEqual(h["second"]["n"], 4)

    def test_an_advantage_in_one_half_only_is_flagged(self):
        self.make(4, True, 0)       # TP2 wins early
        self.make(4, False, 10)     # ties later
        h = self.comparison().summarise()["halves"]
        self.assertEqual(h["first"]["sign"], "+")
        self.assertEqual(h["second"]["sign"], "0")
        self.assertFalse(h["same_sign"])

    def test_a_consistent_advantage_shows_the_same_sign_in_both(self):
        self.make(4, True, 0)
        self.make(4, True, 10)
        h = self.comparison().summarise()["halves"]
        self.assertEqual(h["first"]["sign"], h["second"]["sign"])
        self.assertTrue(h["same_sign"])


class TestBreakdowns(PairCase):
    def setUp(self):
        super().setUp()
        self.add("a1", [(100.0, 102.6, 100.0, 102.5), (102.5, 102.6, 98.0, 98.1)],
                 side="long", score=85.0, atr_pct=1.0, regime="bull")
        # closes at -3.7% = 2R for a 1.8% stop, so CONTROL's target fires
        self.add("a2", [(100.0, 100.1, 96.2, 96.3)], side="short", stop=101.8,
                 score=65.0, atr_pct=0.3, regime="bear", struct=1.0)

    def test_every_requested_breakdown_is_present(self):
        s = self.comparison().summarise()
        for key in ("by_side", "by_confidence_bucket", "by_score_bucket",
                    "by_atr_bucket", "by_btc_regime", "crossover"):
            self.assertIn(key, s)
            self.assertTrue(s[key], key)

    def test_the_crossover_groups_are_reported_separately(self):
        c = self.comparison()
        keys = list(c.summarise()["crossover"])
        self.assertEqual(len(keys), 2)
        self.assertTrue(any("<=" in k for k in keys))
        self.assertTrue(any(k.startswith("atr>") for k in keys))

    def test_the_crossover_is_derived_from_config(self):
        c = self.comparison()
        self.assertAlmostEqual(c.crossover_atr, 2.0 / (2.0 * 1.8), places=9)

    def test_long_and_short_are_separated(self):
        g = self.comparison().summarise()["by_side"]
        self.assertEqual(sorted(g), ["long", "short"])

    def test_each_group_reports_both_policies(self):
        for stats in self.comparison().summarise()["by_side"].values():
            for pol in ("CONTROL", "TP2"):
                self.assertIn(pol, stats)
                for metric in ("expectancy_pct", "win_rate_pct",
                               "profit_factor", "avg_winner_pct",
                               "avg_loser_pct", "hold_minutes_median"):
                    self.assertIn(metric, stats[pol])


# ======================================================== reconciliation
class TestReconciliation(PairCase):
    """The check that decides whether any TP2 number may be believed."""

    def book_trade(self, oid, *, exit_reason, exit_ms, exit_fill, net,
                   entry=100.0, qty=1.0, stop=98.2, side="long",
                   financing=0.0, fees=0.0, slippage=0.0):
        from crypto_edge.models import ClosedTrade
        self.repo.add_trade(ClosedTrade(
            id=f"t_{oid}", position_id=f"p_{oid}", symbol=f"{oid}/USD",
            strategy=B, strategy_version="1", qty=qty, entry_ref_price=entry,
            entry_fill_price=entry, entry_ms=T0, exit_ref_price=exit_fill,
            exit_fill_price=exit_fill, exit_ms=exit_ms,
            exit_reason=exit_reason, initial_stop=stop, final_stop=stop,
            gross_pnl=net + fees + slippage, fees=fees,
            slippage_cost=slippage, net_pnl=net, side=side,
            financing=financing, return_pct=net / entry * 100,
            account_return_pct=net / 10_000 * 100, mfe=0.0, mae=0.0,
            duration_s=(exit_ms - T0) / 1000, equity_after=10_000 + net,
            journal={"candle_id": oid}))

    def test_a_stop_exit_reconciles(self):
        rows = [(100.0, 100.2, 98.0, 98.1)]
        self.add("a1", rows)
        got = sim.replay(tape=self.repo.get_tape("a1"),
                         cfg=self.cfg.aggressive, policy=sim.CONTROL,
                         symbol="a1/USD", side="long", entry_ref=100.0,
                         entry_fill=100.0, entry_ms=T0, initial_stop=98.2,
                         qty=1.0, meta={"price_precision": 6},
                         broker=sim.broker_from(self.cfg.execution))
        self.book_trade("a1", exit_reason=got.exit_reason,
                        exit_ms=got.exit_ms, exit_fill=got.exit_fill_price,
                        net=got.net_pnl, fees=got.fees,
                        slippage=got.slippage_cost)
        r = self.comparison().reconcile()
        self.assertEqual(r.compared, 1)
        self.assertEqual(r.exact_reason, 1)
        self.assertEqual(r.fill_ok, 1)
        self.assertEqual(r.net_ok, 1)
        self.assertTrue(r.trustworthy)
        self.assertEqual(r.discrepancies, [])

    def test_a_wrong_exit_reason_is_reported_not_smoothed(self):
        self.add("a1", [(100.0, 100.2, 98.0, 98.1)])
        self.book_trade("a1", exit_reason="target", exit_ms=T0 + BAR * 2,
                        exit_fill=103.7, net=3.5)
        r = self.comparison().reconcile()
        self.assertEqual(r.exact_reason, 0)
        self.assertFalse(r.trustworthy)
        self.assertTrue(any(d.field == "exit_reason" for d in r.discrepancies))

    def test_a_stop_fill_gets_the_tight_tolerance(self):
        self.assertLess(rep.TOL["stop_fill_bps"], rep.TOL["quoted_fill_bps"])

    def test_a_fill_outside_tolerance_is_a_discrepancy(self):
        rows = [(100.0, 100.2, 98.0, 98.1)]
        self.add("a1", rows)
        self.book_trade("a1", exit_reason="stop", exit_ms=T0 + BAR,
                        exit_fill=95.0, net=-5.0)      # nowhere near
        r = self.comparison().reconcile()
        self.assertTrue(any(d.field == "exit_fill_price"
                            for d in r.discrepancies))
        self.assertFalse(r.trustworthy)

    def test_a_trade_with_no_tape_is_counted_as_unreplayable(self):
        self.add("a1", [], with_tape=False)
        self.book_trade("a1", exit_reason="stop", exit_ms=T0 + BAR,
                        exit_fill=98.0, net=-2.0)
        r = self.comparison().reconcile()
        self.assertEqual(r.compared, 0)
        self.assertEqual(r.unreplayable[sim.UNREPLAYABLE_NO_TAPE], 1)

    def test_reconciliation_is_per_exit_reason(self):
        rows = [(100.0, 100.2, 98.0, 98.1)]
        self.add("a1", rows)
        self.book_trade("a1", exit_reason="stop", exit_ms=T0 + BAR,
                        exit_fill=97.853, net=-2.2)
        r = self.comparison().reconcile()
        self.assertIn("stop", r.by_exit_reason)
        self.assertEqual(r.by_exit_reason["stop"]["n"], 1)

    def test_nothing_compared_is_never_trustworthy(self):
        self.assertFalse(rep.Reconciliation().trustworthy)

    def test_direction_of_profit_alone_does_not_pass(self):
        # Same sign, wrong magnitude -- must still fail.
        self.add("a1", [(100.0, 103.7, 100.0, 103.7)])
        self.book_trade("a1", exit_reason="target", exit_ms=T0 + BAR,
                        exit_fill=140.0, net=40.0)
        r = self.comparison().reconcile()
        self.assertFalse(r.trustworthy)


# ============================ nothing about the live strategy changed
class TestLiveLogicUnchanged(unittest.TestCase):
    def test_strategy_b_exit_parameters_are_exactly_as_shipped(self):
        from crypto_edge.config import AggressiveCfg
        c = AggressiveCfg()
        self.assertEqual(c.target_r, 2.0)
        self.assertEqual(c.stop_atr_mult, 1.8)
        self.assertEqual(c.breakeven_at_r, 1.0)
        self.assertEqual(c.trail_start_r, 1.5)
        self.assertEqual(c.trail_atr_mult, 2.5)
        self.assertEqual(c.time_stop_hours, 8.0)
        self.assertEqual(c.time_stop_early_hours, 4.0)
        self.assertEqual(c.leverage, 1.0)
        self.assertEqual(c.min_setup_score, 50.0)
        self.assertEqual(c.ladder_ceilings_pct, [50.0, 75.0, 100.0])

    def test_strategy_a_is_untouched(self):
        from crypto_edge.config import StrategyCfg
        a = StrategyCfg()
        self.assertEqual(a.min_score, 55.0)
        self.assertEqual(a.stop_atr_mult, 2.2)
        self.assertEqual(a.donchian_lookback, 48)

    def test_the_simulator_cannot_reach_the_ledger(self):
        import crypto_edge.research.policy_sim as mod
        with open(mod.__file__, encoding="utf-8") as fh:
            src = fh.read()
        for forbidden in ("open_position", "close_position", "set_stop",
                          "update_account", "add_trade", "upsert_excursion"):
            self.assertNotIn(forbidden, src)

    def test_the_live_runtime_does_not_import_the_simulator(self):
        import inspect

        from crypto_edge import aggressive_runtime
        src = inspect.getsource(aggressive_runtime)
        self.assertNotIn("policy_sim", src)
        self.assertNotIn("policy_report", src)

    def test_the_simulator_drives_the_live_exit_functions(self):
        # If this ever stops being true, the simulator has become a second
        # implementation of the rules and will drift from them.
        import crypto_edge.research.policy_sim as mod
        with open(mod.__file__, encoding="utf-8") as fh:
            src = fh.read()
        for needed in ("ex.check_exit", "ex.update_stop", "broker.stop_exit",
                       "broker.exit_fill", "realise_pnl"):
            self.assertIn(needed, src)


if __name__ == "__main__":
    unittest.main()
