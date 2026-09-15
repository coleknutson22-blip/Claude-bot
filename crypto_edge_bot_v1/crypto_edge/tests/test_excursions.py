"""Forward excursion paths: MFE/MAE, target-before-stop, and what we refuse
to guess.

WHAT THESE TESTS ARE PROTECTING
-------------------------------
The whole point of this table is to settle "does a 2R target leave money on the
table" with evidence rather than opinion. That makes two failure modes far more
dangerous than a crash:

  1. A quiet ORDERING assumption. If a stop and a target fall in the same 5m
     candle, OHLC does not say which came first. Resolving that by convention
     ("stops first") would flow straight into the win-rate comparison and the
     answer would be measuring the convention. Those cases must stay marked
     AMBIGUOUS_SAME_BAR and must be excluded from BOTH sides.

  2. A silent DIRECTION error. A short is favoured by the candle LOW. Score it
     off the high and every winning short reads as its worst moment, which
     would make shorts look systematically worse than longs -- a conclusion
     that looks like a finding.

Both get explicit tests below, and both are mutation-tested.
"""
import unittest

import numpy as np

import helpers  # noqa: F401  -- silences the engine's log handlers
from crypto_edge.models import Series
from crypto_edge.research import excursion as X
from crypto_edge.research.excursion_recorder import ExcursionRecorder
from helpers import open_repo, temp_repo

B = "aggressive_momentum_v2"
MIN5 = 300_000
T0 = 1_700_000_000_000


def path(side="long", ref=100.0, stop=None, t0=T0, **over):
    d = 1 if side == "long" else -1
    stop = stop if stop is not None else ref * (1 - 0.02 * d)   # 2% away
    kw = dict(observation_id="o1", symbol="X/USD", strategy=B, side=side,
              direction=d, ref_price=ref, stop_price=stop,
              stop_distance_pct=abs(ref - stop) / ref * 100.0, signal_ms=t0)
    kw.update(over)
    p = X.Excursion(**kw)
    p.ensure_targets()
    return p


def bar(p, i, high, low, close=None):
    """Fold bar `i` (5m apart) with the given extremes."""
    return p.apply_bar(p.signal_ms + i * MIN5, high, low,
                       close if close is not None else (high + low) / 2)


def series(symbol, rows, t0=T0, start_i=0):
    """rows = [(high, low, close), ...] as consecutive 5m candles."""
    n = len(rows)
    ms = np.array([t0 + (start_i + i) * MIN5 for i in range(n)], dtype=np.int64)
    hi = np.array([r[0] for r in rows], dtype=float)
    lo = np.array([r[1] for r in rows], dtype=float)
    cl = np.array([r[2] for r in rows], dtype=float)
    return Series(symbol, "5m", ms, cl.copy(), hi, lo, cl, np.full(n, 1000.0))


class FakeSignal:
    def __init__(self, **kw):
        self.symbol = kw.get("symbol", "X/USD")
        self.side = kw.get("side", "long")
        self.ref_price = kw.get("ref_price", 100.0)
        self.stop_price = kw.get("stop_price", 98.0)
        self.ts_ms = kw.get("ts_ms", T0)

    @property
    def direction(self):
        return -1 if self.side == "short" else 1


# ==================================================== MFE / MAE, both sides
class TestLongExcursion(unittest.TestCase):
    def test_mfe_tracks_the_high(self):
        p = path("long", ref=100.0, stop=98.0)
        bar(p, 1, high=101.5, low=99.8)
        self.assertAlmostEqual(p.mfe_pct, 1.5)

    def test_mae_tracks_the_low(self):
        p = path("long", ref=100.0, stop=95.0)
        bar(p, 1, high=100.2, low=99.1)
        self.assertAlmostEqual(p.mae_pct, -0.9)

    def test_the_best_bar_wins_not_the_last(self):
        p = path("long", ref=100.0, stop=95.0)
        bar(p, 1, high=103.0, low=99.0)
        bar(p, 2, high=100.5, low=99.5)
        self.assertAlmostEqual(p.mfe_pct, 3.0)

    def test_the_time_of_the_extreme_is_recorded(self):
        p = path("long", ref=100.0, stop=95.0)
        bar(p, 1, high=101.0, low=99.0)
        bar(p, 3, high=104.0, low=99.5)
        self.assertEqual(p.minutes_to_mfe(), 15.0)


class TestShortExcursion(unittest.TestCase):
    """A short is favoured by the LOW. This is the mirror, and it must be exact."""

    def test_mfe_tracks_the_low(self):
        p = path("short", ref=100.0, stop=102.0)
        bar(p, 1, high=100.2, low=98.5)
        self.assertAlmostEqual(p.mfe_pct, 1.5)

    def test_mae_tracks_the_high(self):
        p = path("short", ref=100.0, stop=105.0)
        bar(p, 1, high=100.9, low=99.8)
        self.assertAlmostEqual(p.mae_pct, -0.9)

    def test_a_falling_market_is_favourable_for_a_short(self):
        p = path("short", ref=100.0, stop=105.0)
        bar(p, 1, high=99.0, low=97.0)
        self.assertGreater(p.mfe_pct, 0)
        self.assertAlmostEqual(p.mfe_pct, 3.0)

    def test_long_and_short_are_exact_mirrors(self):
        lo = path("long", ref=100.0, stop=98.0)
        sh = path("short", ref=100.0, stop=102.0, observation_id="o2")
        bar(lo, 1, high=102.5, low=99.5)
        bar(sh, 1, high=100.5, low=97.5)
        self.assertAlmostEqual(lo.mfe_pct, sh.mfe_pct)
        self.assertAlmostEqual(lo.mae_pct, sh.mae_pct)


# ====================================================== each target threshold
class TestTargetThresholds(unittest.TestCase):
    def test_every_percentage_target_is_tracked(self):
        p = path("long", ref=100.0, stop=95.0)
        self.assertEqual(
            sorted(l for l in p.touches if l.startswith("pct_")),
            ["pct_1.0", "pct_1.5", "pct_2.0", "pct_2.5", "pct_3.0"])

    def test_r_targets_scale_with_the_stop(self):
        p = path("long", ref=100.0, stop=98.0)        # 2% stop
        self.assertAlmostEqual(p.targets["r_1.0"], 2.0)
        self.assertAlmostEqual(p.targets["r_2.0"], 4.0)
        q = path("long", ref=100.0, stop=99.0)        # 1% stop
        self.assertAlmostEqual(q.targets["r_2.0"], 2.0)

    def test_thresholds_fire_in_order_as_price_runs(self):
        p = path("long", ref=100.0, stop=95.0)
        bar(p, 1, high=101.2, low=100.0)
        self.assertTrue(p.reached("pct_1.0"))
        self.assertFalse(p.reached("pct_1.5"))
        bar(p, 2, high=102.6, low=101.0)
        for lab in ("pct_1.5", "pct_2.0", "pct_2.5"):
            self.assertTrue(p.reached(lab), lab)
        self.assertFalse(p.reached("pct_3.0"))

    def test_a_short_reaches_the_same_thresholds_falling(self):
        p = path("short", ref=100.0, stop=105.0)
        bar(p, 1, high=100.0, low=97.4)
        for lab in ("pct_1.0", "pct_1.5", "pct_2.0"):
            self.assertTrue(p.reached(lab), lab)
        self.assertFalse(p.reached("pct_3.0"))

    def test_an_exact_touch_counts(self):
        # A stop or target is an order resting AT a price. Requiring the bar to
        # trade strictly through it would miss every exact tag.
        p = path("long", ref=100.0, stop=95.0)
        bar(p, 1, high=102.0, low=100.0)
        self.assertTrue(p.reached("pct_2.0"))
        self.assertFalse(p.reached("pct_2.5"))

    def test_an_exact_stop_touch_counts(self):
        p = path("long", ref=100.0, stop=98.0)
        bar(p, 1, high=99.0, low=98.0)
        self.assertTrue(p.stop_touched)

    def test_a_short_stop_is_touched_from_above(self):
        p = path("short", ref=100.0, stop=102.0)
        bar(p, 1, high=102.0, low=101.0)
        self.assertTrue(p.stop_touched)

    def test_just_short_of_a_level_is_not_a_touch(self):
        p = path("long", ref=100.0, stop=95.0)
        bar(p, 1, high=101.99, low=100.0)
        self.assertFalse(p.reached("pct_2.0"))


# ================================================ ordering against the stop
class TestTargetBeforeStop(unittest.TestCase):
    def test_a_clean_target_then_stop_is_before_stop(self):
        p = path("long", ref=100.0, stop=98.0)
        bar(p, 1, high=102.5, low=100.5)      # target, stop nowhere near
        bar(p, 2, high=100.0, low=97.0)       # stop, later
        self.assertEqual(p.touches["pct_2.0"], X.BEFORE_STOP)
        self.assertTrue(p.stop_touched)

    def test_a_short_target_then_stop(self):
        p = path("short", ref=100.0, stop=102.0)
        bar(p, 1, high=99.5, low=97.5)
        bar(p, 2, high=103.0, low=100.0)
        self.assertEqual(p.touches["pct_2.0"], X.BEFORE_STOP)

    def test_the_path_ends_at_the_stop(self):
        # Excursions measured past the stop describe a trade nobody was in,
        # and would inflate MFE for exactly the signals whose MFE matters most.
        p = path("long", ref=100.0, stop=98.0)
        bar(p, 1, high=99.0, low=97.5)        # stopped
        self.assertEqual(p.status, X.COMPLETE)
        self.assertFalse(bar(p, 2, high=110.0, low=100.0))
        self.assertLess(p.mfe_pct, 1.0)


class TestStopBeforeTarget(unittest.TestCase):
    def test_a_level_reached_after_the_stop_is_marked_after_stop(self):
        p = path("long", ref=100.0, stop=98.0)
        bar(p, 1, high=100.5, low=97.0)       # stop, no target yet
        self.assertEqual(p.touches["pct_2.0"], X.AFTER_STOP)

    def test_untouched_levels_settle_as_after_stop_when_the_stop_goes(self):
        p = path("long", ref=100.0, stop=98.0)
        bar(p, 1, high=101.2, low=97.5)       # +1% AND the stop, same bar
        self.assertEqual(p.touches["pct_1.5"], X.AFTER_STOP)
        self.assertEqual(p.touches["pct_3.0"], X.AFTER_STOP)

    def test_a_stopped_path_never_reports_a_reached_target(self):
        p = path("long", ref=100.0, stop=98.0)
        bar(p, 1, high=99.5, low=97.9)
        self.assertFalse(any(p.reached(l) for l in p.touches))


class TestSameBarAmbiguity(unittest.TestCase):
    """The case we refuse to guess about."""

    def test_stop_and_target_in_one_bar_is_ambiguous(self):
        p = path("long", ref=100.0, stop=98.0)
        bar(p, 1, high=102.5, low=97.5)       # both inside one 5m candle
        self.assertEqual(p.touches["pct_2.0"], X.AMBIGUOUS)

    def test_ambiguous_is_not_counted_as_reached(self):
        p = path("long", ref=100.0, stop=98.0)
        bar(p, 1, high=102.5, low=97.5)
        self.assertFalse(p.reached("pct_2.0"))
        self.assertTrue(p.ambiguous("pct_2.0"))

    def test_ambiguity_is_mirrored_for_a_short(self):
        p = path("short", ref=100.0, stop=102.0)
        bar(p, 1, high=102.5, low=97.5)
        self.assertEqual(p.touches["pct_2.0"], X.AMBIGUOUS)
        self.assertFalse(p.reached("pct_2.0"))

    def test_only_the_levels_actually_inside_the_bar_are_ambiguous(self):
        p = path("long", ref=100.0, stop=98.0)
        bar(p, 1, high=102.5, low=97.5)
        self.assertEqual(p.touches["pct_2.0"], X.AMBIGUOUS)
        # +3% was never reached at all, so there is nothing to be unsure about
        self.assertEqual(p.touches["pct_3.0"], X.AFTER_STOP)

    def test_a_target_cleared_on_an_earlier_bar_stays_unambiguous(self):
        p = path("long", ref=100.0, stop=98.0)
        bar(p, 1, high=102.5, low=100.0)      # clean target
        bar(p, 2, high=103.0, low=97.0)       # stop later, with more upside
        self.assertEqual(p.touches["pct_2.0"], X.BEFORE_STOP)
        self.assertEqual(p.touches["pct_3.0"], X.AMBIGUOUS)

    def test_the_report_excludes_ambiguous_from_both_sides(self):
        from crypto_edge.research.forward_test import ExcursionReport
        repo, _ = temp_repo()
        rec = ExcursionRecorder(repo, B)
        # one clean winner, one ambiguous
        for i, (hi, lo) in enumerate(((102.5, 100.0), (102.5, 97.5))):
            sig = FakeSignal(symbol=f"S{i}/USD", ref_price=100.0, stop_price=98.0)
            rec.start_from_signal(f"obs{i}", sig)
            p = repo.get_excursion(f"obs{i}")
            p.apply_bar(T0 + MIN5, hi, lo, (hi + lo) / 2)
            p.status = X.COMPLETE
            repo.upsert_excursion(p)
        rows = {r["target"]: r for r in ExcursionReport(repo, B).hit_rates()}
        r2 = rows["pct_2.0"]
        self.assertEqual(r2["n"], 2)
        self.assertEqual(r2["ambiguous"], 1)
        self.assertEqual(r2["decided"], 1)
        self.assertEqual(r2["hit"], 1)
        self.assertAlmostEqual(r2["hit_rate_pct"], 100.0)


# ============================================================== causality
class TestNoLookahead(unittest.TestCase):
    def test_bars_before_the_signal_are_ignored(self):
        p = path("long", ref=100.0, stop=98.0)
        self.assertFalse(p.apply_bar(p.signal_ms - MIN5, 105.0, 104.0, 104.5))
        self.assertEqual(p.mfe_pct, 0.0)
        self.assertEqual(p.bars, 0)

    def test_the_signal_bar_itself_is_admitted(self):
        p = path("long", ref=100.0, stop=98.0)
        self.assertTrue(p.apply_bar(p.signal_ms, 101.0, 99.5, 100.5))

    def test_the_walk_skips_everything_at_or_before_the_last_bar(self):
        p = path("long", ref=100.0, stop=95.0)
        bar(p, 1, high=101.0, low=99.0)
        self.assertFalse(bar(p, 1, high=109.0, low=99.0))
        self.assertAlmostEqual(p.mfe_pct, 1.0)

    def test_the_recorder_writes_only_to_its_own_table(self):
        # Structural: research must never be able to move a trading decision.
        import crypto_edge.research.excursion_recorder as mod
        with open(mod.__file__, encoding="utf-8") as fh:
            src = fh.read()
        for forbidden in ("open_position", "close_position", "set_stop",
                          "update_account", "add_trade"):
            self.assertNotIn(forbidden, src)


# ================================================== persistence and restart
class TestRestartSafety(unittest.TestCase):
    def setUp(self):
        self.repo, self.path_db = temp_repo()
        self.rec = ExcursionRecorder(self.repo, B)
        self.sig = FakeSignal(ref_price=100.0, stop_price=98.0)

    def test_a_path_is_created_once(self):
        self.assertTrue(self.rec.start_from_signal("obs1", self.sig))
        self.assertFalse(self.rec.start_from_signal("obs1", self.sig))
        self.assertEqual(len(self.repo.get_excursions(B)), 1)

    def test_an_unfinished_path_survives_a_reopen(self):
        self.rec.start_from_signal("obs1", self.sig)
        s = series("X/USD", [(101.0, 99.5, 100.2), (101.8, 100.1, 101.5)],
                   start_i=1)
        self.rec.advance({"X/USD": s}, T0 + 3 * MIN5)
        before = self.repo.get_excursion("obs1")
        self.assertEqual(before.status, X.OPEN)
        self.assertEqual(before.bars, 2)

        repo2 = open_repo(self.path_db)          # a real restart
        after = repo2.get_excursion("obs1")
        self.assertEqual(after.bars, before.bars)
        self.assertAlmostEqual(after.mfe_pct, before.mfe_pct)
        self.assertEqual(after.last_bar_ms, before.last_bar_ms)
        self.assertEqual(after.touches, before.touches)

    def test_replaying_the_same_candles_changes_nothing(self):
        self.rec.start_from_signal("obs1", self.sig)
        s = series("X/USD", [(101.0, 99.5, 100.2), (102.5, 100.1, 102.0)],
                   start_i=1)
        self.rec.advance({"X/USD": s}, T0 + 3 * MIN5)
        first = self.repo.get_excursion("obs1")

        repo2 = open_repo(self.path_db)
        ExcursionRecorder(repo2, B).advance({"X/USD": s}, T0 + 3 * MIN5)
        again = repo2.get_excursion("obs1")
        self.assertEqual(again.bars, first.bars)
        self.assertAlmostEqual(again.mfe_pct, first.mfe_pct)
        self.assertEqual(again.touches, first.touches)

    def test_a_restart_resumes_and_extends_the_same_path(self):
        self.rec.start_from_signal("obs1", self.sig)
        self.rec.advance({"X/USD": series("X/USD", [(101.0, 99.5, 100.2)],
                                          start_i=1)}, T0 + 2 * MIN5)
        repo2 = open_repo(self.path_db)
        rec2 = ExcursionRecorder(repo2, B)
        # the venue returns overlapping history, as it always does
        rec2.advance({"X/USD": series(
            "X/USD", [(101.0, 99.5, 100.2), (103.2, 100.5, 103.0)],
            start_i=1)}, T0 + 3 * MIN5)
        p = repo2.get_excursion("obs1")
        self.assertEqual(p.bars, 2, "the overlapping bar was counted twice")
        self.assertAlmostEqual(p.mfe_pct, 3.2)
        self.assertTrue(p.reached("pct_3.0"))

    def test_a_touch_is_never_double_counted_across_a_restart(self):
        self.rec.start_from_signal("obs1", self.sig)
        s = series("X/USD", [(102.5, 100.0, 102.0)], start_i=1)
        for _ in range(4):                   # four restarts, same candle
            ExcursionRecorder(open_repo(self.path_db), B).advance(
                {"X/USD": s}, T0 + 2 * MIN5)
        p = self.repo.get_excursion("obs1")
        self.assertEqual(p.bars, 1)
        self.assertEqual(p.touches["pct_2.0"], X.BEFORE_STOP)

    def test_a_completed_path_is_not_reopened(self):
        self.rec.start_from_signal("obs1", self.sig)
        self.rec.advance({"X/USD": series("X/USD", [(99.0, 97.0, 97.5)],
                                          start_i=1)}, T0 + 2 * MIN5)
        self.assertEqual(self.repo.get_excursion("obs1").status, X.COMPLETE)
        self.rec.advance({"X/USD": series("X/USD", [(110.0, 105.0, 108.0)],
                                          start_i=2)}, T0 + 3 * MIN5)
        p = self.repo.get_excursion("obs1")
        self.assertEqual(p.status, X.COMPLETE)
        self.assertLess(p.mfe_pct, 1.0)

    def test_open_paths_are_what_a_restart_picks_up(self):
        for i in range(3):
            self.rec.start_from_signal(f"obs{i}", FakeSignal(
                symbol=f"S{i}/USD", ref_price=100.0, stop_price=98.0))
        self.assertEqual(len(open_repo(self.path_db).open_excursions(B)), 3)

    def test_the_horizon_completes_a_path_that_never_stopped(self):
        self.rec.start_from_signal("obs1", self.sig)
        self.rec.advance({}, T0 + (max(X.HORIZONS_MIN) + 1) * 60_000)
        self.assertEqual(self.repo.get_excursion("obs1").status, X.COMPLETE)


class TestSchemaMigration(unittest.TestCase):
    def version(self, repo):
        return int(repo.conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()["value"])

    def test_the_database_reports_the_current_version(self):
        from crypto_edge.storage import db
        repo, _ = temp_repo()
        self.assertEqual(self.version(repo), db.SCHEMA_VERSION)
        self.assertEqual(db.SCHEMA_VERSION, 8)

    def regress_to(self, repo, path, version, *drops):
        repo.ensure_account(B, 10_000.0)
        for table in drops:
            repo.conn.execute(f"DROP TABLE {table}")
        repo.conn.execute("UPDATE meta SET value=? WHERE key='schema_version'",
                          (str(version),))
        repo.conn.commit()
        repo.conn.close()
        return open_repo(path)

    def test_a_v6_database_migrates_and_keeps_its_rows(self):
        repo, p = temp_repo()
        repo2 = self.regress_to(repo, p, 6, "excursion_bars", "market_regime",
                                "excursions")
        self.assertEqual(self.version(repo2), 8)
        self.assertEqual(repo2.get_excursions(B), [])
        self.assertIsNotNone(repo2.get_account(B))

    def test_a_v7_database_gains_the_tape_and_keeps_its_paths(self):
        # v7 paths keep their aggregates; they simply have no tape, which a
        # replay must report rather than invent.
        repo, p = temp_repo()
        rec = ExcursionRecorder(repo, B)
        rec.start_from_signal("obs1", FakeSignal(ref_price=100.0, stop_price=98.0))
        repo2 = self.regress_to(repo, p, 7, "excursion_bars", "market_regime")
        self.assertEqual(self.version(repo2), 8)
        self.assertEqual(len(repo2.get_excursions(B)), 1)
        self.assertEqual(repo2.get_tape("obs1"), [])
        self.assertEqual(repo2.regime_at(T0), "unknown")

    def test_the_new_tables_exist_after_migration(self):
        repo, p = temp_repo()
        repo2 = self.regress_to(repo, p, 7, "excursion_bars", "market_regime")
        names = {r["name"] for r in repo2.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertIn("excursion_bars", names)
        self.assertIn("market_regime", names)


# ============================================ coverage: rejected signals too
class TestRejectedSignalsAreTracked(unittest.TestCase):
    def setUp(self):
        from crypto_edge.engine import TradingEngine
        from crypto_edge.notify.telegram import TelegramNotifier
        from fixtures_fast import engine_feed
        from helpers import engine_config

        syms = [f"S{i:02d}/USDT" for i in range(6)] + ["BTC/USDT"]
        self.feed, _ = engine_feed(syms)
        self.repo, _ = temp_repo()
        self.cfg = engine_config()
        self.cfg.apply_runtime_mode("b")
        self.cfg.universe.broad_static_assets = [s.split("/")[0] for s in syms]
        self.eng = TradingEngine(
            self.cfg, self.repo, self.feed,
            TelegramNotifier("t", "c", self.repo, enabled=False,
                             transport=None, sleep=lambda _: None))
        self.eng.cycle()
        self.rt = self.eng.aggressive

    def test_a_cycle_opens_paths(self):
        self.assertGreater(len(self.repo.get_excursions(B)), 0)

    def test_rejected_signals_get_paths_too(self):
        # The whole exercise is about trades we did NOT take, so tracking only
        # entries would leave the central question unanswerable.
        tracked = {e.observation_id for e in self.repo.get_excursions(B)}
        rejected = [o for o in self.repo.get_observations(strategy=B)
                    if o["decision"] != "ENTERED" and o["id"] in tracked]
        self.assertGreater(len(rejected), 0)

    def test_every_tracked_path_has_a_real_stop(self):
        for e in self.repo.get_excursions(B):
            self.assertGreater(e.stop_distance_pct, 0.0)
            self.assertGreater(e.ref_price, 0.0)

    def test_the_coverage_gap_is_counted_not_hidden(self):
        cov = self.rt.excursions.coverage()
        self.assertEqual(cov["observations"],
                         cov["tracked"] + cov["untracked_no_stop"])
        if cov["untracked_no_stop"]:
            self.assertTrue(cov["reasons"])

    def test_paths_are_direction_correct(self):
        for e in self.repo.get_excursions(B):
            if e.side == "long":
                self.assertLess(e.stop_price, e.ref_price)
            else:
                self.assertGreater(e.stop_price, e.ref_price)

    def test_a_second_cycle_does_not_duplicate_paths(self):
        before = len(self.repo.get_excursions(B))
        self.eng.cycle()
        ids = [e.observation_id for e in self.repo.get_excursions(B)]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(ids), before)


# ================================================================ reporting
class TestExcursionReport(unittest.TestCase):
    def setUp(self):
        from crypto_edge.config import AggressiveCfg
        self.repo, _ = temp_repo()
        self.cfg = AggressiveCfg()
        self.rec = ExcursionRecorder(self.repo, B)

    def add(self, oid, *, side="long", ref=100.0, stop=98.0, score=70.0,
            atr_pct=1.0, regime="bull", bars=(), complete=True):
        self.repo.add_observation({
            "id": oid, "ts_ms": T0, "symbol": f"{oid}/USD", "candle_id": oid,
            "strategy": B, "strategy_version": "1", "decision": "ENTERED",
            "reject_reason": "", "side": side, "score": score, "rank": 1,
            "price": ref, "features": {"atr_pct": atr_pct, "btc_regime": regime}})
        self.rec.start_from_signal(oid, FakeSignal(
            symbol=f"{oid}/USD", side=side, ref_price=ref, stop_price=stop))
        p = self.repo.get_excursion(oid)
        for i, (hi, lo) in enumerate(bars, start=1):
            p.apply_bar(T0 + i * MIN5, hi, lo, (hi + lo) / 2)
        if complete:
            p.status = X.COMPLETE
        self.repo.upsert_excursion(p)
        return p

    def report(self, **kw):
        from crypto_edge.research.forward_test import ExcursionReport
        return ExcursionReport(self.repo, B, cfg=self.cfg, **kw)

    def test_the_crossover_comes_from_config_not_a_constant(self):
        r = self.report()
        self.assertAlmostEqual(r.crossover_atr, 2.0 / (2.0 * 1.8), places=6)

    def test_open_paths_are_excluded_from_rates(self):
        # Counting an unfinished path as a miss biases every rate downward.
        self.add("o1", bars=[(102.5, 100.0)], complete=False)
        self.assertEqual(self.report().complete(), [])
        self.assertEqual(self.report().hit_rates()[0]["n"], 0)

    def test_hit_rates_count_only_clean_touches(self):
        self.add("o1", bars=[(102.5, 100.0)])          # clean +2%
        self.add("o2", bars=[(101.2, 100.0)])          # only +1%
        rows = {r["target"]: r for r in self.report().hit_rates()}
        self.assertEqual(rows["pct_1.0"]["hit"], 2)
        self.assertEqual(rows["pct_2.0"]["hit"], 1)
        self.assertEqual(rows["pct_3.0"]["hit"], 0)

    def test_q1_finds_the_stalled_between_2pct_and_2r_case(self):
        # atr 1.0% -> stop 1.8% -> 2R needs +3.6%. A move to +2.5% is exactly
        # the case the whole question is about.
        self.add("o1", ref=100.0, stop=98.2, atr_pct=1.0,
                 bars=[(102.5, 100.0), (99.0, 98.0)])
        q = self.report().stalled_between_2pct_and_2r()
        self.assertEqual(q["decided"], 1)
        self.assertEqual(q["stalled"], 1)
        self.assertAlmostEqual(q["stalled_pct"], 100.0)

    def test_q1_does_not_count_a_trade_that_reached_2r(self):
        self.add("o1", ref=100.0, stop=98.2, atr_pct=1.0,
                 bars=[(104.0, 100.0)])
        q = self.report().stalled_between_2pct_and_2r()
        self.assertEqual(q["stalled"], 0)
        self.assertEqual(q["reached_2r"], 1)

    def test_q1_ignores_signals_below_the_crossover(self):
        self.add("o1", ref=100.0, stop=99.5, atr_pct=0.28,
                 bars=[(102.5, 100.0)])
        self.assertEqual(self.report().stalled_between_2pct_and_2r()["n"], 0)

    def test_q2_finds_where_2r_fires_and_a_fixed_2pct_would_not(self):
        # atr 0.3% -> stop 0.54% -> 2R needs only +1.08%
        self.add("o1", ref=100.0, stop=99.46, atr_pct=0.3,
                 bars=[(101.2, 100.0), (99.5, 99.4)])
        q = self.report().two_r_cheaper_than_2pct()
        self.assertEqual(q["two_r_only"], 1)
        self.assertAlmostEqual(q["two_r_only_pct"], 100.0)

    def test_ambiguous_paths_are_reported_as_a_share(self):
        self.add("o1", bars=[(102.5, 97.5)])       # same-bar stop + target
        self.add("o2", bars=[(102.5, 100.0)])
        amb = self.report().ambiguity_share()
        self.assertEqual(amb["paths"], 2)
        self.assertEqual(amb["any_ambiguous"], 1)
        self.assertAlmostEqual(amb["share_pct"], 50.0)

    def test_breakdowns_split_long_and_short(self):
        self.add("o1", side="long", ref=100.0, stop=98.0, bars=[(102.5, 100.0)])
        self.add("o2", side="short", ref=100.0, stop=102.0, bars=[(100.0, 97.5)])
        g = self.report().breakdowns()["side"]
        self.assertEqual(sorted(g), ["long", "short"])
        self.assertEqual(len(g["long"]), 1)
        self.assertEqual(len(g["short"]), 1)

    def test_breakdowns_cover_every_requested_dimension(self):
        self.add("o1", bars=[(102.5, 100.0)])
        g = self.report().breakdowns()
        for dim in ("side", "score_bucket", "confidence_bucket",
                    "atr_bucket", "btc_regime"):
            self.assertIn(dim, g)
            self.assertTrue(g[dim], dim)

    def test_atr_pct_falls_back_to_the_stop_when_unrecorded(self):
        p = self.add("o1", atr_pct=None, ref=100.0, stop=98.2,
                     bars=[(101.0, 100.0)])
        got = self.report().atr_pct(p)
        self.assertAlmostEqual(got, 1.8 / 1.8, places=3)

    def test_excursion_stats_summarise_mfe_and_mae(self):
        self.add("o1", ref=100.0, stop=95.0, bars=[(103.0, 99.0)])
        self.add("o2", ref=100.0, stop=95.0, bars=[(101.0, 98.0)])
        e = self.report().excursion_stats()
        self.assertEqual(e["n"], 2)
        self.assertAlmostEqual(e["mfe_max"], 3.0)
        self.assertAlmostEqual(e["mae_min"], -2.0)

    def test_small_samples_are_flagged_not_hidden(self):
        self.add("o1", bars=[(102.5, 100.0)])
        row = self.report(min_sample=30).hit_rates()[0]
        self.assertFalse(row["sufficient_sample"])
        self.assertGreater(row["n"], 0)


# ==================================== nothing about trading logic changed
class TestTradingLogicUnchanged(unittest.TestCase):
    def test_strategy_b_gates_are_exactly_as_shipped(self):
        from crypto_edge.config import AggressiveCfg
        c = AggressiveCfg()
        self.assertEqual(c.min_setup_score, 50.0)
        self.assertEqual(c.min_rel_volume, 0.9)
        self.assertEqual(c.min_atr_pct, 0.25)
        self.assertEqual(c.min_ema_struct_15m, 0.5)
        self.assertEqual(c.stop_atr_mult, 1.8)

    def test_the_exit_model_is_exactly_as_shipped(self):
        from crypto_edge.config import AggressiveCfg
        c = AggressiveCfg()
        self.assertEqual(c.target_r, 2.0)
        self.assertEqual(c.breakeven_at_r, 1.0)
        self.assertEqual(c.trail_start_r, 1.5)
        self.assertEqual(c.time_stop_hours, 8.0)
        self.assertEqual(c.leverage, 1.0)
        self.assertEqual(c.ladder_ceilings_pct, [50.0, 75.0, 100.0])

    def test_strategy_a_is_untouched(self):
        from crypto_edge.config import StrategyCfg
        a = StrategyCfg()
        self.assertEqual(a.min_score, 55.0)
        self.assertEqual(a.stop_atr_mult, 2.2)
        self.assertEqual(a.donchian_lookback, 48)
        self.assertEqual(a.min_adx, 20.0)

    def test_the_recorder_is_the_only_new_thing_in_the_entry_path(self):
        # `_journal` may record and track. It must not size, price or fill.
        import inspect

        from crypto_edge.aggressive_runtime import AggressiveRuntime
        src = inspect.getsource(AggressiveRuntime._journal)
        for forbidden in ("open_position", "entry_fill", "plan(", "notional"):
            self.assertNotIn(forbidden, src)


if __name__ == "__main__":
    unittest.main()


# =========================================================== THE TAPE (A2)
def series15(symbol, rows, t0=T0, start_i=0):
    """15m candles. rows = [(high, low, close), ...]."""
    step = 900_000
    n = len(rows)
    ms = np.array([t0 + (start_i + i) * step for i in range(n)], dtype=np.int64)
    hi = np.array([r[0] for r in rows], dtype=float)
    lo = np.array([r[1] for r in rows], dtype=float)
    cl = np.array([r[2] for r in rows], dtype=float)
    return Series(symbol, "15m", ms, cl.copy(), hi, lo, cl, np.full(n, 1000.0))


def ramp15(n=120, start=100.0, step=0.4, t0=None, width=1.0):
    """A rising 15m series long enough for ema_structure (needs > 50 bars).

    The bar RANGE breathes rather than being constant: a fixed-width ramp has a
    constant ATR by construction, which would let a frozen-ATR bug pass the
    causality test for the wrong reason.
    """
    # Centred on T0 so the series EXTENDS past the signal, as a live 15m feed
    # does. Ending it at T0 would make every forward 5m bar resolve to the same
    # last-closed 15m bar -- correct behaviour, but it would hide a frozen-ATR
    # bug behind a fixture that never gave the context a chance to advance.
    t0 = t0 if t0 is not None else T0 - (n - 24) * 900_000
    rows = []
    for i in range(n):
        mid = start + i * step
        w = width * (1.0 + 0.6 * ((i * 7) % 5) / 4.0)
        rows.append((mid + w, mid - w, mid))
    return series15("X/USD", rows, t0=t0)


class TestTapePersistence(unittest.TestCase):
    def setUp(self):
        self.repo, self.db = temp_repo()
        self.rec = ExcursionRecorder(self.repo, B)
        self.sig = FakeSignal(ref_price=100.0, stop_price=98.0)

    def test_a_bar_is_stored_for_every_bar_walked(self):
        self.rec.start_from_signal("obs1", self.sig)
        s = series("X/USD", [(101.0, 99.5, 100.2), (101.8, 100.1, 101.5)],
                   start_i=1)
        self.rec.advance({"X/USD": s}, T0 + 3 * MIN5)
        tape = self.repo.get_tape("obs1")
        self.assertEqual(len(tape), 2)
        self.assertEqual(self.repo.get_excursion("obs1").bars, len(tape))

    def test_the_tape_carries_full_ohlc_not_just_extremes(self):
        # `stop_exit` needs the OPEN to detect a gap through the stop. Without
        # it a replay cannot tell a gap fill from a clean stop fill.
        self.rec.start_from_signal("obs1", self.sig)
        self.rec.advance({"X/USD": series(
            "X/USD", [(101.0, 99.5, 100.2)], start_i=1)}, T0 + 2 * MIN5)
        bar = self.repo.get_tape("obs1")[0]
        for field in ("open", "high", "low", "close"):
            self.assertIn(field, bar)
            self.assertIsNotNone(bar[field])

    def test_elapsed_time_since_the_signal_is_stored(self):
        self.rec.start_from_signal("obs1", self.sig)
        self.rec.advance({"X/USD": series(
            "X/USD", [(101.0, 99.5, 100.2), (101.2, 100.0, 101.0)],
            start_i=3)}, T0 + 6 * MIN5)
        got = [b["elapsed_min"] for b in self.repo.get_tape("obs1")]
        self.assertEqual(got, [15.0, 20.0])

    def test_bars_are_returned_in_time_order(self):
        self.rec.start_from_signal("obs1", self.sig)
        self.rec.advance({"X/USD": series(
            "X/USD", [(101.0, 99.5, 100.2)] * 5, start_i=1)}, T0 + 9 * MIN5)
        ms = [b["open_ms"] for b in self.repo.get_tape("obs1")]
        self.assertEqual(ms, sorted(ms))

    def test_the_venue_rounding_rules_travel_with_the_path(self):
        from crypto_edge.models import MarketMeta
        meta = MarketMeta("X/USD", "X", "USD", True, amount_precision=6,
                          price_precision=4, min_amount=1e-6, min_cost=5.0)
        self.rec.start_from_signal("obs1", self.sig, meta)
        stored = self.repo.get_excursion("obs1").meta
        self.assertEqual(stored["price_precision"], 4)
        self.assertEqual(stored["amount_precision"], 6)

    def test_a_missing_market_is_recorded_as_unknown_not_invented(self):
        self.rec.start_from_signal("obs1", self.sig, None)
        self.assertEqual(self.repo.get_excursion("obs1").meta, {})


class TestTapeContextPersistence(unittest.TestCase):
    """ATR, 15m structure and regime -- the non-price exit inputs."""

    def setUp(self):
        self.repo, self.db = temp_repo()
        self.rec = ExcursionRecorder(self.repo, B)
        self.rec.start_from_signal(
            "obs1", FakeSignal(ref_price=100.0, stop_price=98.0))

    def advance(self, n=3, s15=None, start_i=1):
        s5 = series("X/USD", [(101.0 + i, 99.5 + i, 100.2 + i) for i in range(n)],
                    start_i=start_i)
        self.rec.advance({"X/USD": s5}, T0 + (start_i + n + 1) * MIN5,
                         frames_by_symbol={"X/USD": s15} if s15 is not None else None)
        return self.repo.get_tape("obs1")

    def test_the_chandelier_atr_is_stored_per_bar(self):
        tape = self.advance(s15=ramp15())
        self.assertTrue(tape)
        for b in tape:
            self.assertIsNotNone(b["atr"])
            self.assertGreater(b["atr"], 0)

    def test_the_atr_source_timeframe_is_recorded(self):
        # The live code falls back from 15m to 5m on a short 15m history, so a
        # replay has to know which series the number came from.
        tape = self.advance(s15=ramp15())
        self.assertEqual({b["atr_tf"] for b in tape}, {"15m"})

    def test_a_short_15m_history_falls_back_exactly_as_live_does(self):
        tape = self.advance(s15=ramp15(n=5))      # below the live 20-bar guard
        self.assertEqual({b["atr_tf"] for b in tape}, {"5m"})

    def test_the_momentum_invalidation_input_is_stored(self):
        tape = self.advance(s15=ramp15())
        for b in tape:
            self.assertIsNotNone(b["ema_struct_15m"])
            self.assertGreaterEqual(b["ema_struct_15m"], -1.0)
            self.assertLessEqual(b["ema_struct_15m"], 1.0)

    def test_structure_is_absent_rather_than_guessed_on_short_history(self):
        # Live computes ema_structure only when len(s15) > 50.
        tape = self.advance(s15=ramp15(n=30))
        self.assertTrue(all(b["ema_struct_15m"] is None for b in tape))

    def test_no_15m_series_means_no_invented_substitute(self):
        tape = self.advance(s15=None)
        self.assertTrue(all(b["ema_struct_15m"] is None for b in tape))

    def test_the_regime_in_force_at_the_bar_is_stored(self):
        self.repo.record_regime(T0 - MIN5, "bull", 55.0)
        tape = self.advance(s15=ramp15())
        self.assertEqual({b["btc_regime"] for b in tape}, {"bull"})

    def test_a_bar_predating_any_reading_is_unknown_not_assumed(self):
        self.repo.record_regime(T0 + 100 * MIN5, "bear", 20.0)
        tape = self.advance(s15=ramp15())
        self.assertEqual({b["btc_regime"] for b in tape}, {"unknown"})

    def test_a_backfilled_bar_gets_the_historical_regime(self):
        # The whole reason for a timeline: a bar folded late must not be
        # stamped with the regime in force when the recorder caught up.
        self.repo.record_regime(T0 - MIN5, "bull", 55.0)
        self.repo.record_regime(T0 + 50 * MIN5, "bear", 15.0)
        tape = self.advance(n=2, s15=ramp15(), start_i=1)
        self.assertEqual({b["btc_regime"] for b in tape}, {"bull"})

    def test_the_timeline_only_records_changes(self):
        self.assertTrue(self.repo.record_regime(T0, "bull"))
        self.assertFalse(self.repo.record_regime(T0 + MIN5, "bull"))
        self.assertTrue(self.repo.record_regime(T0 + 2 * MIN5, "bear"))
        self.assertEqual(len(self.repo.regime_timeline()), 2)


class TestTapeIsCausal(unittest.TestCase):
    """The failure that would quietly invalidate every replay."""

    def test_the_15m_context_advances_with_the_bar(self):
        # Computing last_valid(atr(s15)) once and stamping it on every bar
        # would apply a number derived from the END of the window to bars at
        # its start -- a trail replayed on information the bot never had.
        from crypto_edge.research.tape import TapeContext
        s15 = ramp15(n=120)
        s5 = series("X/USD", [(101.0 + i*0.5, 99.5 + i*0.5, 100.2 + i*0.5)
                              for i in range(24)], start_i=0)
        ctx = TapeContext(s5, s15)
        atrs = [ctx.bar_at(i, T0).atr for i in range(len(s5))]
        self.assertGreater(len(set(atrs)), 1,
                           "the ATR is frozen -- it was stamped from the end")

    def test_a_bar_only_sees_15m_bars_that_had_closed(self):
        from crypto_edge.research.tape import TapeContext
        s15 = ramp15(n=120)
        s5 = series("X/USD", [(101.0, 99.5, 100.2)] * 4, start_i=0)
        ctx = TapeContext(s5, s15)
        last_close_15 = int(s15.open_ms[-1]) + 900_000
        for i in range(len(s5)):
            j = ctx._index_15m_as_of(int(s5.open_ms[i]) + MIN5)
            if j >= 0:
                self.assertLessEqual(int(s15.open_ms[j]) + 900_000,
                                     int(s5.open_ms[i]) + MIN5)
            self.assertLess(j, len(s15))
        self.assertGreater(last_close_15, 0)

    def test_the_forward_fill_is_last_valid_evaluated_in_place(self):
        from crypto_edge.research.tape import _ffill
        a = np.array([np.nan, np.nan, 1.0, np.nan, 3.0, np.nan])
        got = _ffill(a)
        self.assertTrue(np.isnan(got[0]) and np.isnan(got[1]))
        self.assertEqual(list(got[2:]), [1.0, 1.0, 3.0, 3.0])

    def test_structure_matches_the_live_function_at_the_last_bar(self):
        # The per-index series must agree with features.ema_structure where
        # they overlap, or the replay is using a different indicator.
        from crypto_edge.research.tape import _structure_series
        from crypto_edge.strategy.features import ema_structure
        close = np.asarray(ramp15(n=120).close, dtype=float)
        self.assertAlmostEqual(_structure_series(close)[-1],
                               ema_structure(close), places=9)

    def test_structure_matches_live_on_a_truncated_prefix_too(self):
        from crypto_edge.research.tape import _structure_series
        from crypto_edge.strategy.features import ema_structure
        close = np.asarray(ramp15(n=120).close, dtype=float)
        full = _structure_series(close)
        for cut in (60, 80, 100):
            self.assertAlmostEqual(full[cut - 1], ema_structure(close[:cut]),
                                   places=9,
                                   msg=f"prefix of {cut} bars disagrees with live")


class TestTapeIdempotence(unittest.TestCase):
    def setUp(self):
        self.repo, self.db = temp_repo()
        self.rec = ExcursionRecorder(self.repo, B)
        self.rec.start_from_signal(
            "obs1", FakeSignal(ref_price=100.0, stop_price=98.0))

    def test_a_duplicate_bar_cannot_be_inserted(self):
        s = series("X/USD", [(101.0, 99.5, 100.2)], start_i=1)
        self.rec.advance({"X/USD": s}, T0 + 2 * MIN5)
        self.assertEqual(self.repo.tape_bar_count("obs1"), 1)
        # write the same bar again, directly, bypassing the walk guard
        from crypto_edge.research.tape import TapeBar
        dup = TapeBar(open_ms=T0 + MIN5, elapsed_min=5.0, open=1.0, high=1.0,
                      low=1.0, close=1.0, atr=1.0, atr_tf="15m",
                      ema_struct_15m=0.0, btc_regime="bull")
        self.assertEqual(self.repo.add_tape_bars("obs1", [dup]), 0)
        self.assertEqual(self.repo.tape_bar_count("obs1"), 1)

    def test_overlapping_venue_history_replays_to_the_same_tape(self):
        first = series("X/USD", [(101.0, 99.5, 100.2)], start_i=1)
        self.rec.advance({"X/USD": first}, T0 + 2 * MIN5)
        # the venue returns the old bar again plus a new one, as it always does
        overlap = series("X/USD", [(101.0, 99.5, 100.2), (102.0, 100.5, 101.5)],
                         start_i=1)
        ExcursionRecorder(open_repo(self.db), B).advance(
            {"X/USD": overlap}, T0 + 3 * MIN5)
        tape = self.repo.get_tape("obs1")
        self.assertEqual(len(tape), 2)
        self.assertEqual(len({b["open_ms"] for b in tape}), 2)

    def test_many_restarts_over_one_candle_store_one_bar(self):
        s = series("X/USD", [(101.0, 99.5, 100.2)], start_i=1)
        for _ in range(5):
            ExcursionRecorder(open_repo(self.db), B).advance(
                {"X/USD": s}, T0 + 2 * MIN5)
        self.assertEqual(self.repo.tape_bar_count("obs1"), 1)

    def test_a_partially_recorded_path_resumes_from_its_last_bar(self):
        self.rec.advance({"X/USD": series(
            "X/USD", [(101.0, 99.5, 100.2), (101.5, 100.0, 101.0)],
            start_i=1)}, T0 + 3 * MIN5)
        self.assertEqual(self.repo.tape_bar_count("obs1"), 2)

        rec2 = ExcursionRecorder(open_repo(self.db), B)
        rec2.advance({"X/USD": series(
            "X/USD", [(101.0, 99.5, 100.2), (101.5, 100.0, 101.0),
                      (102.4, 101.0, 102.0)], start_i=1)}, T0 + 4 * MIN5)
        tape = self.repo.get_tape("obs1")
        self.assertEqual(len(tape), 3)
        self.assertEqual(self.repo.get_excursion("obs1").bars, 3)

    def test_a_completed_path_is_never_rewritten(self):
        self.rec.advance({"X/USD": series(
            "X/USD", [(99.0, 97.0, 97.5)], start_i=1)}, T0 + 2 * MIN5)
        p = self.repo.get_excursion("obs1")
        self.assertEqual(p.status, X.COMPLETE)
        before = self.repo.get_tape("obs1")
        self.rec.advance({"X/USD": series(
            "X/USD", [(110.0, 105.0, 108.0)] * 3, start_i=2)}, T0 + 6 * MIN5)
        self.assertEqual(self.repo.get_tape("obs1"), before)

    def test_the_bar_that_completes_a_path_is_on_its_tape(self):
        # The stop bar is the most important row in the whole tape.
        self.rec.advance({"X/USD": series(
            "X/USD", [(101.0, 100.0, 100.5), (99.0, 97.0, 97.5)],
            start_i=1)}, T0 + 3 * MIN5)
        tape = self.repo.get_tape("obs1")
        self.assertEqual(len(tape), 2)
        self.assertEqual(tape[-1]["low"], 97.0)


class TestTapeRetention(unittest.TestCase):
    def test_the_horizon_is_covered_by_the_default_retention(self):
        from crypto_edge.config import AggressiveCfg
        from crypto_edge.research import excursion as XX
        horizon_days = max(XX.HORIZONS_MIN) / (60 * 24)
        self.assertGreaterEqual(AggressiveCfg().tape_retention_days,
                                horizon_days)

    def test_pruning_drops_whole_completed_paths_only(self):
        repo, _ = temp_repo()
        rec = ExcursionRecorder(repo, B)
        from crypto_edge.research.tape import TapeBar
        for oid, ts, status in (("old", T0, X.COMPLETE),
                                ("new", T0 + 10**7, X.COMPLETE),
                                ("open", T0, X.OPEN)):
            rec.start_from_signal(oid, FakeSignal(
                symbol=f"{oid}/USD", ref_price=100.0, stop_price=98.0,
                ts_ms=ts))
            p = repo.get_excursion(oid)
            p.status = status
            repo.upsert_excursion(p)
            repo.add_tape_bars(oid, [TapeBar(
                open_ms=ts + MIN5, elapsed_min=5.0, open=1.0, high=1.0,
                low=1.0, close=1.0, atr=1.0, atr_tf="15m",
                ema_struct_15m=0.0, btc_regime="bull")])
        repo.prune_tape(T0 + 10**6)
        self.assertEqual(repo.tape_bar_count("old"), 0)
        self.assertEqual(repo.tape_bar_count("new"), 1)
        self.assertEqual(repo.tape_bar_count("open"), 1,
                         "an OPEN path lost its tape and can no longer resume")


class TestTapeInTheEngine(unittest.TestCase):
    def setUp(self):
        from crypto_edge.engine import TradingEngine
        from crypto_edge.notify.telegram import TelegramNotifier
        from fixtures_fast import engine_feed
        from helpers import engine_config

        syms = [f"S{i:02d}/USDT" for i in range(6)] + ["BTC/USDT"]
        self.feed, _ = engine_feed(syms)
        self.repo, _ = temp_repo()
        self.cfg = engine_config()
        self.cfg.apply_runtime_mode("b")
        self.cfg.universe.broad_static_assets = [s.split("/")[0] for s in syms]
        self.eng = TradingEngine(
            self.cfg, self.repo, self.feed,
            TelegramNotifier("t", "c", self.repo, enabled=False,
                             transport=None, sleep=lambda _: None))
        self.eng.cycle()

    def test_a_real_cycle_writes_tape(self):
        self.assertGreater(self.repo.tape_bar_count(), 0)

    def test_the_regime_timeline_is_written(self):
        self.assertTrue(self.repo.regime_timeline())

    def test_tape_context_is_populated_from_the_live_frames(self):
        for p in self.repo.get_excursions(B):
            tape = self.repo.get_tape(p.observation_id)
            if tape:
                self.assertIsNotNone(tape[0]["atr"])
                self.assertEqual(tape[0]["atr_tf"], "15m")
                return
        self.fail("no path recorded any tape")

    def test_a_second_cycle_extends_rather_than_duplicates(self):
        before = self.repo.tape_bar_count()
        self.eng.cycle()
        rows = self.repo.conn.execute(
            "SELECT observation_id, open_ms, COUNT(*) n FROM excursion_bars "
            "GROUP BY observation_id, open_ms HAVING n > 1").fetchall()
        self.assertEqual(rows, [])
        self.assertGreaterEqual(self.repo.tape_bar_count(), before)

    def test_both_sides_record_tape(self):
        for p in self.repo.get_excursions(B):
            self.assertIn(p.side, ("long", "short"))
        self.assertTrue(self.repo.get_excursions(B))


class TestTapeNeverReachesExecution(unittest.TestCase):
    def test_the_tape_module_cannot_touch_the_ledger(self):
        import crypto_edge.research.tape as mod
        with open(mod.__file__, encoding="utf-8") as fh:
            src = fh.read()
        for forbidden in ("open_position", "close_position", "set_stop",
                          "update_account", "add_trade", "broker"):
            self.assertNotIn(forbidden, src)

    def test_the_recorder_runs_after_every_trading_decision(self):
        # `_advance_excursions` must be called AFTER the entry loop's decisions
        # are settled, never before one of them.
        import inspect

        from crypto_edge.aggressive_runtime import AggressiveRuntime
        src = inspect.getsource(AggressiveRuntime.scan_and_enter)
        self.assertLess(src.index("_advance_excursions"), src.index("taken = 0"))

    def test_no_tape_field_is_read_by_the_exit_engine(self):
        import inspect

        from crypto_edge.portfolio import aggressive_exits
        src = inspect.getsource(aggressive_exits)
        for forbidden in ("excursion", "tape", "TapeBar", "get_tape"):
            self.assertNotIn(forbidden, src)
