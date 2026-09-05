"""Runtime readiness: which strategies trade, what they report, what preflight
proves before a live forward test starts.

WHY THIS FILE EXISTS SEPARATELY
-------------------------------
Everything here is about OPERATION rather than strategy. None of it changes a
threshold, a weight or a sizing rule -- and several tests exist specifically to
prove that, because a "runtime-only" change that quietly moved a gate would be
indistinguishable from one that did not unless something checks.

The preflight is driven offline against a Kraken/USD-SHAPED fixture: USD quote,
USD symbols, and 5m/15m/1h/4h all served, which is what the real venue does.
That verifies the harness -- what it checks, what it calls a failure, and that
its summary cannot contradict its own checks. What it cannot verify is the
network, which is exactly what running it for real is for.
"""
import io
import unittest
from contextlib import redirect_stdout

import helpers  # noqa: F401  -- silences the engine's log handlers
from crypto_edge.config import AggressiveCfg, Config, StrategyCfg
from crypto_edge.notify import formatters as fmt
from crypto_edge.notify.telegram import TelegramNotifier
from crypto_edge.verify_live import (FORWARD_TEST_CONTRACT, VerifyReport,
                                     verify_fast_timeframes,
                                     verify_restart_recovery, verify_schema,
                                     verify_strategy_b_contract)
from helpers import engine_config, temp_repo

A = "trend_breakout"
B = "aggressive_momentum_v2"


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = fn(*a, **kw)
    return out, buf.getvalue()


class RecordingTransport:
    def __init__(self, ok=True):
        self.ok, self.sent = ok, []

    def send(self, token, chat_id, text, timeout):
        if not self.ok:
            return False, "simulated outage"
        self.sent.append(text)
        return True, "ok"


# ===================================================== 1. runtime mode
class TestRuntimeMode(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_strategy_a_only(self):
        self.cfg.apply_runtime_mode("a")
        self.assertTrue(self.cfg.strategy.enabled)
        self.assertFalse(self.cfg.aggressive.enabled)
        self.assertEqual(self.cfg.enabled_strategies(), [A])

    def test_strategy_b_only(self):
        self.cfg.apply_runtime_mode("b")
        self.assertFalse(self.cfg.strategy.enabled)
        self.assertTrue(self.cfg.aggressive.enabled)
        self.assertEqual(self.cfg.enabled_strategies(), [B])

    def test_both_together(self):
        self.cfg.apply_runtime_mode("both")
        self.assertEqual(self.cfg.enabled_strategies(), [A, B])

    def test_the_mode_round_trips(self):
        for mode in Config.RUNTIME_MODES:
            self.cfg.apply_runtime_mode(mode)
            self.assertEqual(self.cfg.runtime_mode(), mode)

    def test_a_strategy_name_selects_its_own_mode(self):
        self.assertEqual(self.cfg.apply_runtime_mode(B), "b")
        self.assertEqual(self.cfg.apply_runtime_mode(A), "a")

    def test_an_unknown_mode_is_refused_not_guessed(self):
        with self.assertRaises(ValueError) as e:
            self.cfg.apply_runtime_mode("aggresive")   # typo on purpose
        self.assertIn("aggresive", str(e.exception))

    def test_both_disabled_is_reported_rather_than_silently_idle(self):
        self.cfg.strategy.enabled = False
        self.cfg.aggressive.enabled = False
        self.assertEqual(self.cfg.runtime_mode(), "none")
        self.assertEqual(self.cfg.enabled_strategies(), [])

    def test_the_default_config_runs_both(self):
        self.assertTrue(StrategyCfg().enabled)
        self.assertTrue(AggressiveCfg().enabled)


class EngineModeCase(unittest.TestCase):
    """The mode has to reach the CYCLE, not just the config object."""

    def setUp(self):
        from crypto_edge.engine import TradingEngine
        from fixtures_fast import engine_feed

        self.symbols = [f"S{i:02d}/USDT" for i in range(6)] + ["BTC/USDT"]
        self.feed, self.markets = engine_feed(self.symbols)
        self.repo, _ = temp_repo()
        self.cfg = engine_config()
        self.cfg.universe.broad_static_assets = [s.split("/")[0]
                                                 for s in self.symbols]
        self._Engine = TradingEngine

    def engine(self, mode):
        self.cfg.apply_runtime_mode(mode)
        notifier = TelegramNotifier("t", "c", self.repo, enabled=False,
                                    transport=None, sleep=lambda _: None)
        return self._Engine(self.cfg, self.repo, self.feed, notifier)


class TestModeReachesTheCycle(EngineModeCase):
    def test_b_only_takes_no_strategy_a_entries(self):
        eng = self.engine("b")
        eng.cycle()
        self.assertEqual(len(eng.account.positions()), 0,
                         "Strategy A traded during a Strategy B forward test")
        self.assertGreater(len(eng.aggressive.account.positions()), 0)

    def test_b_only_still_evaluates_strategy_b(self):
        eng = self.engine("b")
        eng.cycle()
        self.assertGreater(eng.aggressive.status.evaluated, 0)

    def test_a_only_does_not_run_strategy_b_at_all(self):
        # Disabled and holding nothing means NOT ATTACHED: no scan, no deep
        # fetches, no journal rows. "Observe without trading" would be a third
        # mode with a real per-cycle cost, and inventing one nobody asked for
        # is not the runtime's call to make.
        eng = self.engine("a")
        eng.cycle()
        self.assertIsNone(eng.aggressive)
        self.assertEqual(self.repo.get_positions(B), [])
        self.assertEqual(self.repo.get_observations(strategy=B), [])

    def test_a_only_still_lets_a_trade(self):
        eng = self.engine("a")
        eng.cycle()
        self.assertTrue(eng.cfg.strategy.enabled)
        self.assertEqual(eng.status.halt_reason, "")

    def test_both_runs_both(self):
        eng = self.engine("both")
        eng.cycle()
        self.assertGreater(len(eng.aggressive.account.positions()), 0)
        self.assertTrue(eng.cfg.strategy.enabled)

    def test_a_strategy_still_holding_scans_but_never_enters(self):
        # The one case where a disabled strategy still runs: it has positions.
        # It must manage them and journal what it sees, and open nothing.
        eng = self.engine("b")
        eng.cycle()
        self.assertGreater(len(eng.aggressive.account.positions()), 0)
        before = len(self.repo.get_positions(B))

        self.cfg.apply_runtime_mode("a")
        eng2 = self.engine("a")
        self.assertIsNotNone(eng2.aggressive)
        eng2.cycle()
        self.assertLessEqual(len(self.repo.get_positions(B)), before,
                             "a disabled strategy opened a new position")
        entered = [o for o in self.repo.get_observations(strategy=B)
                   if o["decision"] == "ENTERED"]
        self.assertEqual(len(entered), before,
                         "an entry was journalled while entries were off")

    def test_disabling_a_strategy_never_strands_its_open_positions(self):
        # Open something as B, then turn B off, and make sure the runtime is
        # still attached to manage it. A stop nobody watches is the worst
        # outcome available here.
        eng = self.engine("b")
        eng.cycle()
        self.assertGreater(len(eng.aggressive.account.positions()), 0)
        held = self.repo.get_positions(B)

        self.cfg.apply_runtime_mode("a")
        notifier = TelegramNotifier("t", "c", self.repo, enabled=False,
                                    transport=None, sleep=lambda _: None)
        eng2 = self._Engine(self.cfg, self.repo, self.feed, notifier)
        self.assertIsNotNone(eng2.aggressive,
                             "B was detached while still holding positions")
        self.assertEqual(len(eng2.aggressive.account.positions()), len(held))

    def test_a_disabled_strategy_with_nothing_open_is_not_attached(self):
        self.cfg.apply_runtime_mode("a")
        repo, _ = temp_repo()
        notifier = TelegramNotifier("t", "c", repo, enabled=False,
                                    transport=None, sleep=lambda _: None)
        eng = self._Engine(self.cfg, repo, self.feed, notifier)
        self.assertIsNone(eng.aggressive)


# ============================================ 2. the Strategy B contract
class TestStrategyBPaperAccount(unittest.TestCase):
    """The parameters the forward test is DEFINED by, against literals."""

    def setUp(self):
        self.cfg = Config()
        self.repo, _ = temp_repo()

    def test_every_contract_term_holds(self):
        for label, getter, expected in FORWARD_TEST_CONTRACT:
            with self.subTest(term=label):
                self.assertEqual(getter(self.cfg), expected)

    def test_the_contract_covers_everything_that_was_asked_for(self):
        # A contract that quietly stops checking a term is worse than no
        # contract, because it still reports PASS.
        labels = {label for label, _, _ in FORWARD_TEST_CONTRACT}
        for required in ("starting equity", "max open positions", "leverage",
                         "ladder ceilings %", "confidence = identity",
                         "min setup score", "min relative volume",
                         "min ATR %", "min 15m structure",
                         "max loss % of equity", "daily buffer fraction"):
            self.assertIn(required, labels)

    def test_the_sub_account_starts_at_ten_thousand(self):
        rep = VerifyReport()
        quiet(verify_strategy_b_contract, self.cfg, self.repo, rep)
        acct = self.repo.get_account(B)
        self.assertEqual(float(acct["starting_equity"]), 10_000.0)
        self.assertEqual(float(acct["cash"]), 10_000.0)

    def test_the_two_ledgers_are_independent(self):
        self.repo.ensure_account(A, 10_000.0)
        self.repo.ensure_account(B, 10_000.0)
        names = {a["strategy"] for a in self.repo.all_accounts()}
        self.assertEqual(names, {A, B})

    def test_the_contract_check_fails_when_a_term_drifts(self):
        rep = VerifyReport()
        self.cfg.aggressive.leverage = 3.0        # someone edits the TOML
        quiet(verify_strategy_b_contract, self.cfg, self.repo, rep)
        self.assertFalse(rep.passed)
        self.assertIn("leverage", rep.failures_for("bcfg"))

    def test_the_max_loss_formula_is_evaluated_not_described(self):
        from crypto_edge.portfolio.ladder import loss_ceiling
        # 1% of equity binds while the daily buffer is intact...
        self.assertAlmostEqual(
            loss_ceiling(10_000, 300, max_loss_pct=1.0,
                         daily_buffer_fraction=0.60), 100.0)
        # ...and 60% of what is LEFT binds once the day has gone badly.
        self.assertAlmostEqual(
            loss_ceiling(10_000, 100, max_loss_pct=1.0,
                         daily_buffer_fraction=0.60), 60.0)


# ==================================================== 3 + 4. telegram
class TestStrategyBTelegram(unittest.TestCase):
    """Every message must say WHICH experiment it is reporting on."""

    def entry(self, **over):
        kw = dict(strategy=B, symbol="SOL/USD", side="long", setup_score=84.7,
                  confidence=84.7, bucket="80-89", multiplier=0.8, slot=2,
                  ceiling_cash=3_750.0, target_notional=3_000.0,
                  notional=2_500.0, binding_constraint="risk_cap",
                  entry_price=142.35, qty=17.56, stop=138.0, target=151.05,
                  stop_distance_pct=3.06, expected_loss=76.5,
                  expected_loss_pct=0.77, leverage=1.0, equity=9_980.0,
                  free_cash_after=7_480.0, open_positions=2, max_positions=3,
                  btc_regime="bull", breadth=46.0)
        kw.update(over)
        return fmt.aggressive_entry(**kw)

    def exit(self, **over):
        kw = dict(strategy=B, symbol="SOL/USD", side="short",
                  exit_reason="trailing_stop", entry_price=142.35,
                  exit_price=136.10, qty=17.56, held_s=9_240.0, gross=109.75,
                  financing=0.62, fees=4.31, slippage=3.02, net=101.80,
                  return_pct=4.07, equity=10_081.8, open_positions=1,
                  max_positions=3, total_pnl=81.8)
        kw.update(over)
        return fmt.aggressive_exit(**kw)

    # --- entry ---------------------------------------------------------
    def test_the_entry_names_the_strategy(self):
        self.assertIn(B, self.entry())

    def test_the_entry_carries_every_required_field(self):
        m = self.entry()
        for token in ("SOL/USD", "LONG", "84.7", "80-89", "2/3", "80%",
                      "3,750.00", "2,500.00", "risk_cap", "138", "151.05",
                      "76.50", "0.77%", "1x"):
            with self.subTest(token=token):
                self.assertIn(token, m)

    def test_setup_score_and_confidence_are_shown_as_separate_lines(self):
        m = self.entry(setup_score=84.7, confidence=70.0, bucket="70-79")
        self.assertIn("Setup score: 84.7", m)
        self.assertIn("Confidence: 70.0", m)

    def test_a_short_entry_says_short(self):
        m = self.entry(side="short")
        self.assertIn("SHORT", m)
        self.assertNotIn("🟩", m)

    def test_the_binding_constraint_is_always_stated(self):
        for c in ("ladder", "risk_cap", "exposure", "cash"):
            self.assertIn(c, self.entry(binding_constraint=c))

    # --- exit ----------------------------------------------------------
    def test_the_exit_names_the_strategy(self):
        self.assertIn(B, self.exit())

    def test_the_exit_carries_every_required_field(self):
        m = self.exit()
        for token in ("SOL/USD", "SHORT", "trailing_stop", "109.75", "0.62",
                      "4.31", "3.02", "101.80", "10,081.80", "1/3"):
            with self.subTest(token=token):
                self.assertIn(token, m)

    def test_financing_is_its_own_line_not_folded_into_fees(self):
        m = self.exit(financing=12.5, fees=4.31)
        self.assertIn("Financing: -$12.50", m)
        self.assertIn("Fees: -$4.31", m)

    def test_a_loss_is_labelled_a_loss(self):
        self.assertIn("LOSS", self.exit(net=-40.0))
        self.assertIn("PROFIT", self.exit(net=40.0))

    def test_hold_time_is_human_readable(self):
        self.assertIn("Held:", self.exit(held_s=9_240.0))

    # --- heartbeat -----------------------------------------------------
    def beat(self, **over):
        kw = dict(strategy=B, uptime_s=93_600.0, equity=10_240.0,
                  today_pnl=140.0, total_pnl=240.0, open_positions=2,
                  max_positions=3, longs=1, shorts=1, btc_regime="bull",
                  breadth=46.0, candidates=13, shortlist=12, evaluated=9,
                  entries=4, exits=2, drawdown_pct=1.4,
                  daily_buffer_remaining=160.0, halted=False,
                  last_scan_ms=1_700_000_000_000)
        kw.update(over)
        return fmt.aggressive_heartbeat(**kw)

    def test_the_heartbeat_carries_every_required_field(self):
        m = self.beat()
        for token in ("aggressive_momentum_v2", "10,240.00", "+$140.00",
                      "+$240.00", "2/3", "1 long", "1 short", "bull", "46%",
                      "Candidates: 13", "Shortlist: 12", "Setups evaluated: 9",
                      "Entries: 4", "Exits: 2", "1.40%", "160.00", "Uptime"):
            with self.subTest(token=token):
                self.assertIn(token, m)

    def test_a_halt_is_impossible_to_miss(self):
        m = self.beat(halted=True, halt_reason="daily loss limit reached")
        self.assertIn("HALTED", m)
        self.assertIn("daily loss limit reached", m)

    def test_entries_being_off_is_stated_with_its_consequence(self):
        m = self.beat(entries_enabled=False)
        self.assertIn("Entries OFF", m)
        self.assertIn("still managed", m)


class TestHeartbeatNumbersComeFromTheLedger(unittest.TestCase):
    """The fields are computed from the account, not passed in by a caller."""

    def setUp(self):
        from unittest import mock

        from crypto_edge.aggressive_runtime import AggressiveRuntime
        from crypto_edge.execution.paper_broker import PaperBroker
        from crypto_edge.research.journal import ResearchJournal
        from crypto_edge.strategy.base import MarketContext

        self.repo, _ = temp_repo()
        self.cfg = Config()
        self.cfg.telegram.enabled = False
        self.rt = AggressiveRuntime(self.cfg, self.repo, mock.MagicMock(),
                                    mock.MagicMock(), PaperBroker(7.5, 6.0, 15.0),
                                    ResearchJournal(self.repo))
        self.ctx = MarketContext(ts_ms=1_700_000_000_000, btc_regime="bull",
                                 btc_regime_score=60.0, breadth_pct=46.0)

    def test_a_flat_account_reports_its_starting_equity(self):
        f = self.rt.heartbeat_fields({}, self.ctx)
        self.assertEqual(f["equity"], 10_000.0)
        self.assertEqual(f["total_pnl"], 0.0)
        self.assertEqual(f["open_positions"], 0)
        self.assertEqual(f["max_positions"], 3)

    def test_the_scan_funnel_is_reported_from_the_last_scan(self):
        self.rt.status.candidates = 13
        self.rt.status.shortlist = 12
        self.rt.status.evaluated = 9
        f = self.rt.heartbeat_fields({}, self.ctx)
        self.assertEqual((f["candidates"], f["shortlist"], f["evaluated"]),
                         (13, 12, 9))

    def test_the_regime_comes_from_the_context(self):
        f = self.rt.heartbeat_fields({}, self.ctx)
        self.assertEqual(f["btc_regime"], "bull")
        self.assertEqual(f["breadth"], 46.0)

    def test_the_daily_buffer_is_what_is_left_before_the_halt(self):
        f = self.rt.heartbeat_fields({}, self.ctx)
        self.assertGreater(f["daily_buffer_remaining"], 0.0)

    def test_entries_disabled_is_carried_through_to_the_message(self):
        self.cfg.aggressive.enabled = False
        self.assertFalse(self.rt.heartbeat_fields({}, self.ctx)["entries_enabled"])

    def test_every_field_the_message_needs_is_supplied(self):
        # The runtime passes its dict straight into the formatter, so a missing
        # key is a TypeError at 3am rather than a test failure now.
        fmt.aggressive_heartbeat(**self.rt.heartbeat_fields({}, self.ctx))


# ================================================= 5 + 6. reporting
class ReportingCase(unittest.TestCase):
    """A Strategy B ledger with closed trades on both sides."""

    def setUp(self):
        from crypto_edge.performance import PerformanceCalculator
        self.repo, _ = temp_repo()
        self.repo.ensure_account(A, 10_000.0)
        self.repo.ensure_account(B, 10_000.0)
        self._add(B, "long", "80-89", 1, "ladder", net=120.0, financing=0.0)
        self._add(B, "short", "60-69", 2, "risk_cap", net=-40.0, financing=1.5)
        self._add(B, "long", "95+", 3, "ladder", net=60.0, financing=0.0)
        # Strategy A's own trade, which must never appear in a B report.
        self._add(A, "long", None, None, None, net=999.0, financing=0.0)
        self.perf = PerformanceCalculator(self.repo, B)

    def _add(self, strategy, side, bucket, slot, binding, *, net, financing):
        from crypto_edge.models import ClosedTrade
        n = len(self.repo.get_trades()) + 1
        journal = {}
        if bucket:
            journal = {"conf_bucket": bucket, "ladder_slot": slot,
                       "binding_constraint": binding, "final_notional": 2_500.0,
                       "expected_loss_cash": 95.0}
        self.repo.add_trade(ClosedTrade(
            id=f"t{n}", position_id=f"p{n}", symbol=f"X{n}/USD",
            strategy=strategy, strategy_version="1", qty=1.0,
            entry_ref_price=100.0, entry_fill_price=100.0, entry_ms=0,
            exit_ref_price=110.0, exit_fill_price=110.0, exit_ms=3_600_000,
            exit_reason="target", initial_stop=98.0, final_stop=98.0,
            gross_pnl=net + 5.0 + financing, fees=3.0, slippage_cost=2.0,
            net_pnl=net, side=side, financing=financing, return_pct=net / 25.0,
            account_return_pct=net / 100.0, mfe=1.0, mae=-1.0,
            duration_s=3600.0, equity_after=10_000.0 + net, journal=journal))


class TestPerformanceIsScopedToOneLedger(ReportingCase):
    def test_the_report_excludes_the_other_strategy(self):
        rep = self.perf.report().as_dict()
        self.assertEqual(rep["trading"]["closed_trades"], 3)
        self.assertAlmostEqual(rep["account"]["net_pnl"], 140.0)

    def test_categories_exclude_the_other_strategy_too(self):
        # by_category once read EVERY trade in the database regardless of
        # strategy, which silently blended the two ledgers it exists to compare.
        cats = self.perf.by_category()
        total = sum(r["n"] for r in cats["side"])
        self.assertEqual(total, 3, "a Strategy A trade leaked into B's buckets")

    def test_financing_is_reported_separately(self):
        rep = self.perf.report().as_dict()
        self.assertAlmostEqual(rep["account"]["total_financing"], 1.5)

    def test_the_long_and_short_sides_are_reported_apart(self):
        sides = self.perf.report().as_dict()["sides"]
        self.assertEqual(sides["long"]["trades"], 2)
        self.assertEqual(sides["short"]["trades"], 1)
        self.assertAlmostEqual(sides["long"]["net_pnl"], 180.0)
        self.assertAlmostEqual(sides["short"]["net_pnl"], -40.0)
        self.assertAlmostEqual(sides["short"]["financing"], 1.5)

    def test_an_empty_side_is_still_reported(self):
        from crypto_edge.performance import PerformanceCalculator
        repo, _ = temp_repo()
        repo.ensure_account(B, 10_000.0)
        sides = PerformanceCalculator(repo, B).report().as_dict()["sides"]
        self.assertIn("short", sides)
        self.assertEqual(sides["short"]["trades"], 0)

    def test_results_by_confidence_bucket(self):
        rows = {r["bucket"]: r for r in self.perf.by_category()["conf_bucket"]}
        self.assertEqual(set(rows), {"80-89", "60-69", "95+"})
        self.assertAlmostEqual(rows["60-69"]["net_pnl"], -40.0)

    def test_results_by_ladder_slot(self):
        rows = {r["bucket"]: r for r in self.perf.by_category()["ladder_slot"]}
        self.assertEqual(set(rows), {"1", "2", "3"})

    def test_results_by_binding_constraint(self):
        rows = {r["bucket"]: r for r in self.perf.by_category()["binding_constraint"]}
        self.assertEqual(set(rows), {"ladder", "risk_cap"})
        self.assertEqual(rows["risk_cap"]["n"], 1)

    def test_average_risk_reads_strategy_bs_vocabulary(self):
        # A records `risk_amount`, B records `expected_loss_cash`. Reading only
        # the first reported a flat zero for every Strategy B ledger.
        rep = self.perf.report().as_dict()
        self.assertAlmostEqual(rep["risk"]["average_risk_per_trade"], 95.0)

    def test_small_buckets_are_flagged_never_hidden(self):
        for row in self.perf.by_category()["conf_bucket"]:
            self.assertFalse(row["sufficient_sample"])
            self.assertGreater(row["n"], 0)


class TestForwardTestResearch(unittest.TestCase):
    def setUp(self):
        from crypto_edge.research.forward_test import ForwardTestReport
        self.repo, _ = temp_repo()
        self.repo.ensure_account(B, 10_000.0)
        self._obs("o1", "long", 42.0, "REJECTED_STRATEGY",
                  "relative volume 0.71 < 0.90")
        self._obs("o2", "long", 44.0, "REJECTED_STRATEGY",
                  "relative volume 0.44 < 0.90")
        self._obs("o3", "short", 61.0, "REJECTED_RISK",
                  "max new entries per cycle reached")
        self._obs("o4", "long", 88.0, "ENTERED", "")
        self._obs("o5", "short", 30.0, "REJECTED_STRATEGY",
                  "atr 0.11% < 0.25%")
        # what the rejected setups then did (HYPOTHETICAL, never executed)
        self.repo.add_counterfactual("o1", 24, 100.0, 104.0, 4.0, 1)
        self.repo.add_counterfactual("o2", 24, 100.0, 101.0, 1.0, 1)
        self.repo.add_counterfactual("o5", 24, 100.0, 97.0, -3.0, 1)
        self.rep = ForwardTestReport(self.repo, B, min_sample=2)

    def _obs(self, oid, side, score, decision, reason):
        self.repo.add_observation({
            "id": oid, "ts_ms": 1, "symbol": f"{oid}/USD",
            "candle_id": f"c{oid}", "strategy": B, "strategy_version": "1",
            "decision": decision, "reject_reason": reason, "side": side,
            "score": score, "rank": 1, "price": 100.0,
            "features": {"rel_volume": 0.71, "atr_pct": 0.11,
                         "ema_struct_15m": 0.2}})

    def test_rejected_setups_are_inspectable(self):
        rows = {r.bucket: r for r in self.rep.rejection_counts()}
        self.assertIn("relative volume", rows)
        self.assertEqual(rows["relative volume"].n, 2)

    def test_measured_values_do_not_fragment_the_buckets(self):
        # Two different rel-volume readings are ONE reason, not two.
        buckets = [r.bucket for r in self.rep.rejection_counts()]
        self.assertEqual(len(buckets), len(set(buckets)))
        self.assertEqual(sum(1 for b in buckets if "relative volume" in b), 1)

    def test_counterfactual_returns_are_attached_to_the_reason(self):
        rows = {r.bucket: r for r in self.rep.rejection_counts()}
        self.assertAlmostEqual(rows["relative volume"].avg, 2.5)

    def test_a_short_signal_is_scored_in_its_own_direction(self):
        # o5 is a SHORT whose price fell 3%: the signal was RIGHT, so the
        # filter that rejected it cost something, and the sign must say so.
        rows = {r.bucket: r for r in self.rep.rejection_counts()}
        self.assertAlmostEqual(rows["atr"].avg, 3.0)

    def test_long_versus_short_is_broken_out(self):
        sides = self.rep.by_side()
        self.assertEqual(sides["long"]["evaluated"], 3)
        self.assertEqual(sides["short"]["evaluated"], 2)
        self.assertEqual(sides["long"]["entered"], 1)
        self.assertEqual(sides["short"]["entered"], 0)

    def test_score_buckets_span_taken_and_rejected_setups(self):
        rows = {r.bucket: r for r in self.rep.score_buckets()}
        self.assertIn("40-50", rows)
        self.assertIn("80-90", rows)
        self.assertEqual(rows["80-90"].extra["entered"], 1)

    def test_each_gate_reports_what_it_rejected_and_what_happened(self):
        gates = {g["gate"]: g for g in self.rep.gate_sensitivity()}
        self.assertEqual(gates["rel_volume"]["rejected"], 2)
        self.assertAlmostEqual(gates["rel_volume"]["avg_return_pct"], 2.5)
        self.assertEqual(gates["rel_volume"]["config_key"],
                         "aggressive.min_rel_volume")

    def test_the_five_questions_all_have_a_home(self):
        gates = {g["gate"] for g in self.rep.gate_sensitivity()}
        self.assertLessEqual({"rel_volume", "atr_pct", "setup_score"}, gates)
        # 60-69 losing money, and 90+ outperforming, are read here:
        self.assertTrue(hasattr(self.rep, "confidence_buckets"))

    def test_undersized_samples_are_flagged(self):
        self.assertFalse(self.rep.sufficient(1))
        self.assertTrue(self.rep.sufficient(2))

    def test_the_report_is_scoped_to_one_strategy(self):
        from crypto_edge.research.forward_test import ForwardTestReport
        self.repo.add_observation({
            "id": "a1", "ts_ms": 1, "symbol": "A/USD", "candle_id": "ca1",
            "strategy": A, "strategy_version": "1", "decision": "ENTERED",
            "reject_reason": "", "side": "long", "score": 70.0, "rank": 1,
            "price": 100.0, "features": {}})
        self.assertEqual(len(ForwardTestReport(self.repo, B).obs), 5)
        self.assertEqual(len(ForwardTestReport(self.repo, A).obs), 1)


class TestReasonNormalisation(unittest.TestCase):
    def normalise(self, s):
        from crypto_edge.research.forward_test import normalise_reason
        return normalise_reason(s)

    def test_measurements_are_stripped(self):
        self.assertEqual(self.normalise("setup score 42.0 < 50.0"), "setup score")

    def test_a_timeframe_in_the_rule_name_survives(self):
        # "15m structure" names the gate. Stripping the 15 leaves "m
        # structure", which no longer matches the config key it refers to.
        self.assertEqual(self.normalise("15m structure 0.20 < 0.50"),
                         "15m structure")
        self.assertEqual(self.normalise("hostile 1h ema -0.8 < -0.5"),
                         "hostile 1h ema")

    def test_no_placeholder_debris_leaks_into_the_label(self):
        for text in ("15m structure 0.2 < 0.5", "5m data unavailable",
                     "4h and 15m both stale"):
            out = self.normalise(text)
            self.assertNotIn("\x00", out)
            self.assertNotIn("z", out.replace("hostile", ""))

    def test_different_readings_of_one_rule_group_together(self):
        self.assertEqual(self.normalise("relative volume 0.71 < 0.90"),
                         self.normalise("relative volume 0.44 < 0.90"))

    def test_an_empty_reason_is_named_not_blank(self):
        self.assertEqual(self.normalise(""), "unspecified")


# ================================================== 7. preflight harness
class PreflightCase(unittest.TestCase):
    """Kraken/USD-SHAPED fixtures: USD quote, USD symbols, 5m/15m/1h/4h."""

    def setUp(self):
        from fixtures_fast import engine_feed
        self.symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "DASH/USD"]
        self.feed, self.markets = engine_feed(self.symbols)
        self.repo, _ = temp_repo()
        self.cfg = engine_config()
        self.cfg.exchange.name = "kraken"
        self.cfg.exchange.quote = "USD"
        self.cfg.resolve_symbols()
        self.cfg.apply_runtime_mode("b")
        self.rep = VerifyReport()


class TestPreflightFastTimeframes(PreflightCase):
    def test_it_checks_the_timeframes_strategy_b_actually_reads(self):
        quiet(verify_fast_timeframes, self.cfg, self.feed, self.rep)
        names = " ".join(s.name for s in self.rep.steps)
        self.assertIn("5m", names)
        self.assertIn("15m", names)

    def test_it_passes_when_the_venue_serves_them(self):
        quiet(verify_fast_timeframes, self.cfg, self.feed, self.rep)
        self.assertTrue(self.rep.passed)
        self.assertTrue(self.rep.fact("b_5m").startswith("OK"))

    def test_a_venue_that_cannot_serve_5m_fails_the_preflight(self):
        # This is the failure that matters: Strategy A's 1h/4h would still be
        # perfect, so a preflight checking only those would say READY.
        from crypto_edge.data.feed import DataUnavailable

        class No5m:
            name = "kraken"

            def fetch_ohlcv(inner, symbol, timeframe, limit):
                if timeframe == "5m":
                    raise DataUnavailable("venue does not serve 5m")
                return self.feed.fetch_ohlcv(symbol, timeframe, limit)

        quiet(verify_fast_timeframes, self.cfg, No5m(), self.rep)
        self.assertFalse(self.rep.passed)
        self.assertIn("FAILED", self.rep.fact("b_5m"))

    def test_stale_fast_data_is_a_failure_not_a_warning(self):
        import numpy as np
        from crypto_edge.models import Series
        old = self.feed._series[("BTC/USD", "5m")]
        shifted = Series(old.symbol, old.timeframe,
                         old.open_ms - 6 * 3_600_000, old.open, old.high,
                         old.low, old.close, old.volume)
        self.feed._series[("BTC/USD", "5m")] = shifted
        quiet(verify_fast_timeframes, self.cfg, self.feed, self.rep)
        self.assertFalse(self.rep.passed)


class TestPreflightSchemaAndRestart(PreflightCase):
    def test_the_schema_is_at_the_current_version(self):
        quiet(verify_schema, self.cfg, self.repo, self.rep)
        self.assertTrue(self.rep.passed, self.rep.fact("schema"))

    def test_the_financing_column_is_required(self):
        quiet(verify_schema, self.cfg, self.repo, self.rep)
        names = [s.name for s in self.rep.steps]
        self.assertIn("financing column present", names)

    def test_a_stale_schema_version_fails(self):
        # Defensive: `init_db` migrates on open, so reaching preflight at v5
        # means a migration ran and did NOT record itself. That is precisely
        # the case worth catching, because everything else would look fine.
        self.repo.conn.execute(
            "UPDATE meta SET value='5' WHERE key='schema_version'")
        quiet(verify_schema, self.cfg, self.repo, self.rep)
        self.assertFalse(self.rep.passed)
        # The summary line is OVERRULED by its own failed check rather than
        # reporting the stale version as though it were fine.
        self.assertIn("FAILED", self.rep.fact("schema"))
        self.assertIn("schema at current version", self.rep.fact("schema"))

    def test_restart_recovery_reads_through_a_second_connection(self):
        self.cfg.engine.db_path = self.repo.conn.execute(
            "PRAGMA database_list").fetchone()["file"]
        self.repo.ensure_account(B, 10_000.0)
        quiet(verify_restart_recovery, self.cfg, self.repo, self.rep)
        self.assertTrue(self.rep.passed)
        self.assertEqual(self.rep.fact("restart"), "OK")

    def test_restart_recovery_covers_the_claimed_candles(self):
        # These are what stop a restart re-entering the same signal.
        self.cfg.engine.db_path = self.repo.conn.execute(
            "PRAGMA database_list").fetchone()["file"]
        quiet(verify_restart_recovery, self.cfg, self.repo, self.rep)
        names = [s.name for s in self.rep.steps]
        self.assertIn("candles survive a reopen", names)
        self.assertIn("positions survive a reopen", names)


class TestPreflightSummaryCannotLie(PreflightCase):
    """The summary must never say OK for something a check below rejected."""

    def test_a_failed_check_overrules_its_own_summary_line(self):
        self.rep.facts["b_5m"] = "OK (looks great)"
        self.rep.add("data freshness (5m)", False, "3 bars behind",
                     topic="b_5m")
        self.assertIn("FAILED", self.rep.fact("b_5m"))

    def test_the_verdict_follows_the_checks(self):
        quiet(verify_schema, self.cfg, self.repo, self.rep)
        self.assertIn("READY FOR FORWARD TEST",
                      self.rep.render_preflight(self.cfg))
        self.rep.add("something critical", False, "broken")
        self.assertIn("NOT READY", self.rep.render_preflight(self.cfg))

    def test_the_summary_names_every_requested_item(self):
        out = self.rep.render_preflight(self.cfg)
        for line in ("5m DATA", "15m DATA", "1h DATA", "QUOTES", "SPREADS",
                     "UNIVERSE", "BTC REGIME", "BREADTH", "TELEGRAM",
                     "DATABASE MIGRATION", "RESTART RECOVERY",
                     "STRATEGY B CONFIG", "RUNTIME MODE"):
            with self.subTest(line=line):
                self.assertIn(line, out)

    def test_an_unrun_step_says_so_rather_than_passing_quietly(self):
        out = self.rep.render_preflight(self.cfg)
        self.assertIn("NOT RUN", out)


# ============================================== 8. nothing was retuned
class TestNoStrategyLogicChanged(unittest.TestCase):
    """A runtime pass that moved a threshold would be indistinguishable from
    one that did not, unless something checks. This checks."""

    def test_strategy_b_signal_gates_are_exactly_as_shipped(self):
        c = AggressiveCfg()
        self.assertEqual(c.min_setup_score, 50.0)
        self.assertEqual(c.min_rel_volume, 0.9)
        self.assertEqual(c.rel_volume_bars, 4)
        self.assertEqual(c.min_atr_pct, 0.25)
        self.assertEqual(c.min_ema_struct_15m, 0.5)
        self.assertEqual(c.max_hostile_ema_1h, -0.5)
        self.assertEqual(c.min_momentum_agree, 3)
        self.assertEqual(c.min_vote_atr, 0.15)
        self.assertEqual(c.shortlist_size, 12)

    def test_the_ladder_and_sizing_model_are_unchanged(self):
        c = AggressiveCfg()
        self.assertEqual(c.ladder_ceilings_pct, [50.0, 75.0, 100.0])
        self.assertEqual(c.max_open_positions, 3)
        self.assertEqual(c.leverage, 1.0)
        self.assertEqual(c.max_loss_pct, 1.0)
        self.assertEqual(c.daily_buffer_fraction, 0.60)
        self.assertEqual(c.min_confidence, 60.0)
        self.assertTrue(c.confidence_is_identity)
        self.assertEqual(
            [(b["min"], b["mult"]) for b in c.confidence_buckets],
            [(60.0, 0.40), (70.0, 0.60), (80.0, 0.80), (90.0, 0.90),
             (95.0, 1.00)])

    def test_the_exit_model_is_unchanged(self):
        c = AggressiveCfg()
        self.assertEqual(c.target_r, 2.0)
        self.assertEqual(c.breakeven_at_r, 1.0)
        self.assertEqual(c.trail_start_r, 1.5)
        self.assertEqual(c.trail_atr_mult, 2.5)
        self.assertEqual(c.time_stop_hours, 8.0)
        self.assertEqual(c.time_stop_early_hours, 4.0)
        self.assertEqual(c.short_borrow_bps_per_day, 15.0)
        self.assertEqual(c.short_force_close_at_loss_pct, 80.0)

    def test_strategy_a_is_untouched(self):
        a = StrategyCfg()
        self.assertEqual(a.min_score, 55.0)
        self.assertEqual(a.donchian_lookback, 48)
        self.assertEqual(a.stop_atr_mult, 2.2)
        self.assertEqual(a.min_adx, 20.0)
        self.assertEqual(a.entry_timeframe, "1h")
        self.assertEqual(a.regime_timeframe, "4h")


if __name__ == "__main__":
    unittest.main()
