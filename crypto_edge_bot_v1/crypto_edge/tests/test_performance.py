"""Performance metric tests.

Every expected value here is computed by hand in the test, not read back from
the implementation, so a silent change in the maths fails loudly.
"""
import unittest

from crypto_edge.models import ClosedTrade
from crypto_edge.performance import (MIN_DAYS_FOR_RATIOS, MIN_TRADES_FOR_RATIOS,
                                     PerformanceCalculator)
from helpers import STRATEGY, open_repo, temp_repo

DAY = 86_400_000
T0 = 1_760_000_000_000


def trade(repo, i, net, *, gross=None, fees=1.0, slip=0.5, symbol="SOL/USDT",
          reason="stop_loss", duration_s=7200.0, equity_after=10_000.0,
          risk=50.0):
    """Insert a closed trade whose components reconcile to `net`."""
    gross = net + fees + slip if gross is None else gross
    t = ClosedTrade(
        id=f"t{i}", position_id=f"p{i}", symbol=symbol,
        strategy="trend_breakout", strategy_version="v1",
        qty=10.0, entry_ref_price=100.0, entry_fill_price=100.0,
        entry_ms=T0 + i * DAY, exit_ref_price=100.0 + net / 10.0,
        exit_fill_price=100.0 + net / 10.0, exit_ms=T0 + i * DAY + 7_200_000,
        exit_reason=reason, initial_stop=95.0, final_stop=95.0,
        gross_pnl=gross, fees=fees, slippage_cost=slip, net_pnl=net,
        return_pct=net / 1000.0 * 100.0, account_return_pct=net / 10_000.0 * 100.0,
        mfe=abs(net), mae=-abs(net) / 2, duration_s=duration_s,
        equity_after=equity_after, journal={"risk_amount": risk})
    repo.add_trade(t)
    return t


class TestTradingStatistics(unittest.TestCase):
    def setUp(self):
        self.repo, self.path = temp_repo()
        self.repo.ensure_account(STRATEGY, 10_000.0)
        self.calc = PerformanceCalculator(self.repo)

    def test_win_rate(self):
        for i, net in enumerate([100.0, 100.0, 100.0, -50.0]):
            trade(self.repo, i, net)
        rep = self.calc.report()
        self.assertEqual(rep.trading["closed_trades"], 4)
        self.assertEqual(rep.trading["wins"], 3)
        self.assertEqual(rep.trading["losses"], 1)
        self.assertAlmostEqual(rep.trading["win_rate_pct"], 75.0)

    def test_breakeven_trade_counts_as_a_loss(self):
        # A trade that nets exactly zero has consumed fees and slippage; it is
        # not a win. Counting it as one would flatter the win rate.
        trade(self.repo, 0, 0.0)
        rep = self.calc.report()
        self.assertEqual(rep.trading["losses"], 1)
        self.assertEqual(rep.trading["wins"], 0)

    def test_profit_factor(self):
        for i, net in enumerate([200.0, 100.0, -50.0, -50.0]):
            trade(self.repo, i, net)
        # gross profit 300, gross loss 100 -> 3.0
        self.assertAlmostEqual(self.calc.report().trading["profit_factor"], 3.0)

    def test_profit_factor_with_no_losses_is_infinite_not_a_crash(self):
        trade(self.repo, 0, 100.0)
        self.assertEqual(self.calc.report().trading["profit_factor"], float("inf"))

    def test_profit_factor_with_no_trades_is_zero(self):
        self.assertEqual(self.calc.report().trading["profit_factor"], 0.0)

    def test_expectancy_matches_the_hand_calculation(self):
        for i, net in enumerate([100.0, 100.0, -50.0, -50.0]):
            trade(self.repo, i, net)
        # win rate .5, avg win 100, avg loss 50 -> .5*100 - .5*50 = 25
        self.assertAlmostEqual(
            self.calc.report().trading["expectancy_per_trade"], 25.0)

    def test_average_winner_and_loser(self):
        for i, net in enumerate([150.0, 50.0, -30.0]):
            trade(self.repo, i, net)
        tr = self.calc.report().trading
        self.assertAlmostEqual(tr["average_winner"], 100.0)
        self.assertAlmostEqual(tr["average_loser"], -30.0)
        self.assertAlmostEqual(tr["largest_winner"], 150.0)
        self.assertAlmostEqual(tr["largest_loser"], -30.0)

    def test_average_trade_is_net_over_count(self):
        for i, net in enumerate([100.0, -40.0, 10.0]):
            trade(self.repo, i, net)
        self.assertAlmostEqual(self.calc.report().trading["average_trade"],
                               70.0 / 3)

    def test_consecutive_streaks(self):
        for i, net in enumerate([10.0, 10.0, 10.0, -5.0, -5.0, 10.0]):
            trade(self.repo, i, net)
        tr = self.calc.report().trading
        self.assertEqual(tr["max_consecutive_wins"], 3)
        self.assertEqual(tr["max_consecutive_losses"], 2)


class TestCostAccounting(unittest.TestCase):
    def setUp(self):
        self.repo, _ = temp_repo()
        self.repo.ensure_account(STRATEGY, 10_000.0)
        self.calc = PerformanceCalculator(self.repo)

    def test_fees_and_slippage_are_reported_separately(self):
        for i in range(3):
            trade(self.repo, i, 100.0, fees=2.0, slip=1.5)
        acct = self.calc.report().account
        self.assertAlmostEqual(acct["total_fees"], 6.0)
        self.assertAlmostEqual(acct["estimated_slippage_cost"], 4.5)

    def test_gross_minus_costs_equals_net_in_aggregate(self):
        for i, net in enumerate([100.0, -60.0, 25.0]):
            trade(self.repo, i, net, fees=2.0, slip=1.5)
        a = self.calc.report().account
        self.assertAlmostEqual(
            a["net_pnl"], a["gross_pnl"] - a["total_fees"]
            - a["estimated_slippage_cost"], places=9)

    def test_costs_can_turn_a_gross_winner_into_a_net_loser(self):
        trade(self.repo, 0, -1.0, gross=2.0, fees=2.0, slip=1.0)
        rep = self.calc.report()
        self.assertGreater(rep.account["gross_pnl"], 0)
        self.assertLess(rep.account["net_pnl"], 0)
        self.assertEqual(rep.trading["losses"], 1)


class TestDrawdownAndRisk(unittest.TestCase):
    def setUp(self):
        self.repo, _ = temp_repo()
        self.repo.ensure_account(STRATEGY, 10_000.0)
        self.calc = PerformanceCalculator(self.repo)

    def test_current_drawdown_from_peak(self):
        self.repo.update_account(STRATEGY, peak_equity=12_000.0, cash=9_600.0)
        rep = self.calc.report()
        # (12000 - 9600) / 12000 = 20%
        self.assertAlmostEqual(rep.risk["current_drawdown_pct"], 20.0)

    def test_no_drawdown_at_a_new_high(self):
        self.repo.update_account(STRATEGY, peak_equity=10_000.0, cash=11_000.0)
        self.assertAlmostEqual(
            self.calc.report().risk["current_drawdown_pct"], 0.0)
        self.assertAlmostEqual(
            self.calc.report().risk["peak_equity"], 11_000.0)

    def test_average_risk_per_trade_comes_from_the_entry_journal(self):
        for i in range(4):
            trade(self.repo, i, 10.0, risk=50.0)
        self.assertAlmostEqual(
            self.calc.report().risk["average_risk_per_trade"], 50.0)

    def test_open_position_marks_feed_unrealized_and_exposure(self):
        from crypto_edge.models import Position
        pos = Position(id="p1", symbol="SOL/USDT", side="long", qty=10.0,
                       entry_ref_price=100.0, entry_fill_price=100.0,
                       entry_ms=T0, initial_stop=95.0, current_stop=95.0,
                       strategy="trend_breakout", strategy_version="v1",
                       entry_fee=1.0, entry_slippage=0.5,
                       highest_price=100.0, lowest_price=100.0,
                       risk_amount=50.0, candle_id="SOL/USDT|1h|1", journal={})
        self.repo.add_position(pos)
        self.repo.update_account(STRATEGY, cash=9_000.0)
        rep = self.calc.report(marks={"SOL/USDT": 110.0})
        self.assertAlmostEqual(rep.account["unrealized_pnl"], 100.0)
        self.assertAlmostEqual(rep.account["capital_deployed"], 1_100.0)
        self.assertAlmostEqual(rep.account["current_equity"], 10_100.0)
        self.assertEqual(rep.risk["open_positions"], 1)


class TestSampleSizeHonesty(unittest.TestCase):
    """Small samples must be labelled, not silently presented as fact."""

    def setUp(self):
        self.repo, _ = temp_repo()
        self.repo.ensure_account(STRATEGY, 10_000.0)
        self.calc = PerformanceCalculator(self.repo)

    def test_tiny_sample_is_flagged_unreliable(self):
        for i in range(3):
            trade(self.repo, i, 10.0)
        rep = self.calc.report()
        self.assertFalse(rep.sample["ratios_reliable"])
        self.assertIn("SAMPLE TOO SMALL", rep.sample["note"])

    def test_thresholds_are_exposed_so_the_user_knows_the_bar(self):
        rep = self.calc.report()
        self.assertEqual(rep.sample["min_trades_for_ratios"], MIN_TRADES_FOR_RATIOS)
        self.assertEqual(rep.sample["min_days_for_ratios"], MIN_DAYS_FOR_RATIOS)

    def test_advanced_ratios_carry_a_reliability_flag(self):
        for i in range(3):
            trade(self.repo, i, 10.0)
        adv = self.calc.report().advanced
        self.assertIn("reliable", adv)
        self.assertFalse(adv["reliable"])

    def test_zero_trades_does_not_crash(self):
        rep = self.calc.report()
        self.assertEqual(rep.trading["closed_trades"], 0)
        self.assertAlmostEqual(rep.trading["win_rate_pct"], 0.0)
        self.assertAlmostEqual(rep.account["current_equity"], 10_000.0)

    def test_category_breakdown_flags_thin_buckets(self):
        for i in range(3):
            trade(self.repo, i, 10.0, reason="stop_loss")
        cats = self.calc.by_category(min_sample=10)
        self.assertTrue(cats)
        for rows in cats.values():
            for r in rows:
                self.assertIn("sufficient_sample", r)
                self.assertFalse(r["sufficient_sample"])


class TestPersistence(unittest.TestCase):
    def test_report_is_identical_after_restart(self):
        repo, path = temp_repo()
        repo.ensure_account(STRATEGY, 10_000.0)
        for i, net in enumerate([100.0, -40.0, 25.0]):
            trade(repo, i, net)
        before = PerformanceCalculator(repo).report().as_dict()
        after = PerformanceCalculator(open_repo(path)).report().as_dict()
        self.assertEqual(before["trading"], after["trading"])
        self.assertEqual(before["account"], after["account"])


if __name__ == "__main__":
    unittest.main()
