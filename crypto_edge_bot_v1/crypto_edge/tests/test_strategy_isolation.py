"""Two strategies must not be able to interfere with each other.

FIVE COLLISIONS FOUND BY READING THE PRE-STAGE-1 CODE
----------------------------------------------------
Every one of these was a real line, not a hypothetical, and each fails SILENTLY
-- no exception, no log, just a strategy that mysteriously never trades:

1. `processed_candles.candle_id` was the PRIMARY KEY and `candle_id()` is
   `(symbol, timeframe, open_ms)` with no strategy in it. Whichever strategy
   claimed a bar first locked every other strategy out of it forever.
2. `positions.candle_id` carried a bare `UNIQUE`, so only one strategy could
   ever hold a position opened from a given bar.
3. `get_position_by_symbol(symbol)` ignored strategy, so A holding SOL/USD made
   B's entry abort as "position already open for symbol".
4. `account` was `CHECK (id = 1)` -- one row, one cash balance. Two strategies
   would have spent the same money, and a percentage-of-available-cash ladder
   would have silently depended on what the other strategy was holding.
5. `set_halt` was global: a daily loss limit breached by one strategy stopped
   the other one too.

The fix is scoping, not coordination. These tests hold that scoping in place.
"""
import unittest

import helpers  # noqa: F401  -- silences the engine's log handlers
from crypto_edge.execution.paper_broker import PaperBroker
from crypto_edge.models import MarketMeta
from crypto_edge.performance import PerformanceCalculator
from crypto_edge.portfolio.account import PaperAccount
from crypto_edge.timeutils import candle_id
from helpers import temp_repo

A = "trend_breakout"
B = "aggressive_momentum_v2"
META = MarketMeta("SOL/USDT", "SOL", "USDT", True, 4, 4, 0.0001, 10.0)


class TwoLedgers(unittest.TestCase):
    def setUp(self):
        self.repo, self.path = temp_repo()
        self.broker = PaperBroker(taker_fee_bps=7.5, slippage_bps=6.0,
                                  stop_slippage_bps=15.0)
        self.a = PaperAccount(self.repo, self.broker, 10_000.0, A)
        self.b = PaperAccount(self.repo, self.broker, 10_000.0, B)

    def enter(self, account, symbol="SOL/USDT", qty=10.0, ref=100.0,
              cid=None, side="long"):
        fill = self.broker.entry_fill(symbol, qty, ref,
                                      -1 if side == "short" else 1, meta=META)
        return account.open_position(
            symbol=symbol, strategy=account.strategy, strategy_version="1.0.0",
            qty=qty, ref_price=ref, fill=fill,
            initial_stop=ref * (1.05 if side == "short" else 0.95),
            risk_amount=50.0, candle_id=cid or candle_id(symbol, "1h", 1_000),
            signal_score=70.0, journal={}, side=side)


class TestIndependentLedgers(TwoLedgers):
    def test_each_strategy_starts_with_its_own_ten_thousand(self):
        self.assertAlmostEqual(self.a.cash(), 10_000.0)
        self.assertAlmostEqual(self.b.cash(), 10_000.0)
        self.assertEqual(len(self.repo.all_accounts()), 2)

    def test_one_strategy_spending_does_not_touch_the_other(self):
        before = self.b.cash()
        self.assertIsNotNone(self.enter(self.a))
        self.assertLess(self.a.cash(), 10_000.0, "A actually spent something")
        self.assertAlmostEqual(self.b.cash(), before,
                               msg="B's available cash must be untouched")

    def test_available_cash_for_the_ladder_ignores_the_other_strategy(self):
        """The ladder is a percentage of AVAILABLE CASH -- whose cash matters."""
        self.enter(self.a, qty=90.0)               # A deploys most of its ledger
        self.assertAlmostEqual(self.b.cash(), 10_000.0)
        self.assertAlmostEqual(self.b.equity({}), 10_000.0)

    def test_equity_is_reported_per_strategy(self):
        self.enter(self.a)
        marks = {"SOL/USDT": 120.0}
        self.assertGreater(self.a.equity(marks), self.b.equity(marks))
        self.assertAlmostEqual(self.b.equity(marks), 10_000.0)

    def test_a_realised_profit_credits_only_its_own_ledger(self):
        pos = self.enter(self.a)
        self.a.close_position(
            pos, self.broker.sell("SOL/USDT", pos.qty, 130.0, meta=META), "target")
        self.assertGreater(self.a.cash(), 10_000.0)
        self.assertAlmostEqual(self.b.cash(), 10_000.0)

    def test_peak_equity_ratchets_independently(self):
        self.enter(self.a)
        self.a.update_marks({"SOL/USDT": 200.0})
        self.assertGreater(float(self.a.state["peak_equity"]), 10_000.0)
        self.assertAlmostEqual(float(self.b.state["peak_equity"]), 10_000.0)

    def test_an_account_refuses_to_open_a_position_for_another_strategy(self):
        fill = self.broker.buy("SOL/USDT", 1.0, 100.0, meta=META)
        with self.assertRaises(ValueError):
            self.a.open_position(
                symbol="SOL/USDT", strategy=B, strategy_version="1",
                qty=1.0, ref_price=100.0, fill=fill, initial_stop=95.0,
                risk_amount=1.0, candle_id="x|1h|1", signal_score=1.0, journal={})


class TestCandleClaimsAreScoped(TwoLedgers):
    def test_both_strategies_can_claim_the_same_candle(self):
        cid = candle_id("SOL/USDT", "1h", 5_000)
        self.assertTrue(self.repo.mark_candle_processed(cid, A))
        self.assertTrue(self.repo.mark_candle_processed(cid, B),
                        "one strategy's claim must not lock the other out")

    def test_a_strategy_still_cannot_claim_the_same_candle_twice(self):
        cid = candle_id("SOL/USDT", "1h", 5_000)
        self.assertTrue(self.repo.mark_candle_processed(cid, A))
        self.assertFalse(self.repo.mark_candle_processed(cid, A),
                         "duplicate protection within a strategy still holds")

    def test_is_processed_is_answered_per_strategy(self):
        cid = candle_id("SOL/USDT", "1h", 5_000)
        self.repo.mark_candle_processed(cid, A)
        self.assertTrue(self.repo.is_candle_processed(cid, A))
        self.assertFalse(self.repo.is_candle_processed(cid, B))

    def test_both_strategies_can_enter_on_the_same_bar(self):
        cid = candle_id("SOL/USDT", "1h", 5_000)
        self.assertIsNotNone(self.enter(self.a, cid=cid))
        self.assertIsNotNone(self.enter(self.b, cid=cid),
                             "the bar is not one strategy's property")

    def test_one_strategy_still_cannot_enter_twice_on_one_bar(self):
        cid = candle_id("SOL/USDT", "1h", 5_000)
        self.assertIsNotNone(self.enter(self.a, cid=cid))
        self.assertIsNone(self.enter(self.a, symbol="ETH/USDT", cid=cid))


class TestPositionsAreScoped(TwoLedgers):
    def test_both_strategies_can_hold_the_same_symbol(self):
        self.assertIsNotNone(self.enter(self.a, cid="a|1h|1"))
        self.assertIsNotNone(self.enter(self.b, cid="b|1h|1"),
                             "A holding SOL must not block B from trading SOL")

    def test_they_can_even_hold_it_on_opposite_sides(self):
        self.assertIsNotNone(self.enter(self.a, cid="a|1h|1", side="long"))
        self.assertIsNotNone(self.enter(self.b, cid="b|1h|1", side="short"))
        self.assertEqual(self.a.positions()[0].direction, 1)
        self.assertEqual(self.b.positions()[0].direction, -1)

    def test_a_strategy_still_cannot_double_up_on_one_symbol(self):
        self.assertIsNotNone(self.enter(self.a, cid="a|1h|1"))
        self.assertIsNone(self.enter(self.a, cid="a|1h|2"),
                          "duplicate-entry protection within a strategy holds")

    def test_each_strategy_sees_only_its_own_positions(self):
        self.enter(self.a, cid="a|1h|1")
        self.enter(self.b, cid="b|1h|1")
        self.assertEqual(len(self.a.positions()), 1)
        self.assertEqual(len(self.b.positions()), 1)
        self.assertEqual(self.a.positions()[0].strategy, A)
        self.assertEqual(self.b.positions()[0].strategy, B)

    def test_the_whole_portfolio_view_still_sees_everything(self):
        self.enter(self.a, cid="a|1h|1")
        self.enter(self.b, cid="b|1h|1")
        self.assertEqual(len(self.repo.get_positions()), 2)

    def test_closing_one_leaves_the_other_open(self):
        pa = self.enter(self.a, cid="a|1h|1")
        self.enter(self.b, cid="b|1h|1")
        self.a.close_position(
            pa, self.broker.sell("SOL/USDT", pa.qty, 105.0, meta=META), "target")
        self.assertEqual(len(self.a.positions()), 0)
        self.assertEqual(len(self.b.positions()), 1)


class TestHaltsAreScoped(TwoLedgers):
    def test_halting_one_strategy_leaves_the_other_trading(self):
        self.repo.set_halt(A, True, "daily loss limit")
        self.assertTrue(int(self.repo.get_account(A)["halted"]))
        self.assertFalse(int(self.repo.get_account(B)["halted"]),
                         "a daily loss belongs to the account that lost it")

    def test_the_halt_reason_belongs_to_the_halted_strategy(self):
        self.repo.set_halt(B, True, "B specific")
        self.assertEqual(self.repo.get_account(A)["halt_reason"], "")
        self.assertEqual(self.repo.get_account(B)["halt_reason"], "B specific")

    def test_clearing_one_halt_does_not_clear_the_other(self):
        self.repo.set_halt(A, True, "x")
        self.repo.set_halt(B, True, "y")
        self.repo.set_halt(A, False, "")
        self.assertFalse(int(self.repo.get_account(A)["halted"]))
        self.assertTrue(int(self.repo.get_account(B)["halted"]))


class TestReportingIsAttributable(TwoLedgers):
    def _round_trip(self, account, exit_px, cid):
        pos = self.enter(account, cid=cid)
        return account.close_position(
            pos, self.broker.sell("SOL/USDT", pos.qty, exit_px, meta=META), "target")

    def test_each_strategy_reports_only_its_own_trades(self):
        self._round_trip(self.a, 130.0, "a|1h|1")
        self._round_trip(self.b, 70.0, "b|1h|1")
        ra = PerformanceCalculator(self.repo, A).report().as_dict()
        rb = PerformanceCalculator(self.repo, B).report().as_dict()
        self.assertEqual(ra["trading"]["closed_trades"], 1)
        self.assertEqual(rb["trading"]["closed_trades"], 1)
        self.assertGreater(ra["account"]["realized_pnl"], 0)
        self.assertLess(rb["account"]["realized_pnl"], 0)

    def test_a_winner_and_a_loser_do_not_average_into_each_other(self):
        self._round_trip(self.a, 130.0, "a|1h|1")
        self._round_trip(self.b, 70.0, "b|1h|1")
        self.assertEqual(
            PerformanceCalculator(self.repo, A).report().trading["win_rate_pct"], 100.0)
        self.assertEqual(
            PerformanceCalculator(self.repo, B).report().trading["win_rate_pct"], 0.0)

    def test_an_ambiguous_report_refuses_rather_than_picking_one(self):
        """With two ledgers, "the account" is not a well-formed question."""
        with self.assertRaises(ValueError):
            PerformanceCalculator(self.repo).report()

    def test_a_single_ledger_still_needs_no_ceremony(self):
        repo, _ = temp_repo()
        PaperAccount(repo, self.broker, 10_000.0, A)
        rep = PerformanceCalculator(repo).report().as_dict()
        self.assertAlmostEqual(rep["account"]["cash"], 10_000.0)


class TestRestartKeepsBothLedgers(TwoLedgers):
    def test_both_sub_accounts_and_positions_survive_a_reopen(self):
        from helpers import open_repo
        self.enter(self.a, cid="a|1h|1")
        self.enter(self.b, cid="b|1h|1", side="short")
        cash_a, cash_b = self.a.cash(), self.b.cash()
        self.repo.conn.close()

        repo2 = open_repo(self.path)
        self.assertAlmostEqual(repo2.get_account(A)["cash"], cash_a)
        self.assertAlmostEqual(repo2.get_account(B)["cash"], cash_b)
        self.assertEqual(len(repo2.get_positions(A)), 1)
        self.assertEqual(len(repo2.get_positions(B)), 1)
        self.assertEqual(repo2.get_positions(B)[0].side, "short")
        self.assertTrue(repo2.get_positions(B)[0].is_short)

    def test_collateral_survives_the_round_trip_to_disk(self):
        from helpers import open_repo
        pos = self.enter(self.a, cid="a|1h|1")
        margin = pos.margin_held
        self.repo.conn.close()
        repo2 = open_repo(self.path)
        self.assertAlmostEqual(repo2.get_positions(A)[0].margin_held, margin,
                               places=9)


if __name__ == "__main__":
    unittest.main()
