"""Persistent account: cash invariants, restart recovery, duplicate protection."""
import unittest

from crypto_edge.execution.paper_broker import PaperBroker
from crypto_edge.models import Candle
from crypto_edge.portfolio.account import PaperAccount
from helpers import default_meta, open_repo, temp_repo


class AccountTestBase(unittest.TestCase):
    def setUp(self):
        self.repo, self.path = temp_repo()
        self.broker = PaperBroker(10.0, 5.0, 10.0, use_book_spread=False)
        self.acct = PaperAccount(self.repo, self.broker, 10_000.0)
        self.meta = default_meta()

    def _open(self, symbol="SOL/USDT", price=100.0, qty=10.0, stop=90.0,
              candle_id="SOL/USDT|1h|1000", ts=1_000_000):
        fill = self.broker.buy(symbol, qty, price, None, self.meta, ts_ms=ts)
        return self.acct.open_position(
            symbol=symbol, strategy="trend_breakout", strategy_version="1.0.0",
            qty=qty, ref_price=price, fill=fill, initial_stop=stop,
            risk_amount=qty * (price - stop), candle_id=candle_id,
            signal_score=70.0, journal={"adx": 25.0}), fill


class TestCashInvariants(AccountTestBase):
    def test_entry_debits_cash_by_notional_plus_fee(self):
        pos, fill = self._open()
        self.assertIsNotNone(pos)
        expected = 10_000.0 - (fill.notional + fill.fee)
        self.assertAlmostEqual(self.acct.cash(), expected, places=9)

    def test_round_trip_cash_change_equals_net_pnl(self):
        """The single most important invariant: the ledger and the account
        must agree to the cent."""
        start_cash = self.acct.cash()
        pos, _ = self._open()
        exit_fill = self.broker.sell("SOL/USDT", pos.qty, 120.0, None, self.meta,
                                     ts_ms=2_000_000)
        trade = self.acct.close_position(pos, exit_fill, "take_profit", {})
        self.assertIsNotNone(trade)
        self.assertAlmostEqual(self.acct.cash() - start_cash, trade.net_pnl, places=8)

    def test_losing_trade_also_reconciles(self):
        start_cash = self.acct.cash()
        pos, _ = self._open()
        exit_fill = self.broker.sell("SOL/USDT", pos.qty, 85.0, None, self.meta,
                                     ts_ms=2_000_000)
        trade = self.acct.close_position(pos, exit_fill, "stop_loss", {})
        self.assertLess(trade.net_pnl, 0)
        self.assertAlmostEqual(self.acct.cash() - start_cash, trade.net_pnl, places=8)

    def test_equity_equals_cash_plus_marked_positions(self):
        pos, _ = self._open()
        marks = {"SOL/USDT": 110.0}
        self.assertAlmostEqual(self.acct.equity(marks),
                               self.acct.cash() + pos.qty * 110.0, places=9)

    def test_unrealized_uses_fill_price_as_basis(self):
        pos, fill = self._open()
        marks = {"SOL/USDT": 110.0}
        self.assertAlmostEqual(self.acct.unrealized(marks),
                               (110.0 - fill.fill_price) * pos.qty, places=9)

    def test_realized_pnl_accumulates_on_account(self):
        pos, _ = self._open()
        f = self.broker.sell("SOL/USDT", pos.qty, 120.0, None, self.meta, ts_ms=2_000_000)
        t = self.acct.close_position(pos, f, "take_profit", {})
        self.assertAlmostEqual(float(self.acct.state["realized_pnl"]), t.net_pnl, places=8)


class TestDuplicateProtection(AccountTestBase):
    def test_same_candle_cannot_enter_twice(self):
        p1, _ = self._open(candle_id="SOL/USDT|1h|1000")
        self.assertIsNotNone(p1)
        cash_after_first = self.acct.cash()
        p2, _ = self._open(candle_id="SOL/USDT|1h|1000")
        self.assertIsNone(p2, "second entry on the same candle must be refused")
        self.assertAlmostEqual(self.acct.cash(), cash_after_first, places=9)
        self.assertEqual(len(self.acct.positions()), 1)

    def test_second_position_in_same_symbol_refused(self):
        self._open(candle_id="SOL/USDT|1h|1000")
        p2, _ = self._open(candle_id="SOL/USDT|1h|2000")   # different candle
        self.assertIsNone(p2)
        self.assertEqual(len(self.acct.positions()), 1)

    def test_double_close_is_refused(self):
        pos, _ = self._open()
        f = self.broker.sell("SOL/USDT", pos.qty, 120.0, None, self.meta, ts_ms=2_000_000)
        t1 = self.acct.close_position(pos, f, "take_profit", {})
        cash_after = self.acct.cash()
        t2 = self.acct.close_position(pos, f, "take_profit", {})
        self.assertIsNotNone(t1)
        self.assertIsNone(t2, "closing an already-closed position must be refused")
        self.assertAlmostEqual(self.acct.cash(), cash_after, places=9)
        self.assertEqual(len(self.repo.get_trades()), 1)

    def test_candle_claim_is_atomic(self):
        self.assertTrue(self.repo.mark_candle_processed("X|1h|1"))
        self.assertFalse(self.repo.mark_candle_processed("X|1h|1"))

    def test_insufficient_cash_refused_and_candle_not_consumed_wrongly(self):
        pos, _ = self._open(qty=95.0)          # ~$9,500 of $10,000
        self.assertIsNotNone(pos)
        p2, _ = self._open(symbol="ETH/USDT", qty=95.0, candle_id="ETH/USDT|1h|1")
        self.assertIsNone(p2)
        self.assertGreaterEqual(self.acct.cash(), 0.0)


class TestRestartPersistence(AccountTestBase):
    def test_full_state_survives_restart(self):
        pos, fill = self._open()
        self.acct.update_marks({"SOL/USDT": 115.0})
        cash_before = self.acct.cash()
        equity_before = self.acct.equity({"SOL/USDT": 115.0})

        # simulate process death and cold start from the same file
        self.repo.conn.close()
        repo2 = open_repo(self.path)
        acct2 = PaperAccount(repo2, self.broker, 10_000.0)

        self.assertAlmostEqual(acct2.cash(), cash_before, places=9)
        self.assertAlmostEqual(acct2.equity({"SOL/USDT": 115.0}), equity_before, places=9)
        positions = acct2.positions()
        self.assertEqual(len(positions), 1)
        p = positions[0]
        self.assertEqual(p.symbol, "SOL/USDT")
        self.assertAlmostEqual(p.qty, pos.qty, places=9)
        self.assertAlmostEqual(p.entry_fill_price, fill.fill_price, places=9)
        self.assertAlmostEqual(p.current_stop, 90.0, places=9)
        self.assertEqual(p.candle_id, "SOL/USDT|1h|1000")
        self.assertEqual(p.journal.get("adx"), 25.0)

    def test_duplicate_guard_survives_restart(self):
        self._open(candle_id="SOL/USDT|1h|1000")
        self.repo.conn.close()
        repo2 = open_repo(self.path)
        self.assertTrue(repo2.is_candle_processed("SOL/USDT|1h|1000"))
        self.assertFalse(repo2.mark_candle_processed("SOL/USDT|1h|1000"))

    def test_closed_trades_survive_restart(self):
        pos, _ = self._open()
        f = self.broker.sell("SOL/USDT", pos.qty, 120.0, None, self.meta, ts_ms=2_000_000)
        t = self.acct.close_position(pos, f, "take_profit", {})
        self.repo.conn.close()
        repo2 = open_repo(self.path)
        trades = repo2.get_trades()
        self.assertEqual(len(trades), 1)
        self.assertAlmostEqual(trades[0]["net_pnl"], t.net_pnl, places=9)
        self.assertEqual(trades[0]["exit_reason"], "take_profit")

    def test_history_is_never_rewritten(self):
        """A losing trade must remain in the ledger permanently."""
        pos, _ = self._open()
        f = self.broker.sell("SOL/USDT", pos.qty, 80.0, None, self.meta, ts_ms=2_000_000)
        self.acct.close_position(pos, f, "stop_loss", {})
        p2, _ = self._open(candle_id="SOL/USDT|1h|3000", ts=3_000_000)
        f2 = self.broker.sell("SOL/USDT", p2.qty, 130.0, None, self.meta, ts_ms=4_000_000)
        self.acct.close_position(p2, f2, "take_profit", {})
        trades = self.repo.get_trades()
        self.assertEqual(len(trades), 2)
        self.assertLess(trades[0]["net_pnl"], 0)
        self.assertGreater(trades[1]["net_pnl"], 0)


class TestExcursions(AccountTestBase):
    def test_mfe_and_mae_tracked(self):
        pos, _ = self._open()
        self.acct.update_marks({"SOL/USDT": 120.0})
        self.acct.update_marks({"SOL/USDT": 85.0})
        p = self.acct.positions()[0]
        self.assertGreater(p.mfe, 0)
        self.assertLess(p.mae, 0)
        self.assertAlmostEqual(p.highest_price, 120.0, places=9)
        self.assertAlmostEqual(p.lowest_price, 85.0, places=9)

    def test_peak_equity_ratchets_up_only(self):
        self._open()
        self.acct.update_marks({"SOL/USDT": 200.0})
        peak = float(self.acct.state["peak_equity"])
        self.acct.update_marks({"SOL/USDT": 50.0})
        self.assertAlmostEqual(float(self.acct.state["peak_equity"]), peak, places=9)


if __name__ == "__main__":
    unittest.main()
