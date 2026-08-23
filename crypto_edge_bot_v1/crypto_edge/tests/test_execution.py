"""Position sizing, fees, slippage, stop fills, and the P&L identity."""
import math
import unittest

from crypto_edge.execution.paper_broker import (PaperBroker, realise_pnl,
                                                round_amount)
from crypto_edge.models import Candle, Quote
from helpers import default_meta


class TestSizing(unittest.TestCase):
    def setUp(self):
        self.b = PaperBroker(taker_fee_bps=10.0, slippage_bps=0.0,
                             stop_slippage_bps=0.0, use_book_spread=False)
        self.meta = default_meta()

    def test_risk_based_size_is_exact(self):
        # $10,000 equity, 1% risk = $100. Stop $10 below entry -> 10 units.
        r = self.b.size_position(equity=10_000, cash=10_000, entry_price=100.0,
                                 stop_price=90.0, risk_pct=1.0,
                                 max_position_pct=100.0, current_exposure=0.0,
                                 max_exposure_pct=100.0, meta=self.meta)
        self.assertTrue(r.ok, r.reason)
        self.assertAlmostEqual(r.qty, 10.0, places=6)
        self.assertAlmostEqual(r.risk_amount, 100.0, places=6)
        self.assertAlmostEqual(r.stop_distance, 10.0, places=9)

    def test_risk_scales_with_stop_distance(self):
        """Halving the stop distance must double the size, exactly."""
        wide = self.b.size_position(equity=10_000, cash=10_000, entry_price=100.0,
                                    stop_price=90.0, risk_pct=1.0,
                                    max_position_pct=100.0, current_exposure=0.0,
                                    max_exposure_pct=100.0, meta=self.meta)
        tight = self.b.size_position(equity=10_000, cash=10_000, entry_price=100.0,
                                     stop_price=95.0, risk_pct=1.0,
                                     max_position_pct=100.0, current_exposure=0.0,
                                     max_exposure_pct=100.0, meta=self.meta)
        self.assertAlmostEqual(tight.qty, wide.qty * 2, places=6)
        self.assertAlmostEqual(tight.risk_amount, wide.risk_amount, places=6)

    def test_max_position_pct_caps_size(self):
        r = self.b.size_position(equity=10_000, cash=10_000, entry_price=100.0,
                                 stop_price=99.0, risk_pct=1.0,
                                 max_position_pct=10.0, current_exposure=0.0,
                                 max_exposure_pct=100.0, meta=self.meta)
        self.assertTrue(r.ok)
        self.assertLessEqual(r.notional, 1_000.0 + 1e-9)
        # risk is now BELOW budget because allocation capped us -- never above
        self.assertLess(r.risk_amount, 100.0)

    def test_portfolio_exposure_cap(self):
        r = self.b.size_position(equity=10_000, cash=10_000, entry_price=100.0,
                                 stop_price=90.0, risk_pct=1.0,
                                 max_position_pct=100.0, current_exposure=5_900.0,
                                 max_exposure_pct=60.0, meta=self.meta)
        self.assertTrue(r.ok)
        self.assertLessEqual(r.notional, 100.0 + 1e-9)

    def test_exposure_full_rejects(self):
        r = self.b.size_position(equity=10_000, cash=10_000, entry_price=100.0,
                                 stop_price=90.0, risk_pct=1.0,
                                 max_position_pct=100.0, current_exposure=6_000.0,
                                 max_exposure_pct=60.0, meta=self.meta)
        self.assertFalse(r.ok)
        self.assertIn("exposure", r.reason)

    def test_cash_cap_includes_fee(self):
        """Size must never require more cash than we have, fee included."""
        r = self.b.size_position(equity=10_000, cash=500.0, entry_price=100.0,
                                 stop_price=90.0, risk_pct=1.0,
                                 max_position_pct=100.0, current_exposure=0.0,
                                 max_exposure_pct=100.0, meta=self.meta)
        self.assertTrue(r.ok)
        cost = r.qty * 100.0 * (1 + 10.0 / 10_000)
        self.assertLessEqual(cost, 500.0 + 1e-9)

    def test_min_notional_rejected(self):
        r = self.b.size_position(equity=100, cash=100, entry_price=100.0,
                                 stop_price=90.0, risk_pct=0.5,
                                 max_position_pct=100.0, current_exposure=0.0,
                                 max_exposure_pct=100.0, meta=self.meta)
        self.assertFalse(r.ok)
        self.assertIn("min notional", r.reason)

    def test_stop_above_entry_rejected(self):
        r = self.b.size_position(equity=10_000, cash=10_000, entry_price=100.0,
                                 stop_price=110.0, risk_pct=1.0,
                                 max_position_pct=100.0, current_exposure=0.0,
                                 max_exposure_pct=100.0, meta=self.meta)
        self.assertFalse(r.ok)

    def test_too_tight_stop_rejected(self):
        r = self.b.size_position(equity=10_000, cash=10_000, entry_price=100.0,
                                 stop_price=99.95, risk_pct=1.0,
                                 max_position_pct=100.0, current_exposure=0.0,
                                 max_exposure_pct=100.0, meta=self.meta,
                                 min_stop_distance_pct=0.3)
        self.assertFalse(r.ok)
        self.assertIn("too tight", r.reason)

    def test_rounding_is_always_down(self):
        self.assertEqual(round_amount(1.239999, 2), 1.23)
        self.assertEqual(round_amount(1.999, 0), 1.0)


class TestFeesAndSlippage(unittest.TestCase):
    def test_fee_is_bps_of_notional(self):
        b = PaperBroker(7.5, 0.0, 0.0, use_book_spread=False)
        self.assertAlmostEqual(b.fee(10_000.0), 7.50, places=9)

    def test_buy_slippage_moves_price_against_us(self):
        b = PaperBroker(0.0, 10.0, 0.0, use_book_spread=False)
        f = b.buy("X/USDT", 1.0, 100.0, ts_ms=0)
        self.assertAlmostEqual(f.fill_price, 100.10, places=9)
        self.assertAlmostEqual(f.slippage_cost, 0.10, places=9)

    def test_sell_slippage_moves_price_against_us(self):
        b = PaperBroker(0.0, 10.0, 0.0, use_book_spread=False)
        f = b.sell("X/USDT", 1.0, 100.0, ts_ms=0)
        self.assertAlmostEqual(f.fill_price, 99.90, places=9)
        self.assertAlmostEqual(f.slippage_cost, 0.10, places=9)

    def test_book_spread_is_crossed_when_available(self):
        b = PaperBroker(0.0, 0.0, 0.0, use_book_spread=True)
        q = Quote("X/USDT", bid=99.0, ask=101.0, last=100.0, ts_ms=0)
        buy = b.buy("X/USDT", 1.0, 100.0, quote=q, ts_ms=0)
        sell = b.sell("X/USDT", 1.0, 100.0, quote=q, ts_ms=0)
        self.assertAlmostEqual(buy.fill_price, 101.0, places=9)   # pay the ask
        self.assertAlmostEqual(sell.fill_price, 99.0, places=9)   # hit the bid
        self.assertAlmostEqual(buy.slippage_cost, 1.0, places=9)

    def test_wide_spread_is_rejected(self):
        b = PaperBroker(0.0, 0.0, 0.0, max_spread_bps_entry=25.0)
        wide = Quote("X/USDT", 99.0, 101.0, 100.0, 0)     # 200 bps
        ok, bps = b.spread_acceptable(wide)
        self.assertFalse(ok)
        self.assertAlmostEqual(bps, 200.0, places=6)


class TestStopExecution(unittest.TestCase):
    def setUp(self):
        self.b = PaperBroker(0.0, 0.0, stop_slippage_bps=10.0, use_book_spread=False)

    def test_no_fill_when_stop_untouched(self):
        c = Candle(0, 100.0, 102.0, 96.0, 101.0, 10.0)
        self.assertIsNone(self.b.stop_exit("X/USDT", 1.0, 95.0, c))

    def test_fill_at_stop_with_slippage(self):
        c = Candle(0, 100.0, 101.0, 94.0, 96.0, 10.0)
        f = self.b.stop_exit("X/USDT", 1.0, 95.0, c)
        self.assertIsNotNone(f)
        self.assertEqual(f.reason, "stop")
        self.assertAlmostEqual(f.fill_price, 95.0 * (1 - 0.001), places=9)

    def test_gap_through_stop_fills_at_open_not_at_stop(self):
        """The critical honesty test: a market that gaps down must NOT be
        allowed to fill at the stop price."""
        c = Candle(0, 88.0, 89.0, 85.0, 86.0, 10.0)
        f = self.b.stop_exit("X/USDT", 1.0, 95.0, c)
        self.assertIsNotNone(f)
        self.assertEqual(f.reason, "stop_gap")
        self.assertAlmostEqual(f.fill_price, 88.0 * (1 - 0.001), places=9)
        self.assertLess(f.fill_price, 95.0)
        # the loss beyond the stop is recorded as slippage, not hidden
        self.assertGreater(f.slippage_cost, 6.0)


class TestPnLIdentity(unittest.TestCase):
    def test_decomposition_matches_fill_arithmetic(self):
        """net = gross - slippage - fees MUST equal
           (exit_fill - entry_fill) * qty - fees, or slippage is double-counted."""
        cases = [
            (100.0, 100.1, 110.0, 109.8, 5.0, 0.5, 0.55),
            (50.0, 50.05, 45.0, 44.9, 12.0, 0.3, 0.27),
            (7.5, 7.51, 7.4, 7.39, 250.0, 1.9, 1.85),
        ]
        for eref, efill, xref, xfill, qty, efee, xfee in cases:
            with self.subTest(entry=eref):
                p = realise_pnl(eref, efill, xref, xfill, qty, efee, xfee)
                direct = (xfill - efill) * qty - (efee + xfee)
                self.assertAlmostEqual(p["net_pnl"], direct, places=9)

    def test_costs_make_a_flat_trade_a_loss(self):
        p = realise_pnl(100.0, 100.05, 100.0, 99.95, 10.0, 0.5, 0.5)
        self.assertAlmostEqual(p["gross_pnl"], 0.0, places=9)
        self.assertGreater(p["slippage_cost"], 0)
        self.assertLess(p["net_pnl"], 0)

    def test_winning_trade_components(self):
        p = realise_pnl(100.0, 100.0, 110.0, 110.0, 10.0, 1.0, 1.1)
        self.assertAlmostEqual(p["gross_pnl"], 100.0, places=9)
        self.assertAlmostEqual(p["slippage_cost"], 0.0, places=9)
        self.assertAlmostEqual(p["fees"], 2.1, places=9)
        self.assertAlmostEqual(p["net_pnl"], 97.9, places=9)


if __name__ == "__main__":
    unittest.main()
