"""Circuit breakers, exposure limits, correlation, and stop management."""
import unittest

import numpy as np

from crypto_edge.config import RiskCfg, StrategyCfg
from crypto_edge.models import Position
from crypto_edge.portfolio.risk import RiskManager
from crypto_edge.portfolio.stops import (breakeven_stop, chandelier_stop,
                                         initial_stop, time_stop_hit, update_stop)


def make_pos(entry=100.0, stop=90.0, current=None, highest=None, qty=10.0,
             entry_ms=0) -> Position:
    return Position(id="p", symbol="S/USDT", strategy="s", strategy_version="1",
                    side="long", qty=qty, entry_ref_price=entry,
                    entry_fill_price=entry, entry_ms=entry_ms, entry_fee=0.0,
                    entry_slippage=0.0, initial_stop=stop,
                    current_stop=current if current is not None else stop,
                    highest_price=highest if highest is not None else entry,
                    lowest_price=entry, risk_amount=qty * (entry - stop),
                    candle_id="c")


class TestCircuitBreakers(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager(RiskCfg(daily_loss_limit_pct=3.0, max_drawdown_pct=20.0))

    def test_no_halt_when_healthy(self):
        self.assertFalse(self.rm.check_circuit_breakers(10_000, 10_000, 10_000).halted)

    def test_daily_loss_limit_trips_at_threshold(self):
        cs = self.rm.check_circuit_breakers(9_700, 10_000, 10_000)
        self.assertTrue(cs.halted)
        self.assertIn("DAILY LOSS", cs.reason)

    def test_daily_loss_just_inside_limit_does_not_trip(self):
        self.assertFalse(self.rm.check_circuit_breakers(9_701, 10_000, 10_000).halted)

    def test_max_drawdown_kill_switch(self):
        # 20% below peak; daily start set equal so only drawdown can trip
        cs = self.rm.check_circuit_breakers(8_000, 10_000, 8_000)
        self.assertTrue(cs.halted)
        self.assertIn("DRAWDOWN", cs.reason)

    def test_drawdown_takes_priority_in_reason(self):
        cs = self.rm.check_circuit_breakers(7_000, 10_000, 10_000)
        self.assertTrue(cs.halted)
        self.assertIn("DRAWDOWN", cs.reason)

    def test_daily_loss_remaining_shrinks(self):
        self.assertAlmostEqual(self.rm.daily_loss_remaining(10_000, 10_000), 300.0, places=6)
        self.assertAlmostEqual(self.rm.daily_loss_remaining(9_850, 10_000), 150.0, places=6)
        self.assertAlmostEqual(self.rm.daily_loss_remaining(9_000, 10_000), 0.0, places=6)


class TestEntryGate(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager(RiskCfg(max_open_positions=3, max_portfolio_exposure_pct=60.0,
                                      max_correlation=0.85, max_new_entries_per_cycle=2))

    def test_allows_clean_entry(self):
        d = self.rm.check_entry(symbol="A/USDT", equity=10_000, exposure=0,
                                n_open=0, open_symbols=[])
        self.assertTrue(d.allowed)

    def test_blocks_duplicate_symbol(self):
        d = self.rm.check_entry(symbol="A/USDT", equity=10_000, exposure=0,
                                n_open=1, open_symbols=["A/USDT"])
        self.assertFalse(d.allowed)

    def test_blocks_at_max_positions(self):
        d = self.rm.check_entry(symbol="D/USDT", equity=10_000, exposure=0,
                                n_open=3, open_symbols=["A/USDT", "B/USDT", "C/USDT"])
        self.assertFalse(d.allowed)
        self.assertIn("max open positions", d.reason)

    def test_blocks_at_max_exposure(self):
        d = self.rm.check_entry(symbol="D/USDT", equity=10_000, exposure=6_000,
                                n_open=1, open_symbols=["A/USDT"])
        self.assertFalse(d.allowed)
        self.assertIn("exposure", d.reason)

    def test_blocks_highly_correlated_candidate(self):
        rng = np.random.default_rng(1)
        base = rng.normal(0, 0.01, 200)
        candidate = base + rng.normal(0, 0.0005, 200)   # nearly identical
        d = self.rm.check_entry(symbol="B/USDT", equity=10_000, exposure=0,
                                n_open=1, open_symbols=["A/USDT"],
                                candidate_returns=candidate,
                                open_returns={"A/USDT": base})
        self.assertFalse(d.allowed)
        self.assertIn("correlation", d.reason)
        self.assertIn("A/USDT", d.reason)

    def test_allows_uncorrelated_candidate(self):
        rng = np.random.default_rng(2)
        d = self.rm.check_entry(symbol="B/USDT", equity=10_000, exposure=0,
                                n_open=1, open_symbols=["A/USDT"],
                                candidate_returns=rng.normal(0, 0.01, 200),
                                open_returns={"A/USDT": rng.normal(0, 0.01, 200)})
        self.assertTrue(d.allowed, d.reason)

    def test_per_cycle_entry_cap(self):
        d = self.rm.check_entry(symbol="A/USDT", equity=10_000, exposure=0,
                                n_open=0, open_symbols=[], entries_this_cycle=2)
        self.assertFalse(d.allowed)


class TestStops(unittest.TestCase):
    def setUp(self):
        self.cfg = StrategyCfg(stop_atr_mult=2.0, trail_atr_mult=3.0,
                               breakeven_at_r=1.0, breakeven_offset_r=0.1,
                               time_stop_bars=10, time_stop_min_r=0.5)

    def test_initial_stop_is_atr_multiple_below_entry(self):
        self.assertAlmostEqual(initial_stop(100.0, 2.5, 2.0), 95.0, places=9)

    def test_chandelier_trails_from_the_high(self):
        self.assertAlmostEqual(chandelier_stop(120.0, 2.0, 3.0), 114.0, places=9)

    def test_stop_never_moves_down(self):
        pos = make_pos(entry=100.0, stop=90.0, current=95.0, highest=110.0)
        upd = update_stop(pos, 96.0, atr=5.0, cfg=self.cfg)   # chandelier = 95
        self.assertFalse(upd.changed)
        self.assertAlmostEqual(upd.new_stop, 95.0, places=9)

    def test_stop_ratchets_up_with_new_high(self):
        pos = make_pos(entry=100.0, stop=90.0, current=90.0, highest=100.0)
        upd = update_stop(pos, 130.0, atr=2.0, cfg=self.cfg)
        self.assertTrue(upd.changed)
        self.assertAlmostEqual(upd.new_stop, 124.0, places=9)  # 130 - 3*2
        self.assertEqual(upd.kind, "chandelier")

    def test_breakeven_ratchet_engages_at_1r(self):
        # R = 10; at price 110 we are +1R, so stop -> 100 + 0.1*10 = 101
        pos = make_pos(entry=100.0, stop=90.0, current=90.0, highest=110.0)
        upd = update_stop(pos, 110.0, atr=0.0, cfg=self.cfg)
        self.assertTrue(upd.changed)
        self.assertAlmostEqual(upd.new_stop, 101.0, places=9)
        self.assertEqual(upd.kind, "breakeven")

    def test_breakeven_not_engaged_below_1r(self):
        pos = make_pos(entry=100.0, stop=90.0, current=90.0, highest=105.0)
        upd = update_stop(pos, 105.0, atr=0.0, cfg=self.cfg)
        self.assertFalse(upd.changed)

    def test_breakeven_formula(self):
        self.assertAlmostEqual(breakeven_stop(100.0, 90.0, 0.1), 101.0, places=9)

    def test_pct_move_reported_for_notification_filter(self):
        pos = make_pos(entry=100.0, stop=90.0, current=100.0, highest=100.0)
        upd = update_stop(pos, 140.0, atr=2.0, cfg=self.cfg)
        self.assertTrue(upd.changed)
        self.assertGreater(upd.pct_move, 0.0)

    def test_time_stop_fires_on_stalled_trade(self):
        pos = make_pos(entry=100.0, stop=90.0, entry_ms=0)
        # 10 bars later, price only +2 (0.2R < 0.5R threshold)
        self.assertTrue(time_stop_hit(pos, 10 * 3_600_000, 102.0, self.cfg, 3_600_000))

    def test_time_stop_does_not_fire_on_working_trade(self):
        pos = make_pos(entry=100.0, stop=90.0, entry_ms=0)
        self.assertFalse(time_stop_hit(pos, 10 * 3_600_000, 112.0, self.cfg, 3_600_000))

    def test_time_stop_does_not_fire_early(self):
        pos = make_pos(entry=100.0, stop=90.0, entry_ms=0)
        self.assertFalse(time_stop_hit(pos, 5 * 3_600_000, 100.0, self.cfg, 3_600_000))


if __name__ == "__main__":
    unittest.main()
