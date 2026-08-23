"""Look-ahead prevention: closed candles, causal indicators, no future leakage.

These are the tests that make a backtest worth anything.
"""
import unittest

import numpy as np

from crypto_edge import indicators as ind
from crypto_edge.data.fixture_feed import make_series
from crypto_edge.strategy.base import MarketContext
from crypto_edge.strategy.trend_breakout import TrendBreakoutStrategy
from crypto_edge.timeutils import (candle_id, is_closed, last_closed_open_ms,
                                   floor_to_tf, tf_ms)
from helpers import ALIGNED, test_config, uptrend_closes

HOUR = 3_600_000


class TestCandleClosure(unittest.TestCase):
    def test_candle_is_not_closed_before_its_end(self):
        self.assertFalse(is_closed(ALIGNED, "1h", ALIGNED + HOUR - 1))

    def test_candle_is_closed_at_its_end(self):
        self.assertTrue(is_closed(ALIGNED, "1h", ALIGNED + HOUR))

    def test_buffer_delays_closure(self):
        self.assertFalse(is_closed(ALIGNED, "1h", ALIGNED + HOUR + 10_000,
                                   buffer_ms=20_000))
        self.assertTrue(is_closed(ALIGNED, "1h", ALIGNED + HOUR + 30_000,
                                  buffer_ms=20_000))

    def test_last_closed_open_ms_is_previous_bar(self):
        now = ALIGNED + HOUR + 60_000        # 1 minute into the next bar
        self.assertEqual(last_closed_open_ms("1h", now), ALIGNED)

    def test_drop_unclosed_removes_the_forming_bar(self):
        closes = np.arange(100, 110, dtype=float)
        s = make_series("X/USDT", "1h", closes, ALIGNED)
        # "now" is halfway through the final bar
        now = int(s.open_ms[-1]) + HOUR // 2
        trimmed = s.drop_unclosed(now)
        self.assertEqual(len(trimmed), len(s) - 1)
        self.assertEqual(int(trimmed.open_ms[-1]), int(s.open_ms[-2]))

    def test_drop_unclosed_keeps_all_when_all_closed(self):
        s = make_series("X/USDT", "1h", np.arange(100, 110, dtype=float), ALIGNED)
        now = int(s.open_ms[-1]) + HOUR + 1
        self.assertEqual(len(s.drop_unclosed(now)), len(s))

    def test_drop_unclosed_can_empty_a_series(self):
        s = make_series("X/USDT", "1h", [100.0], ALIGNED)
        self.assertEqual(len(s.drop_unclosed(ALIGNED + 5)), 0)

    def test_candle_id_is_deterministic_and_unique(self):
        a = candle_id("SOL/USDT", "1h", ALIGNED)
        b = candle_id("SOL/USDT", "1h", ALIGNED)
        c = candle_id("SOL/USDT", "1h", ALIGNED + HOUR)
        d = candle_id("ETH/USDT", "1h", ALIGNED)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)


class TestIndicatorCausality(unittest.TestCase):
    """An indicator value at bar i must not change when future bars are added."""

    def setUp(self):
        self.closes = uptrend_closes(300, seed=11)
        self.high = self.closes * 1.01
        self.low = self.closes * 0.99
        self.vol = np.full(300, 1000.0)

    def _assert_causal(self, fn, cut=200):
        full = fn(200)
        partial = fn(200, upto=cut)
        self.assertTrue(np.isfinite(full) and np.isfinite(partial))
        self.assertAlmostEqual(full, partial, places=8)

    def test_ema_is_causal(self):
        full = ind.ema(self.closes, 50)[200]
        partial = ind.ema(self.closes[:201], 50)[200]
        self.assertAlmostEqual(full, partial, places=9)

    def test_atr_is_causal(self):
        full = ind.atr(self.high, self.low, self.closes, 14)[200]
        partial = ind.atr(self.high[:201], self.low[:201], self.closes[:201], 14)[200]
        self.assertAlmostEqual(full, partial, places=9)

    def test_rsi_is_causal(self):
        full = ind.rsi(self.closes, 14)[200]
        partial = ind.rsi(self.closes[:201], 14)[200]
        self.assertAlmostEqual(full, partial, places=9)

    def test_adx_is_causal(self):
        full = ind.adx(self.high, self.low, self.closes, 14)[0][200]
        partial = ind.adx(self.high[:201], self.low[:201], self.closes[:201], 14)[0][200]
        self.assertAlmostEqual(full, partial, places=9)

    def test_donchian_excludes_current_bar(self):
        """The channel at bar i must not include bar i's own high, or every
        bar trivially 'breaks out' of itself."""
        high = np.array([10., 11., 12., 20., 13.])
        low = np.array([9., 10., 11., 12., 12.])
        up, _ = ind.donchian(high, low, 3, exclude_current=True)
        self.assertAlmostEqual(up[3], 12.0, places=9)   # max of bars 0..2, not 20
        self.assertAlmostEqual(up[4], 20.0, places=9)   # now bar 3's high counts

    def test_donchian_is_causal(self):
        full = ind.donchian(self.high, self.low, 48)[0][200]
        partial = ind.donchian(self.high[:201], self.low[:201], 48)[0][200]
        self.assertAlmostEqual(full, partial, places=9)

    def test_rel_volume_uses_only_prior_bars(self):
        vol = np.array([100.] * 10 + [1000.])
        rv = ind.rel_volume(vol, 5)
        self.assertAlmostEqual(rv[10], 10.0, places=9)  # 1000 / avg(prior 5 = 100)

    def test_warmup_region_is_nan_not_fabricated(self):
        e = ind.ema(self.closes, 50)
        self.assertTrue(np.all(np.isnan(e[:49])))
        self.assertTrue(np.isfinite(e[49]))


class TestStrategyClosedCandleOnly(unittest.TestCase):
    def setUp(self):
        self.cfg = test_config()
        self.strat = TrendBreakoutStrategy(self.cfg.strategy)
        self.ctx = MarketContext(ts_ms=ALIGNED, btc_regime="bull",
                                 btc_regime_score=70.0, breadth_pct=60.0)

    def test_insufficient_history_fails_closed(self):
        s = make_series("X/USDT", "1h", uptrend_closes(50), ALIGNED)
        sig = self.strat.evaluate(s, None, self.ctx)
        self.assertFalse(sig.passed)
        self.assertIn("insufficient history", sig.reject_reason)

    def test_bearish_btc_regime_blocks_entry(self):
        closes = uptrend_closes(400, seed=5)
        s = make_series("X/USDT", "1h", closes, ALIGNED)
        htf = make_series("X/USDT", "4h", closes[::4], ALIGNED)
        ctx = MarketContext(ts_ms=ALIGNED, btc_regime="bear", btc_regime_score=20.0)
        sig = self.strat.evaluate(s, htf, ctx)
        self.assertFalse(sig.passed)
        self.assertEqual(sig.reject_reason, "BTC regime bearish")

    def test_unknown_btc_regime_fails_closed(self):
        closes = uptrend_closes(400, seed=5)
        s = make_series("X/USDT", "1h", closes, ALIGNED)
        ctx = MarketContext(ts_ms=ALIGNED, btc_regime="unknown")
        sig = self.strat.evaluate(s, None, ctx)
        self.assertFalse(sig.passed)
        self.assertIn("BTC regime", sig.reject_reason)

    def test_blocked_symbol_is_refused(self):
        closes = uptrend_closes(400, seed=5)
        s = make_series("X/USDT", "1h", closes, ALIGNED)
        htf = make_series("X/USDT", "4h", closes[::4], ALIGNED)
        ctx = MarketContext(ts_ms=ALIGNED, btc_regime="bull", btc_regime_score=70.0,
                            blocked_symbols={"X/USDT": "exploit: bridge drained"})
        sig = self.strat.evaluate(s, htf, ctx)
        self.assertFalse(sig.passed)
        self.assertIn("no-trade list", sig.reject_reason)

    def test_corrupt_candles_fail_closed(self):
        closes = uptrend_closes(400, seed=5)
        s = make_series("X/USDT", "1h", closes, ALIGNED)
        s.high[-1] = 0.0          # high below close: impossible
        sig = self.strat.evaluate(s, None, self.ctx)
        self.assertFalse(sig.passed)
        self.assertIn("sanity", sig.reject_reason)

    def test_identical_input_gives_identical_output(self):
        """Determinism: the same market state must always produce the same
        decision. No randomness, no clock dependence, no model in the loop."""
        closes = uptrend_closes(400, seed=5)
        s = make_series("X/USDT", "1h", closes, ALIGNED)
        htf = make_series("X/USDT", "4h", closes[::4], ALIGNED)
        a = self.strat.evaluate(s, htf, self.ctx, {"dollar_volume": 5e7})
        b = self.strat.evaluate(s, htf, self.ctx, {"dollar_volume": 5e7})
        self.assertEqual(a.passed, b.passed)
        self.assertAlmostEqual(a.score, b.score, places=12)
        self.assertEqual(a.reject_reason, b.reject_reason)
        self.assertAlmostEqual(a.stop_price, b.stop_price, places=12)

    def test_future_bars_do_not_change_a_past_decision(self):
        """The core no-look-ahead property: evaluating bar N must give the same
        answer whether or not bars N+1.. exist in the array."""
        closes = uptrend_closes(500, seed=9)
        htf_full = make_series("X/USDT", "4h", closes[::4], ALIGNED)
        s_short = make_series("X/USDT", "1h", closes[:400], ALIGNED)
        htf_short = make_series("X/USDT", "4h", closes[:400][::4], ALIGNED)
        a = self.strat.evaluate(s_short, htf_short, self.ctx)
        # same final bar, but the caller happens to hold more history afterwards
        s_full = make_series("X/USDT", "1h", closes, ALIGNED)
        from crypto_edge.backtest import _slice
        b = self.strat.evaluate(_slice(s_full, 399), htf_short, self.ctx)
        self.assertEqual(a.passed, b.passed)
        self.assertAlmostEqual(a.score, b.score, places=10)
        self.assertEqual(a.reject_reason, b.reject_reason)


class TestBacktestSlicing(unittest.TestCase):
    def test_slice_never_exposes_future(self):
        from crypto_edge.backtest import _slice
        s = make_series("X/USDT", "1h", np.arange(100, 200, dtype=float), ALIGNED)
        sub = _slice(s, 50)
        self.assertEqual(len(sub), 51)
        self.assertEqual(int(sub.open_ms[-1]), int(s.open_ms[50]))

    def test_htf_slice_only_includes_closed_higher_tf_bars(self):
        from crypto_edge.backtest import _slice_htf
        closes = np.arange(100, 200, dtype=float)
        ltf = make_series("X/USDT", "1h", closes, ALIGNED)
        htf = make_series("X/USDT", "4h", closes[::4], ALIGNED)
        # at 1h bar index 5, the 4h bar starting at index 4 is still forming
        sub = _slice_htf(htf, ltf, 5)
        last_open = int(sub.open_ms[-1])
        close_of_last = last_open + 4 * HOUR
        cutoff = int(ltf.open_ms[5]) + HOUR
        self.assertLessEqual(close_of_last, cutoff,
                             "a still-forming 4h bar leaked into the regime calc")


if __name__ == "__main__":
    unittest.main()
