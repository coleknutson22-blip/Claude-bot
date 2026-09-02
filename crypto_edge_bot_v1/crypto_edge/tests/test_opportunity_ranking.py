"""Ranking, the shortlist cut, and the cost of a scan cycle.

TWO PROPERTIES THIS FILE EXISTS TO HOLD
---------------------------------------
1. THE RANKING IS ACTUALLY USED. The bot this strategy is modelled on computed
   a candidate score and then called `random.shuffle()` on the result before
   picking one -- the ranking was discarded, and the effect was a random choice
   among the top twelve. Order here is a pure function of the inputs, ties break
   on symbol, and these tests assert it rather than trusting it.

2. DEEP-ANALYSIS COST IS BOUNDED BY CONFIGURATION. Fetching 5m and 15m for every
   liquid market would undo the caching work that took a measured cycle from
   74.6 seconds to near zero between bar closes. Ranking runs on the hourly
   series the engine already holds and costs nothing; only the shortlist is
   deepened. A quiet universe and a frantic one cost the same.
"""
import unittest

import numpy as np

import helpers  # noqa: F401  -- silences the engine's log handlers
from crypto_edge.config import Config
from crypto_edge.data.feed import DataUnavailable
from crypto_edge.scan import scan
from crypto_edge.strategy import ranking
from crypto_edge.strategy.base import MarketContext
from fixtures_fast import frames

CTX = MarketContext(ts_ms=1_700_000_000_000, btc_regime="neutral",
                    btc_regime_score=50.0, breadth_pct=50.0)


def universe(n=30, seed0=100):
    """n markets with a spread of drifts, so the ranking has work to do."""
    out = {}
    for i in range(n):
        drift = (i - n / 2) * 0.00008
        sym = f"A{i:02d}/USD"
        out[sym] = frames(drift, seed=seed0 + i, symbol=sym, n_5m=3600)
    return out


def meta_for(symbols, dollar_volume=2e7, spread=6.0):
    return {s: {"dollar_volume": dollar_volume, "spread_bps": spread}
            for s in symbols}


class FakeFeed:
    """Serves prepared frames and counts every request."""

    def __init__(self, all_frames):
        self.all = all_frames
        self.calls = []

    def fetch_ohlcv(self, symbol, timeframe, limit):
        self.calls.append((symbol, timeframe))
        fr = self.all.get(symbol)
        if fr is None or timeframe not in fr:
            raise DataUnavailable(f"no {timeframe} for {symbol}")
        return fr[timeframe]


class RankingCase(unittest.TestCase):
    def setUp(self):
        self.frames = universe(30)
        self.hourly = {s: fr["1h"] for s, fr in self.frames.items()}
        self.meta = meta_for(self.hourly)

    def rank(self, hourly=None, meta=None):
        return ranking.rank(hourly if hourly is not None else self.hourly,
                            None, meta if meta is not None else self.meta,
                            min_dollar_volume=5e6)


class TestRankingIsDeterministic(RankingCase):
    def test_the_same_inputs_give_the_same_order(self):
        a = [c.symbol for c in self.rank()]
        b = [c.symbol for c in self.rank()]
        self.assertEqual(a, b)

    def test_the_order_does_not_depend_on_dict_insertion_order(self):
        first = [c.symbol for c in self.rank()]
        shuffled = dict(reversed(list(self.hourly.items())))
        self.assertEqual(first, [c.symbol for c in self.rank(hourly=shuffled)])

    def test_scores_are_reproducible_to_the_bit(self):
        a = {c.symbol: c.score for c in self.rank()}
        b = {c.symbol: c.score for c in self.rank()}
        self.assertEqual(a, b)

    def test_ties_break_on_symbol_name(self):
        """Identical inputs must not produce an arbitrary order."""
        one = self.frames["A05/USD"]["1h"]
        same = {f"T{i}/USD": one for i in range(5)}
        ranked = ranking.rank(same, None, meta_for(same), min_dollar_volume=5e6)
        self.assertEqual([c.symbol for c in ranked], sorted(same))

    def test_no_module_in_the_ranking_path_uses_randomness(self):
        """The specific defect in the source bot, kept out by test.

        Parsed rather than grepped: the docstrings here DESCRIBE the defect, so
        a text search matches the explanation and would pass or fail for the
        wrong reason. Only real imports and real calls count.
        """
        import ast
        import inspect

        from crypto_edge import scan as scan_mod
        from crypto_edge.strategy import aggressive_momentum
        for mod in (ranking, scan_mod, aggressive_momentum):
            tree = ast.parse(inspect.getsource(mod))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        self.assertNotEqual(n.name, "random", mod.__name__)
                if isinstance(node, ast.ImportFrom):
                    self.assertNotEqual(node.module, "random", mod.__name__)
                if isinstance(node, ast.Call):
                    fn = node.func
                    name = (fn.attr if isinstance(fn, ast.Attribute)
                            else getattr(fn, "id", ""))
                    self.assertNotIn(name, ("shuffle", "sample", "choice"),
                                     mod.__name__)


class TestRankingOrder(RankingCase):
    def test_ranks_are_dense_and_start_at_one(self):
        ranked = self.rank()
        self.assertEqual([c.rank for c in ranked], list(range(1, len(ranked) + 1)))

    def test_scores_are_non_increasing(self):
        scores = [c.score for c in self.rank()]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_a_more_liquid_market_outranks_an_identical_thin_one(self):
        one = self.frames["A20/USD"]["1h"]
        pair = {"THICK/USD": one, "THIN/USD": one}
        meta = {"THICK/USD": {"dollar_volume": 5e8, "spread_bps": 3.0},
                "THIN/USD": {"dollar_volume": 6e6, "spread_bps": 28.0}}
        ranked = ranking.rank(pair, None, meta, min_dollar_volume=5e6)
        self.assertEqual(ranked[0].symbol, "THICK/USD")

    def test_ranking_is_direction_agnostic(self):
        """A collapse is as interesting as a rally; both must rank on merit."""
        up = frames(0.0006, seed=1, symbol="UP/USD")["1h"]
        down = frames(-0.0006, seed=1, symbol="DOWN/USD")["1h"]
        pair = {"UP/USD": up, "DOWN/USD": down}
        ranked = ranking.rank(pair, None, meta_for(pair), min_dollar_volume=5e6)
        scores = {c.symbol: c.components["movement"] for c in ranked}
        self.assertGreater(scores["DOWN/USD"], 0.0)
        self.assertGreater(scores["UP/USD"], 0.0)

    def test_every_component_is_recorded_for_research(self):
        c = self.rank()[0]
        self.assertEqual(set(c.components), set(ranking.RANK_WEIGHTS))
        self.assertIn("atr_pct", c.inputs)
        self.assertIn("dollar_volume", c.inputs)

    def test_every_rank_weight_has_a_justification(self):
        self.assertEqual(set(ranking.RANK_JUSTIFICATIONS),
                         set(ranking.RANK_WEIGHTS))

    def test_a_series_too_short_to_rank_is_skipped_not_guessed(self):
        from crypto_edge.models import Series
        s = self.hourly["A00/USD"]
        short = Series(s.symbol, "1h", s.open_ms[:10], s.open[:10], s.high[:10],
                       s.low[:10], s.close[:10], s.volume[:10])
        ranked = ranking.rank({"A00/USD": short}, None, self.meta,
                              min_dollar_volume=5e6)
        self.assertEqual(ranked, [])

    def test_volatility_fit_is_a_band_not_a_ramp(self):
        """Too quiet cannot pay costs; too wild cannot be sized."""
        self.assertEqual(ranking.score_volatility_fit(1.0, 0.3, 6.0), 100.0)
        self.assertLess(ranking.score_volatility_fit(0.05, 0.3, 6.0), 100.0)
        self.assertLess(ranking.score_volatility_fit(20.0, 0.3, 6.0), 100.0)


class TestShortlistCutoff(RankingCase):
    def test_the_shortlist_is_the_top_n_in_order(self):
        ranked = self.rank()
        short = ranking.shortlist(ranked, 12)
        self.assertEqual(len(short), 12)
        self.assertEqual([c.symbol for c in short],
                         [c.symbol for c in ranked[:12]])

    def test_a_shortlist_larger_than_the_field_returns_the_field(self):
        ranked = self.rank()
        self.assertEqual(len(ranking.shortlist(ranked, 1000)), len(ranked))

    def test_a_zero_or_negative_shortlist_is_empty_not_everything(self):
        ranked = self.rank()
        self.assertEqual(ranking.shortlist(ranked, 0), [])
        self.assertEqual(ranking.shortlist(ranked, -3), [])

    def test_the_configured_size_is_in_the_ten_to_fifteen_band(self):
        from crypto_edge.config import AggressiveCfg
        self.assertGreaterEqual(AggressiveCfg().shortlist_size, 10)
        self.assertLessEqual(AggressiveCfg().shortlist_size, 15)


class TestScanCycleCost(unittest.TestCase):
    def setUp(self):
        self.frames = universe(30)
        self.hourly = {s: fr["1h"] for s, fr in self.frames.items()}
        self.meta = meta_for(self.hourly)
        self.feed = FakeFeed(self.frames)
        self.cfg = Config()
        self.cfg.telegram.enabled = False

    def run_scan(self, **over):
        for k, v in over.items():
            setattr(self.cfg.aggressive, k, v)
        return scan(self.cfg, self.feed, CTX, rank_series=self.hourly,
                    btc_1h=None, meta_by_symbol=self.meta,
                    now_ms=CTX.ts_ms + 10 ** 9, buffer_ms=0)

    def test_ranking_itself_costs_no_requests(self):
        ranked = ranking.rank(self.hourly, None, self.meta, min_dollar_volume=5e6)
        self.assertEqual(len(ranked), 30)
        self.assertEqual(self.feed.calls, [],
                         "the ranking pass must reuse candles already held")

    def test_deep_fetches_are_bounded_by_the_shortlist(self):
        res = self.run_scan(shortlist_size=12)
        self.assertEqual(len(res.shortlist), 12)
        self.assertLessEqual(res.deep_fetches, 12 * 2,
                             "two extra timeframes per shortlisted symbol, no more")

    def test_the_hourly_series_is_never_refetched(self):
        res = self.run_scan(shortlist_size=12)
        hourly_calls = [c for c in self.feed.calls if c[1] == "1h"]
        self.assertEqual(hourly_calls, [],
                         "1h is already in memory; refetching it is pure waste")

    def test_a_bigger_universe_does_not_cost_more(self):
        """The bound is CONFIGURATION, not how many markets qualify."""
        small = self.run_scan(shortlist_size=12).deep_fetches
        self.frames = universe(120, seed0=500)
        self.hourly = {s: fr["1h"] for s, fr in self.frames.items()}
        self.meta = meta_for(self.hourly)
        self.feed = FakeFeed(self.frames)
        big = self.run_scan(shortlist_size=12).deep_fetches
        self.assertEqual(small, big)

    def test_only_shortlisted_symbols_are_deepened(self):
        res = self.run_scan(shortlist_size=8)
        fetched = {c[0] for c in self.feed.calls}
        self.assertEqual(fetched, {c.symbol for c in res.shortlist})

    def test_every_shortlisted_symbol_produces_a_signal(self):
        res = self.run_scan(shortlist_size=10)
        self.assertEqual(len(res.signals), 10)
        for sig in res.signals:
            self.assertIn(sig.side, ("long", "short", "none"))

    def test_the_rank_travels_with_the_signal_for_research(self):
        res = self.run_scan(shortlist_size=10)
        ranks = [s.features["rank"] for s in res.signals]
        self.assertEqual(ranks, sorted(ranks))
        self.assertEqual(ranks[0], 1)
        for s in res.signals:
            self.assertIn("rank_score", s.features)
            self.assertIn("rank_components", s.features)

    def test_a_failed_deep_fetch_skips_that_symbol_and_records_why(self):
        broken = dict(self.frames)
        victim = ranking.rank(self.hourly, None, self.meta,
                              min_dollar_volume=5e6)[0].symbol
        broken[victim] = {"1h": self.frames[victim]["1h"]}     # 5m/15m missing
        self.feed = FakeFeed(broken)
        res = self.run_scan(shortlist_size=6)
        self.assertIn(victim, res.fetch_failures)
        self.assertNotIn(victim, [s.symbol for s in res.signals])
        self.assertEqual(len(res.signals), 5, "the others are unaffected")

    def test_the_scan_is_reproducible(self):
        a = self.run_scan(shortlist_size=10)
        self.feed = FakeFeed(self.frames)
        b = self.run_scan(shortlist_size=10)
        self.assertEqual([s.symbol for s in a.signals],
                         [s.symbol for s in b.signals])
        self.assertEqual([s.side for s in a.signals], [s.side for s in b.signals])
        self.assertEqual([round(s.score, 9) for s in a.signals],
                         [round(s.score, 9) for s in b.signals])

    def test_entries_are_a_subset_of_the_signals(self):
        res = self.run_scan(shortlist_size=12)
        self.assertTrue(set(id(s) for s in res.entries)
                        <= set(id(s) for s in res.signals))
        for s in res.entries:
            self.assertTrue(s.passed)
            self.assertIn(s.side, ("long", "short"))


class TestWillingnessToTrade(unittest.TestCase):
    """Strategy B exists to take more trades than Strategy A. Measure it."""

    def test_it_finds_entries_a_donchian_breakout_would_miss(self):
        from crypto_edge.config import AggressiveCfg, StrategyCfg
        from crypto_edge.strategy.aggressive_momentum import \
            AggressiveMomentumStrategy
        from crypto_edge.strategy.trend_breakout import TrendBreakoutStrategy

        b = AggressiveMomentumStrategy(AggressiveCfg())
        a = TrendBreakoutStrategy(StrategyCfg())
        rng = np.random.default_rng(99)
        a_n = b_n = 0
        sides = set()
        for seed in range(60):
            drift = float(rng.normal(0, 0.00035))
            fr = frames(drift, seed=seed, n_5m=4000)
            ctx = MarketContext(ts_ms=CTX.ts_ms,
                                btc_regime="bull" if drift > 0 else "neutral",
                                btc_regime_score=60.0, breadth_pct=55.0)
            sb = b.evaluate_frames(fr, ctx)
            if sb.passed:
                b_n += 1
                sides.add(sb.side)
            sa = a.evaluate(fr["1h"], fr["1h"], ctx,
                            meta={"dollar_volume": 2e7, "spread_bps": 6.0,
                                  "min_dollar_volume": 5e6, "btc_roc": 0.0})
            a_n += 1 if sa.passed else 0
        self.assertGreater(b_n, a_n * 3,
                           f"aggressive={b_n} vs trend_breakout={a_n}")
        self.assertEqual(sides, {"long", "short"}, "both sides must be reachable")


if __name__ == "__main__":
    unittest.main()
