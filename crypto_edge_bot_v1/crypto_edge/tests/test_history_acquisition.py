"""History acquisition: fetch depth must not masquerade as an asset filter.

WHAT WAS REPORTED
-----------------
A live Kraken paper trader evaluated only BTC/USDT and ETH/USDT, cycle after
cycle, despite 29 Kraken markets intersecting the broad universe and 24
surviving stage 1.

WHAT THE AUDIT FOUND -- two unsatisfiable conditions, both provable
-------------------------------------------------------------------
1. `exchange.ohlcv_limit` was 300 while `universe.min_candles_1h` is 400. On any
   venue that honours `limit`, EVERY symbol arrived with 100 fewer bars than
   stage 2 demands and was rejected for "insufficient 1h history" -- for history
   it was never given the chance to supply.

2. Worse, and the binding constraint on Kraken: `filter_by_history` measured
   market age as `open_ms[-1] - open_ms[0]`, the SPAN OF THE WINDOW WE FETCHED,
   and compared it against `min_market_age_days` (45). Kraken's OHLC endpoint
   ignores `limit` entirely and caps every response at 720 candles == 30 days at
   1h. So a ten-year-old market returned 30 days of data and was rejected as
   "market too new". No amount of waiting would ever fix it, on any asset.

Satisfying both filters at 1h needs max(400, 45*24) = 1080 bars. We asked for
300 and Kraken would give at most 720 in one request.

THE FIX -- data acquisition, not thresholds
-------------------------------------------
`min_candles_1h` and `min_market_age_days` are unchanged; they are legitimate
quality gates. What changed is that the feed now PAGES with `since` until it
holds the depth those gates require, `ohlcv_limit` is demoted to a per-request
paging hint, the required depth is DERIVED from the filters so the two cannot
drift apart, an unsatisfiable configuration now refuses to start, and a venue
history cap is reported as a venue limitation rather than blamed on the asset.
"""
import unittest

import numpy as np

import helpers  # noqa: F401  -- silences the engine's log handlers
from crypto_edge.config import Config, load_config
from crypto_edge.data.ccxt_feed import CCXTFeed
from crypto_edge.data.fixture_feed import make_series
from crypto_edge.data.universe import UniverseBuilder
from crypto_edge.timeutils import now_ms, tf_ms
from helpers import test_config

HOUR = 3_600_000


class KrakenLikeClient:
    """A faithful stand-in for Kraken's OHLC endpoint.

    Two behaviours matter and both are real: `limit` is IGNORED, and no response
    ever contains more than `cap` candles. `since` is honoured, which is the
    only reason paging can work at all.
    """

    cap = 720

    def __init__(self, total_bars=5000, step=HOUR, cap=None):
        if cap is not None:
            self.cap = cap
        self.step = step
        newest = (now_ms() // step) * step - step
        self.bars = [[newest - (total_bars - 1 - i) * step,
                      100.0, 101.0, 99.0, 100.5, 10.0]
                     for i in range(total_bars)]
        self.calls = 0

    def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None):
        self.calls += 1
        rows = self.bars
        if since is not None:
            rows = [r for r in rows if r[0] >= since]
        return [list(r) for r in rows[:self.cap]]      # `limit` deliberately ignored


def bare_feed(client, page_limit=300):
    """A CCXTFeed without ccxt or a network, for the paging logic."""
    feed = object.__new__(CCXTFeed)
    feed.name, feed.quote, feed.rate_limit_ms = "kraken", "USDT", 0
    feed.client = client
    feed.page_limit = page_limit
    feed.precision_mode = 4
    feed.quote_ts_fallback = "local"
    feed._markets = {}
    feed.quote_ts_venue = feed.quote_ts_local = 0
    feed._sleep = lambda: None
    return feed


class TestTheUnsatisfiableConfiguration(unittest.TestCase):
    """The arithmetic that made every symbol fail, stated as tests."""

    def test_satisfying_both_filters_needs_more_bars_than_one_request_gives(self):
        cfg = load_config("config/config.toml", "/nonexistent")
        need = cfg.required_history_bars("1h")
        self.assertGreaterEqual(need, cfg.universe.min_candles_1h)
        self.assertGreaterEqual(need, cfg.universe.min_market_age_days * 24)
        self.assertGreater(need, KrakenLikeClient.cap,
                           "the required depth exceeds one Kraken response, "
                           "which is exactly why paging is required")

    def test_required_depth_is_derived_from_the_filters_not_hardcoded(self):
        cfg = test_config()
        cfg.universe.min_candles_1h = 400
        cfg.universe.min_market_age_days = 45
        self.assertGreaterEqual(cfg.required_history_bars("1h"), 45 * 24)
        cfg.universe.min_market_age_days = 90
        self.assertGreaterEqual(cfg.required_history_bars("1h"), 90 * 24,
                                "raising the age gate must raise the fetch depth")

    def test_the_htf_timeframe_needs_proportionally_fewer_bars(self):
        cfg = test_config()
        cfg.universe.min_market_age_days = 45
        self.assertLess(cfg.required_history_bars("4h"),
                        cfg.required_history_bars("1h"))

    def test_an_unsatisfiable_history_budget_refuses_to_start(self):
        cfg = load_config("config/config.toml", "/nonexistent")
        cfg.telegram.enabled = False
        cfg.exchange.max_history_bars = 300
        errs = cfg.validate()
        self.assertTrue(any("max_history_bars" in e for e in errs),
                        "a budget that can never satisfy the filters must be "
                        "rejected loudly, not silently reject every symbol")

    def test_the_shipped_configuration_is_satisfiable(self):
        cfg = load_config("config/config.toml", "/nonexistent")
        cfg.telegram.enabled = False
        self.assertEqual(cfg.validate(), [])
        self.assertGreaterEqual(cfg.exchange.max_history_bars,
                                cfg.required_history_bars("1h"))


class TestPagingReachesTheRequiredDepth(unittest.TestCase):
    def test_a_single_request_cannot_reach_the_required_depth(self):
        """The precondition -- without paging the fix would be impossible."""
        client = KrakenLikeClient()
        one_shot = client.fetch_ohlcv("BTC/USDT", "1h", limit=1085)
        self.assertEqual(len(one_shot), 720,
                         "Kraken ignores limit and caps at 720")

    def test_paging_reaches_the_full_requested_depth(self):
        feed = bare_feed(KrakenLikeClient())
        series = feed.fetch_ohlcv("BTC/USDT", "1h", 1085)
        self.assertEqual(len(series), 1085)
        self.assertGreater(feed.client.calls, 1, "more than one request needed")

    def test_paged_bars_are_ordered_unique_and_evenly_spaced(self):
        feed = bare_feed(KrakenLikeClient())
        series = feed.fetch_ohlcv("BTC/USDT", "1h", 1085)
        opens = series.open_ms
        self.assertTrue(np.all(np.diff(opens) == HOUR),
                        "pages must join seamlessly, with no gap or overlap")
        self.assertEqual(len(set(opens.tolist())), len(opens))
        self.assertTrue(series.is_sane())

    def test_the_newest_bar_is_still_the_newest(self):
        """Paging backwards must not cost us the most recent candle."""
        client = KrakenLikeClient()
        feed = bare_feed(client)
        series = feed.fetch_ohlcv("BTC/USDT", "1h", 1085)
        self.assertEqual(int(series.open_ms[-1]), client.bars[-1][0])

    def test_a_venue_with_less_history_than_asked_returns_what_it_has(self):
        feed = bare_feed(KrakenLikeClient(total_bars=500))
        series = feed.fetch_ohlcv("NEW/USDT", "1h", 1085)
        self.assertEqual(len(series), 500, "no invention, no error -- just less")

    def test_paging_terminates_when_the_venue_stops_advancing(self):
        """A venue that keeps returning the same page must not spin forever."""
        class Stuck(KrakenLikeClient):
            def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None):
                self.calls += 1
                return [list(r) for r in self.bars[:10]]

        feed = bare_feed(Stuck())
        series = feed.fetch_ohlcv("X/USDT", "1h", 1085)
        self.assertEqual(len(series), 10)
        self.assertLess(feed.client.calls, 20, "must not loop unboundedly")

    def test_a_venue_that_honours_limit_still_works(self):
        class Honest(KrakenLikeClient):
            def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None):
                self.calls += 1
                rows = self.bars
                if since is not None:
                    rows = [r for r in rows if r[0] >= since]
                return [list(r) for r in rows[:min(limit or self.cap, self.cap)]]

        feed = bare_feed(Honest(), page_limit=1000)
        self.assertEqual(len(feed.fetch_ohlcv("BTC/USDT", "1h", 1085)), 1085)

    def test_a_mid_page_failure_keeps_what_was_already_collected(self):
        class Flaky(KrakenLikeClient):
            def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None):
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("rate limited")
                return super().fetch_ohlcv(symbol, timeframe, since, limit)

        feed = bare_feed(Flaky())
        series = feed.fetch_ohlcv("BTC/USDT", "1h", 1085)
        self.assertEqual(len(series), 720, "partial history beats no history")

    def test_a_first_page_failure_is_still_an_error(self):
        from crypto_edge.data.feed import DataUnavailable

        class Dead(KrakenLikeClient):
            def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None):
                raise RuntimeError("exchange down")

        with self.assertRaises(DataUnavailable):
            bare_feed(Dead()).fetch_ohlcv("BTC/USDT", "1h", 1085)

    def test_4h_paging_works_at_its_own_step(self):
        feed = bare_feed(KrakenLikeClient(step=4 * HOUR, total_bars=3000))
        series = feed.fetch_ohlcv("BTC/USDT", "4h", 900)
        self.assertEqual(len(series), 900)
        self.assertTrue(np.all(np.diff(series.open_ms) == 4 * HOUR))


class TestTruncationIsNotBlamedOnTheAsset(unittest.TestCase):
    """A venue history cap and a young market look identical in the data."""

    def setUp(self):
        cfg = test_config()
        cfg.universe.min_candles_1h = 400
        cfg.universe.min_market_age_days = 45
        cfg.universe.min_atr_pct = 0.0
        cfg.universe.max_atr_pct = 1000.0
        self.b = UniverseBuilder(cfg.universe)

    def _series(self, bars):
        closes = 100.0 + np.arange(bars, dtype=float) * 0.01
        start = now_ms() - bars * HOUR
        return make_series("X/USDT", "1h", closes, start)

    def test_a_truncated_response_is_reported_as_a_venue_limitation(self):
        """720 bars when 1085 were asked for: our fetch is short, not the asset."""
        reason = self.b.filter_by_history("X/USDT", self._series(720),
                                          requested_bars=1085)
        self.assertIn("truncated by venue", reason)
        self.assertIn("1085", reason)
        self.assertNotIn("too new", reason,
                         "the asset must not be blamed for our fetch depth")

    def test_a_genuinely_young_market_is_still_called_young(self):
        """Everything we asked for arrived and it really is only 30 days old."""
        reason = self.b.filter_by_history("X/USDT", self._series(720),
                                          requested_bars=720)
        self.assertIn("too new", reason)
        self.assertNotIn("truncated", reason)

    def test_full_history_passes_the_age_gate(self):
        self.assertEqual(
            self.b.filter_by_history("X/USDT", self._series(1085),
                                     requested_bars=1085),
            "", "1085 bars is 45 days; the gate must be satisfiable")

    def test_a_short_truncated_response_names_the_venue_too(self):
        reason = self.b.filter_by_history("X/USDT", self._series(300),
                                          requested_bars=1085)
        self.assertIn("venue supplied only", reason)
        self.assertIn("300", reason)

    def test_without_the_request_context_behaviour_is_unchanged(self):
        """Callers that do not pass requested_bars keep the original reasons."""
        reason = self.b.filter_by_history("X/USDT", self._series(300))
        self.assertIn("1h candles", reason)
        self.assertNotIn("venue", reason)

    def test_both_outcomes_still_block_the_entry(self):
        """Distinguishing the reasons must not make either of them permissive."""
        for requested in (720, 1085):
            self.assertNotEqual(
                self.b.filter_by_history("X/USDT", self._series(720),
                                         requested_bars=requested), "",
                "fail closed either way")


class TestEndToEndOnAKrakenLikeVenue(unittest.TestCase):
    """The reported symptom, reproduced and then shown fixed."""

    def _pipeline(self, feed, cfg, symbol="SOL/USDT"):
        want = cfg.required_history_bars("1h")
        series = feed.fetch_ohlcv(symbol, "1h", want)
        return UniverseBuilder(cfg.universe).filter_by_history(
            symbol, series, None, requested_bars=want), len(series)

    def test_the_old_single_request_behaviour_rejected_everything(self):
        """Pinning the defect: one capped response can never satisfy the gates."""
        cfg = load_config("config/config.toml", "/nonexistent")
        client = KrakenLikeClient()
        one_page = client.fetch_ohlcv("SOL/USDT", "1h", limit=300)
        series = make_series(
            "SOL/USDT", "1h",
            [r[4] for r in one_page],
            one_page[0][0])
        reason = UniverseBuilder(cfg.universe).filter_by_history(
            "SOL/USDT", series, None)
        self.assertNotEqual(reason, "",
                            "a single capped response is always rejected")

    def test_with_paging_a_mature_market_becomes_eligible(self):
        cfg = load_config("config/config.toml", "/nonexistent")
        cfg.universe.min_atr_pct = 0.0
        feed = bare_feed(KrakenLikeClient(total_bars=5000))
        reason, bars = self._pipeline(feed, cfg)
        self.assertGreaterEqual(bars, cfg.universe.min_candles_1h)
        self.assertGreaterEqual(bars, cfg.universe.min_market_age_days * 24)
        self.assertEqual(reason, "",
                         f"a mature market must survive stage 2; got {reason!r}")

    def test_a_genuinely_new_listing_is_still_excluded(self):
        """The fix must not smuggle young markets in -- that would be loosening."""
        cfg = load_config("config/config.toml", "/nonexistent")
        feed = bare_feed(KrakenLikeClient(total_bars=200))
        reason, bars = self._pipeline(feed, cfg)
        self.assertEqual(bars, 200)
        self.assertNotEqual(reason, "", "200 bars must never be eligible")

    def test_thresholds_themselves_are_untouched(self):
        """Guard against 'fixing' this by weakening the strategy."""
        cfg = load_config("config/config.toml", "/nonexistent")
        self.assertEqual(cfg.universe.min_candles_1h, 400)
        self.assertEqual(cfg.universe.min_market_age_days, 45)
        self.assertEqual(cfg.strategy.warmup_bars, 250)
        self.assertEqual(cfg.strategy.ema_trend, 200)
        self.assertEqual(cfg.risk.risk_per_trade_pct, 0.5)


if __name__ == "__main__":
    unittest.main()
