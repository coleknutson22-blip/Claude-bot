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
    """A faithful stand-in for Kraken's OHLC endpoint AS SEEN THROUGH CCXT.

    Three behaviours matter and all three are real. The third was missing from
    this stand-in for a long time, and its absence is the direct reason a
    one-bar staleness bug reached a live machine with the suite fully green.

    1. Kraken IGNORES `limit` server-side, and no response ever contains more
       than `cap` candles.
    2. The newest row is the candle CURRENTLY IN PROGRESS. Real venues return
       it; a stand-in that stops at the last closed bar leaves `drop_unclosed`
       with nothing to drop and hides every off-by-one around the live edge.
    3. CCXT then post-processes the response in `filter_by_since_limit`. When
       `since` is given it sets `shouldFilterFromStart=True`, so `filter_by_limit`
       returns `array[0:limit]` -- the OLDEST `limit` rows, DISCARDING THE
       NEWEST. Ask for a span wider than `limit` and the bars you lose are the
       most recent ones.

    Verified against ccxt 4.5.75: `base/exchange.py` `filter_by_since_limit`
    (line ~3386) and `filter_by_limit` (line ~3353).
    """

    cap = 720

    def __init__(self, total_bars=5000, step=HOUR, cap=None):
        if cap is not None:
            self.cap = cap
        self.step = step
        # bars[-1] is the IN-PROGRESS candle, exactly as a venue reports it.
        newest = (now_ms() // step) * step
        self.bars = [[newest - (total_bars - 1 - i) * step,
                      100.0, 101.0, 99.0, 100.5, 10.0]
                     for i in range(total_bars)]
        self.calls = 0

    def _venue_rows(self, since):
        # No `since`: Kraken answers with its most recent `cap` candles.
        # With `since`: it walks FORWARD from there, still capped at `cap`.
        if since is None:
            return self.bars[-self.cap:]
        return [r for r in self.bars if r[0] >= since][:self.cap]

    def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None):
        self.calls += 1
        rows = self._venue_rows(since)
        # --- ccxt's filter_by_since_limit, reproduced ---
        if since is not None:
            rows = [r for r in rows if r[0] >= since]
        if limit is not None:
            rows = rows[:limit] if since is not None else rows[-limit:]
        return [list(r) for r in rows]


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


class TestHistoryDepthIsIndicatorsOnly(unittest.TestCase):
    """Age was folded into the history requirement; it no longer is.

    Combining them demanded max(400, 45*24) = 1080 hourly bars from a venue that
    returns 720, so every market failed forever. Indicator depth and calendar
    age are now answered separately, at the resolution each one needs.
    """

    def test_indicator_depth_excludes_the_age_gate(self):
        cfg = load_config("config/config.toml", "/nonexistent")
        need = cfg.required_history_bars("1h")
        self.assertGreaterEqual(need, cfg.universe.min_candles_1h)
        self.assertGreaterEqual(need, cfg.strategy.warmup_bars)
        self.assertLess(need, cfg.universe.min_market_age_days * 24,
                        "the age gate must no longer inflate the bar request")

    def test_the_indicator_depth_now_fits_one_kraken_response(self):
        cfg = load_config("config/config.toml", "/nonexistent")
        self.assertLessEqual(cfg.required_history_bars("1h"),
                             KrakenLikeClient.cap,
                             "405 bars fits inside Kraken's 720-bar cap")

    def test_raising_the_age_gate_does_not_change_the_bar_request(self):
        cfg = test_config()
        before = cfg.required_history_bars("1h")
        cfg.universe.min_market_age_days = 365
        self.assertEqual(cfg.required_history_bars("1h"), before,
                         "age is a calendar span; it must not move a bar count")

    def test_raising_the_indicator_requirements_does_change_it(self):
        cfg = test_config()
        before = cfg.required_history_bars("1h")
        cfg.universe.min_candles_1h += 200
        self.assertGreater(cfg.required_history_bars("1h"), before)

    def test_an_age_probe_that_cannot_reach_the_gate_is_rejected(self):
        cfg = load_config("config/config.toml", "/nonexistent")
        cfg.telegram.enabled = False
        cfg.universe.age_probe_bars = 5          # 5 days of reach vs a 45d gate
        errs = cfg.validate()
        self.assertTrue(any("age_probe_bars" in e for e in errs),
                        "a probe that can never evidence the gate must refuse "
                        "to start, not silently fail every market")

    def test_the_shipped_age_probe_reaches_well_past_the_gate(self):
        cfg = load_config("config/config.toml", "/nonexistent")
        reach_days = cfg.universe.age_probe_bars * tf_ms(
            cfg.universe.age_probe_timeframe) / 86_400_000
        self.assertGreater(reach_days, cfg.universe.min_market_age_days * 2,
                           "comfortable headroom, not a knife edge")

    def test_an_unsatisfiable_history_budget_refuses_to_start(self):
        cfg = load_config("config/config.toml", "/nonexistent")
        cfg.telegram.enabled = False
        cfg.exchange.max_history_bars = 10
        self.assertTrue(any("max_history_bars" in e for e in cfg.validate()))

    def test_the_shipped_configuration_is_satisfiable(self):
        cfg = load_config("config/config.toml", "/nonexistent")
        cfg.telegram.enabled = False
        self.assertEqual(cfg.validate(), [])


class TestPagingReachesTheRequiredDepth(unittest.TestCase):
    def test_a_single_request_cannot_reach_the_required_depth(self):
        """The precondition -- without paging the fix would be impossible."""
        client = KrakenLikeClient()
        one_shot = client.fetch_ohlcv("BTC/USDT", "1h", limit=1085)
        self.assertEqual(len(one_shot), 720,
                         "Kraken ignores limit and caps at 720")

    def test_paging_reaches_the_full_requested_depth(self):
        """The number that matters is CLOSED bars, after the live one is dropped.

        The fetch deliberately returns one extra row -- the candle currently in
        progress -- so that `drop_unclosed` leaves exactly the requested depth
        rather than one short of it.
        """
        feed = bare_feed(KrakenLikeClient())
        series = feed.fetch_ohlcv("BTC/USDT", "1h", 1085)
        closed = series.drop_unclosed(now_ms(), 0)
        self.assertGreaterEqual(len(closed), 1085, "at least the depth asked for")
        self.assertLessEqual(len(closed), 1088, "and not an unbounded over-fetch")
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
            """Binance-shaped: `limit` is applied SERVER-side.

            With no `since` a real venue answers with its most RECENT rows --
            that is what makes the since-less first request safe everywhere.
            """

            def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None):
                self.calls += 1
                n = min(limit or self.cap, self.cap)
                if since is None:
                    return [list(r) for r in self.bars[-n:]]
                return [list(r) for r in
                        [r for r in self.bars if r[0] >= since][:n]]

        feed = bare_feed(Honest(), page_limit=1000)
        closed = feed.fetch_ohlcv("BTC/USDT", "1h", 1085).drop_unclosed(now_ms(), 0)
        self.assertGreaterEqual(len(closed), 1085)

    def test_a_mid_page_failure_keeps_what_was_already_collected(self):
        class Flaky(KrakenLikeClient):
            def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None):
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("rate limited")
                return super().fetch_ohlcv(symbol, timeframe, since, limit)

        feed = bare_feed(Flaky())
        series = feed.fetch_ohlcv("BTC/USDT", "1h", 1085)
        self.assertGreater(len(series), 0, "partial history beats no history")
        # WHICH bars survive matters more than how many. Paging backwards from
        # the present means a failure costs the OLDEST history, never the
        # newest -- the opposite trade to paging forwards, and the right one:
        # a short recent series is usable, a long stale one is not.
        self.assertEqual(int(series.open_ms[-1]), feed.client.bars[-1][0],
                         "the live edge survives a mid-page failure")
        self.assertTrue(series.is_sane(), "no hole spliced into what survived")

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
        self.assertGreaterEqual(len(series.drop_unclosed(now_ms(), 0)), 900)
        self.assertTrue(np.all(np.diff(series.open_ms) == 4 * HOUR))


class TestTruncationIsNotBlamedOnTheAsset(unittest.TestCase):
    """A short response is a fact about our fetch, not about the asset."""

    def setUp(self):
        cfg = test_config()
        cfg.universe.min_candles_1h = 400
        cfg.universe.min_atr_pct = 0.0
        cfg.universe.max_atr_pct = 1000.0
        self.b = UniverseBuilder(cfg.universe)

    def _series(self, bars):
        closes = 100.0 + np.arange(bars, dtype=float) * 0.01
        start = now_ms() - bars * HOUR
        return make_series("X/USDT", "1h", closes, start)

    def test_a_short_truncated_response_names_the_venue(self):
        reason = self.b.filter_by_history("X/USDT", self._series(300),
                                          requested_bars=405)
        self.assertIn("venue supplied only", reason)
        self.assertIn("300", reason)

    def test_a_short_response_without_request_context_reads_as_history(self):
        reason = self.b.filter_by_history("X/USDT", self._series(300))
        self.assertIn("1h candles", reason)

    def test_enough_bars_passes_regardless_of_the_span_they_cover(self):
        """The decoupling, stated directly: 405 bars is 17 days of calendar and
        that is now irrelevant to the history gate."""
        self.assertEqual(
            self.b.filter_by_history("X/USDT", self._series(405),
                                     requested_bars=405), "")

    def test_a_720_bar_kraken_response_satisfies_the_history_gate(self):
        """The exact series that used to be rejected as 'too new'."""
        self.assertEqual(
            self.b.filter_by_history("X/USDT", self._series(720),
                                     requested_bars=405), "")

    def test_the_history_gate_no_longer_mentions_age_at_all(self):
        for bars in (405, 500, 720, 1085):
            reason = self.b.filter_by_history("X/USDT", self._series(bars),
                                              requested_bars=405)
            self.assertNotIn("too new", reason)
            self.assertNotIn("truncated by venue", reason)


class TestEndToEndOnAKrakenLikeVenue(unittest.TestCase):
    """The reported symptom, reproduced and then shown fixed."""

    def _history_reason(self, feed, cfg, symbol="SOL/USDT"):
        want = cfg.required_history_bars("1h")
        series = feed.fetch_ohlcv(symbol, "1h", want)
        return UniverseBuilder(cfg.universe).filter_by_history(
            symbol, series, None, requested_bars=want), len(series)

    def test_the_old_combined_requirement_rejected_a_full_kraken_response(self):
        """Pinning the defect: 720 bars is everything Kraken has at 1h, and the
        old requirement of 1080 rejected it -- so no asset could ever pass."""
        cfg = load_config("config/config.toml", "/nonexistent")
        old_combined = max(cfg.universe.min_candles_1h, cfg.strategy.warmup_bars,
                           cfg.universe.min_market_age_days * 24)
        self.assertGreater(old_combined, KrakenLikeClient.cap)
        self.assertLessEqual(cfg.required_history_bars("1h"),
                             KrakenLikeClient.cap,
                             "the corrected requirement fits what Kraken gives")

    def test_a_kraken_capped_response_now_satisfies_the_history_gate(self):
        cfg = load_config("config/config.toml", "/nonexistent")
        cfg.universe.min_atr_pct = 0.0
        feed = bare_feed(KrakenLikeClient(total_bars=5000))
        reason, bars = self._history_reason(feed, cfg)
        self.assertGreaterEqual(bars, cfg.universe.min_candles_1h)
        self.assertEqual(reason, "", f"expected eligible, got {reason!r}")

    def test_a_genuinely_short_series_is_still_excluded(self):
        cfg = load_config("config/config.toml", "/nonexistent")
        feed = bare_feed(KrakenLikeClient(total_bars=200))
        reason, bars = self._history_reason(feed, cfg)
        self.assertEqual(bars, 200)
        self.assertNotEqual(reason, "", "200 bars must never satisfy 400")

    def test_thresholds_themselves_are_untouched(self):
        """Guard against 'fixing' this by weakening the strategy."""
        cfg = load_config("config/config.toml", "/nonexistent")
        self.assertEqual(cfg.universe.min_candles_1h, 400)
        self.assertEqual(cfg.universe.min_market_age_days, 45)
        self.assertEqual(cfg.universe.min_dollar_volume_24h, 5_000_000.0)
        self.assertEqual(cfg.universe.max_spread_bps, 15.0)
        self.assertEqual(cfg.strategy.warmup_bars, 250)
        self.assertEqual(cfg.strategy.ema_trend, 200)
        self.assertEqual(cfg.risk.risk_per_trade_pct, 0.5)


if __name__ == "__main__":
    unittest.main()
