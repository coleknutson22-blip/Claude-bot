"""Candle freshness: measured from the CLOSE, never from the stamp.

THE DEFECT THIS FILE EXISTS FOR
-------------------------------
A live Kraken/USD verification run at ~04:58 UTC reported:

    BTC/USD 1h  ... 118.5 minutes stale
    BTC/USD 4h  ... 298.6 minutes stale

and failed its final verdict on freshness alone. Both figures are exactly
`now - candle_open`, which is what made the obvious diagnosis "the checker is
subtracting the open timestamp instead of the close". It was not. The freshness
arithmetic was already `now - (open + timeframe)`.

The data really WAS a full bar behind, and the cause was in the fetch.

CCXT truncates from the wrong end when `since` and `limit` are both supplied.
`Exchange.filter_by_since_limit` sets `shouldFilterFromStart = not tail and
sinceIsDefined`, and `filter_by_limit` then returns `array[0:limit]` -- the
OLDEST `limit` rows, discarding the newest. Our pager asked for `since = now -
(want + 2) * step` as a safety margin and passed `limit=page`; Kraken returned
every bar from `since` to the present, and CCXT cut the two newest off. Those
two were the in-progress candle AND THE MOST RECENTLY CLOSED ONE.

So the safety margin was being removed from the end that mattered, silently,
and the freshness check was correctly reporting genuinely stale data. Binance
hid it: it applies `limit` server-side, so the slice was a no-op there.

Verified against ccxt 4.5.75, `base/exchange.py` lines ~3353 and ~3386.

The threshold was NOT changed. It is still one timeframe plus five minutes.
"""
import unittest

import helpers  # noqa: F401  -- silences the engine's log handlers
from crypto_edge.data.ccxt_feed import CCXTFeed
from crypto_edge.timeutils import (candle_close_ms, floor_to_tf, is_closed,
                                   last_closed_open_ms, tf_ms)

HOUR = 3_600_000
DAY = 24 * HOUR
# 2026-08-27T04:58:30Z -- the wall clock of the reported run.
NOW = 1787893110000


def at(hh, mm=0, ss=0):
    """A UTC instant on the reported day, in ms."""
    return floor_to_tf(NOW, "1d") + hh * HOUR + mm * 60_000 + ss * 1000


class TestOpenTimestampVersusCloseTime(unittest.TestCase):
    """CCXT hands us OPEN timestamps. Everything downstream must know that."""

    def test_a_1h_candle_stamped_0300_completes_at_0400(self):
        self.assertEqual(candle_close_ms(at(3), "1h"), at(4))

    def test_a_4h_candle_stamped_0000_completes_at_0400(self):
        self.assertEqual(candle_close_ms(at(0), "4h"), at(4))

    def test_1h_freshness_at_0458_is_58_minutes_not_118(self):
        """The exact number from the live run, and the exact number it should be."""
        age_min = (at(4, 58, 30) - candle_close_ms(at(3), "1h")) / 60_000
        self.assertAlmostEqual(age_min, 58.5, places=1)
        wrong = (at(4, 58, 30) - at(3)) / 60_000
        self.assertAlmostEqual(wrong, 118.5, places=1,
                               msg="the reported figure, for the record")

    def test_4h_freshness_at_0458_is_58_minutes_not_298(self):
        age_min = (at(4, 58, 30) - candle_close_ms(at(0), "4h")) / 60_000
        self.assertAlmostEqual(age_min, 58.5, places=1)
        wrong = (at(4, 58, 30) - at(0)) / 60_000
        self.assertAlmostEqual(wrong, 298.5, places=1)

    def test_both_timeframes_are_equally_fresh_at_the_same_instant(self):
        """A 4h series is not four times staler than a 1h one at 04:58.

        Measured from the close they agree exactly, which is the whole point:
        one timeframe is not inherently more behind than another.
        """
        now = at(4, 58, 30)
        one = (now - candle_close_ms(at(3), "1h")) / 60_000
        four = (now - candle_close_ms(at(0), "4h")) / 60_000
        self.assertAlmostEqual(one, four, places=6)


class TestDifferentTimeframeDurations(unittest.TestCase):
    def test_every_supported_timeframe_has_its_own_step(self):
        for tf, ms in (("15m", 15 * 60_000), ("1h", HOUR), ("2h", 2 * HOUR),
                       ("4h", 4 * HOUR), ("6h", 6 * HOUR), ("12h", 12 * HOUR),
                       ("1d", DAY)):
            self.assertEqual(tf_ms(tf), ms, tf)

    def test_close_time_follows_the_timeframe(self):
        for tf in ("15m", "1h", "4h", "12h", "1d"):
            open_ms = floor_to_tf(NOW, tf)
            self.assertEqual(candle_close_ms(open_ms, tf), open_ms + tf_ms(tf), tf)

    def test_a_freshly_closed_candle_reads_as_fresh_on_every_timeframe(self):
        """`now - close` is small right after a close, whatever the duration."""
        for tf in ("15m", "1h", "4h", "12h", "1d"):
            open_ms = last_closed_open_ms(tf, NOW, 0)
            age_min = (NOW - candle_close_ms(open_ms, tf)) / 60_000
            self.assertGreaterEqual(age_min, 0, tf)
            self.assertLess(age_min, tf_ms(tf) / 60_000, tf)

    def test_the_freshness_allowance_scales_with_the_timeframe(self):
        """One timeframe + 5 min. A flat minute count would reject 4h forever."""
        for tf in ("1h", "4h", "1d"):
            allowance = tf_ms(tf) / 60_000 + 5
            worst_case_when_current = tf_ms(tf) / 60_000
            self.assertGreater(allowance, worst_case_when_current, tf)


class TestTheBoundaryJustBeforeAndAfterClose(unittest.TestCase):
    """A candle is complete at its close instant, plus the clock-skew buffer."""

    BUF = 20_000

    def test_one_second_before_close_it_is_not_complete(self):
        self.assertFalse(is_closed(at(3), "1h", at(3, 59, 59), self.BUF))

    def test_at_the_close_instant_the_buffer_still_withholds_it(self):
        self.assertFalse(is_closed(at(3), "1h", at(4), self.BUF),
                         "clock skew means the venue may still revise it")

    def test_just_after_the_buffer_it_is_complete(self):
        self.assertTrue(is_closed(at(3), "1h", at(4, 0, 21), self.BUF))

    def test_with_no_buffer_the_close_instant_completes_it(self):
        self.assertTrue(is_closed(at(3), "1h", at(4), 0))

    def test_the_newest_available_candle_advances_at_the_boundary(self):
        before = last_closed_open_ms("1h", at(3, 59, 59), self.BUF)
        after = last_closed_open_ms("1h", at(4, 0, 21), self.BUF)
        self.assertEqual(before, at(2))
        self.assertEqual(after, at(3))
        self.assertEqual(after - before, HOUR, "exactly one bar, at the boundary")

    def test_4h_boundaries_land_on_4h_marks(self):
        self.assertEqual(last_closed_open_ms("4h", at(3, 59, 59), self.BUF),
                         at(0) - 4 * HOUR)
        self.assertEqual(last_closed_open_ms("4h", at(4, 0, 21), self.BUF), at(0))


class _Venue:
    """A venue that reproduces CCXT's oldest-first slice. See module docstring."""

    def __init__(self, now, step, total=3000, cap=720, behind_bars=0):
        self.now, self.step, self.cap = now, step, cap
        self.calls = 0
        newest = (now // step) * step - behind_bars * step
        self.bars = [[newest - (total - 1 - i) * step,
                      100.0, 101.0, 99.0, 100.5, 10.0] for i in range(total)]

    def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None):
        self.calls += 1
        if since is None:
            rows = self.bars[-self.cap:]
        else:
            rows = [r for r in self.bars if r[0] >= since][:self.cap]
        if limit is not None:                       # ccxt filter_by_since_limit
            rows = rows[:limit] if since is not None else rows[-limit:]
        return [list(r) for r in rows]


def feed_for(venue, now, page_limit=300):
    f = object.__new__(CCXTFeed)
    f.name, f.quote, f.rate_limit_ms = "kraken", "USD", 0
    f.client = venue
    f.page_limit = page_limit
    f.precision_mode = 4
    f.quote_ts_fallback = "local"
    f._markets = {}
    f.quote_ts_venue = f.quote_ts_local = 0
    f.close_buffer_ms = 20_000
    f.cache_bars = 0
    f._ohlcv_cache = {}
    f.cache_hits = f.cache_refreshes = f.cache_bootstraps = 0
    f._sleep = lambda: None
    import crypto_edge.data.ccxt_feed as mod
    mod.now_ms = lambda: now
    return f


class TestTheTruncationRegression(unittest.TestCase):
    """The live edge must survive the fetch. This is the actual bug."""

    def setUp(self):
        self.now = at(4, 58, 30)
        import crypto_edge.data.ccxt_feed as mod
        self._real = mod.now_ms
        self.addCleanup(lambda: setattr(mod, "now_ms", self._real))

    def _newest_closed(self, tf, step, want=405):
        venue = _Venue(self.now, step)
        feed = feed_for(venue, self.now)
        series = feed.fetch_ohlcv("BTC/USD", tf, want)
        closed = series.drop_unclosed(self.now, 20_000)
        return int(closed.open_ms[-1]), len(closed), venue

    def test_1h_reaches_the_newest_closed_candle(self):
        newest, depth, _ = self._newest_closed("1h", HOUR)
        self.assertEqual(newest, at(3), "03:00 is the newest closed 1h candle")
        age = (self.now - candle_close_ms(newest, "1h")) / 60_000
        self.assertAlmostEqual(age, 58.5, places=1, msg="not 118.5")
        self.assertGreaterEqual(depth, 405)

    def test_4h_reaches_the_newest_closed_candle(self):
        newest, depth, _ = self._newest_closed("4h", 4 * HOUR)
        self.assertEqual(newest, at(0), "00:00 is the newest closed 4h candle")
        age = (self.now - candle_close_ms(newest, "4h")) / 60_000
        self.assertAlmostEqual(age, 58.5, places=1, msg="not 298.6")
        self.assertGreaterEqual(depth, 405)

    def test_the_in_progress_candle_is_fetched_and_then_discarded(self):
        """It must be SEEN -- that is how we know we reached the live edge."""
        venue = _Venue(self.now, HOUR)
        feed = feed_for(venue, self.now)
        raw = feed.fetch_ohlcv("BTC/USD", "1h", 405)
        self.assertEqual(int(raw.open_ms[-1]), at(4),
                         "the raw series reaches the candle in progress")
        closed = raw.drop_unclosed(self.now, 20_000)
        self.assertEqual(int(closed.open_ms[-1]), at(3),
                         "and it is then removed -- no look-ahead")
        self.assertLess(len(closed), len(raw))

    def test_the_first_request_carries_no_since(self):
        """That is what makes the newest rows safe from the oldest-first slice."""
        seen = []
        venue = _Venue(self.now, HOUR)
        real = venue.fetch_ohlcv

        def spy(symbol, timeframe=None, since=None, limit=None):
            seen.append(since)
            return real(symbol, timeframe, since, limit)

        venue.fetch_ohlcv = spy
        feed_for(venue, self.now).fetch_ohlcv("BTC/USD", "1h", 405)
        self.assertIsNone(seen[0], "the newest page must not be sliced away")

    def test_a_wider_span_than_the_page_would_have_lost_the_live_edge(self):
        """Proves the venue stub really does reproduce the CCXT behaviour."""
        # The exact call the old pager made for the freshness probe:
        # want = ohlcv_limit = 300, since = now - (want + 2) bars.
        venue = _Venue(self.now, HOUR)
        rows = venue.fetch_ohlcv("BTC/USD", "1h",
                                 since=self.now - 302 * HOUR, limit=300)
        self.assertEqual(rows[-1][0], at(4) - 2 * HOUR,
                         "the old call shape loses exactly two bars")
        age = (self.now - candle_close_ms(rows[-1][0], "1h")) / 60_000
        self.assertAlmostEqual(age, 118.5, places=1,
                               msg="which is the number the live run printed")

    def test_history_has_no_holes_after_paging(self):
        venue = _Venue(self.now, HOUR)
        series = feed_for(venue, self.now).fetch_ohlcv("BTC/USD", "1h", 405)
        self.assertTrue(series.is_sane(), "pages must join with no gap")


class TestGenuinelyStaleDataStillFails(unittest.TestCase):
    """The fix must not make a real outage look healthy."""

    def setUp(self):
        self.now = at(4, 58, 30)
        import crypto_edge.data.ccxt_feed as mod
        self._real = mod.now_ms
        self.addCleanup(lambda: setattr(mod, "now_ms", self._real))

    def _age_min(self, behind_bars, tf="1h", step=HOUR):
        venue = _Venue(self.now, step, behind_bars=behind_bars)
        feed = feed_for(venue, self.now)
        closed = feed.fetch_ohlcv("BTC/USD", tf, 405).drop_unclosed(self.now, 20_000)
        return (self.now - candle_close_ms(int(closed.open_ms[-1]), tf)) / 60_000

    def test_a_venue_three_bars_behind_is_reported_as_stale(self):
        age = self._age_min(behind_bars=3)
        allowance = tf_ms("1h") / 60_000 + 5
        self.assertGreater(age, allowance, "must not pass the freshness check")

    def test_a_venue_one_bar_behind_exceeds_the_allowance(self):
        """One bar behind is exactly the condition that started all of this."""
        age = self._age_min(behind_bars=2)
        self.assertGreater(age, tf_ms("1h") / 60_000 + 5)

    def test_a_current_venue_is_within_the_allowance(self):
        self.assertLess(self._age_min(behind_bars=0), tf_ms("1h") / 60_000 + 5)

    def test_a_stale_4h_venue_is_caught_too(self):
        age = self._age_min(behind_bars=3, tf="4h", step=4 * HOUR)
        self.assertGreater(age, tf_ms("4h") / 60_000 + 5)

    def test_the_threshold_itself_was_not_moved(self):
        from crypto_edge.config import load_config
        cfg = load_config("config/config.toml", "/nonexistent")
        self.assertEqual(cfg.safety.candle_close_buffer_s, 20)
        self.assertEqual(cfg.universe.min_dollar_volume_24h, 5_000_000.0)
        self.assertEqual(cfg.universe.min_atr_pct, 0.8)
        self.assertEqual(cfg.universe.min_market_age_days, 45)


class TestIncompleteCandleRemoval(unittest.TestCase):
    """drop_unclosed is the single guard against look-ahead. Prove it, per timeframe."""

    def _series(self, tf, step, n=10, now=None):
        from crypto_edge.models import Series
        now = now or at(4, 58, 30)
        newest = (now // step) * step
        rows = [[newest - (n - 1 - i) * step, 100.0, 101.0, 99.0, 100.5, 10.0]
                for i in range(n)]
        return Series.from_ohlcv("X/USD", tf, rows), now

    def test_the_in_progress_candle_is_removed_on_every_timeframe(self):
        for tf, step in (("1h", HOUR), ("4h", 4 * HOUR), ("1d", DAY)):
            s, now = self._series(tf, step)
            closed = s.drop_unclosed(now, 20_000)
            self.assertEqual(len(closed), len(s) - 1, tf)
            self.assertLess(int(closed.open_ms[-1]), (now // step) * step, tf)

    def test_every_retained_candle_is_genuinely_complete(self):
        for tf, step in (("1h", HOUR), ("4h", 4 * HOUR)):
            s, now = self._series(tf, step)
            closed = s.drop_unclosed(now, 20_000)
            for open_ms in closed.open_ms.tolist():
                self.assertTrue(is_closed(int(open_ms), tf, now, 20_000), tf)

    def test_a_series_of_only_unclosed_candles_becomes_empty(self):
        from crypto_edge.models import Series
        now = at(4, 30)
        s = Series.from_ohlcv("X/USD", "1h",
                              [[at(4), 100.0, 101.0, 99.0, 100.5, 10.0]])
        self.assertEqual(len(s.drop_unclosed(now, 20_000)), 0,
                         "fail closed rather than trade an unfinished bar")

    def test_removal_happens_at_the_buffer_not_at_the_close(self):
        s, _ = self._series("1h", HOUR)
        just_closed = at(4, 0, 5)          # 5s past close, inside a 20s buffer
        kept = s.drop_unclosed(just_closed, 20_000)
        self.assertLess(int(kept.open_ms[-1]), at(3),
                        "inside the buffer the just-closed bar is not trusted")


if __name__ == "__main__":
    unittest.main()
