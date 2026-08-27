"""Cycle cadence and the closed-candle cache.

TWO REPORTED FACTS, ONE CAUSE
-----------------------------
A real Kraken/USD cycle over 11 symbols took **74.6 seconds** against a
configured `poll_seconds = 30`. That raised two separate questions.

1. WHAT DOES CONTINUOUS MODE DO WHEN A CYCLE OUTLASTS THE INTERVAL?
   `run()` is single-threaded and calls `cycle()` to completion before it looks
   at the clock, so cycles cannot overlap, cannot process the same candle twice,
   and cannot build a backlog -- there is no queue, and each cycle reads live
   state rather than replaying missed ones. Those properties were already
   sound and are pinned here so they stay that way.

   What was NOT sound: the pause was `max(0, poll_seconds - elapsed)`, which is
   exactly zero for an overrunning cycle. The bot ran flat out, back to back,
   with no gap between bursts of exchange requests and nothing anywhere saying
   it was behind. That is how a paper bot earns a rate-limit ban.

2. WHY 75 SECONDS FOR 11 SYMBOLS?
   44 OHLCV requests per cycle: 11 symbols x 2 timeframes x ~2 pages of history,
   re-downloading ~400 bars per series every 30 seconds. A closed candle is
   IMMUTABLE, so all but the last of those requests returned byte-identical data
   to the previous cycle.

   The cache turns "re-download everything, always" into "download only when a
   candle has actually closed". Its safety rests on one fact rather than on a
   time-to-live: while no newer candle has closed, the cached series IS the
   current state of the market.
"""
import types
import unittest

import helpers  # noqa: F401  -- silences the engine's log handlers
from crypto_edge.data.ccxt_feed import CCXTFeed
from crypto_edge.data.feed import DataUnavailable
from crypto_edge.engine import EngineStatus, TradingEngine
from crypto_edge.timeutils import last_closed_open_ms, tf_ms

HOUR = 3_600_000


# ------------------------------------------------------------------ pacing
def paced_engine(poll=30, floor=5.0):
    eng = TradingEngine.__new__(TradingEngine)
    eng.cfg = types.SimpleNamespace(
        engine=types.SimpleNamespace(poll_seconds=poll, min_pause_seconds=floor))
    eng._running = True
    eng.status = EngineStatus()
    eng.notifier = types.SimpleNamespace(send_error=lambda *a, **k: None)
    return eng


class Recorder:
    """Drives run() on a virtual clock so no test ever really sleeps."""

    def __init__(self, engine, cycle_seconds):
        self.eng = engine
        self.cycle_seconds = cycle_seconds
        self.clock = 1000.0
        self.starts, self.sleeps = [], []
        self.concurrent = 0
        self.max_concurrent = 0
        engine.cycle = self._cycle
        import crypto_edge.engine as mod
        self._mod = mod
        self._real_time = mod.time
        mod.time = types.SimpleNamespace(time=lambda: self.clock, sleep=self.sleep)

    def restore(self):
        self._mod.time = self._real_time

    def _cycle(self):
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        self.starts.append(self.clock)
        self.clock += (self.cycle_seconds() if callable(self.cycle_seconds)
                       else self.cycle_seconds)
        self.concurrent -= 1

    def sleep(self, s):
        self.sleeps.append(round(s, 3))
        self.clock += s

    def run(self, cycles):
        self.eng.run(max_cycles=cycles, sleep=self.sleep)
        return self

    @property
    def gaps(self):
        return [round(self.starts[i + 1] - self.starts[i], 1)
                for i in range(len(self.starts) - 1)]


class TestCyclesNeverOverlap(unittest.TestCase):
    def drive(self, cycle_s, cycles=5, poll=30, floor=5.0):
        r = Recorder(paced_engine(poll, floor), cycle_s)
        self.addCleanup(r.restore)
        return r.run(cycles)

    def test_a_slow_cycle_never_runs_concurrently_with_the_next(self):
        r = self.drive(74.6)
        self.assertEqual(r.max_concurrent, 1,
                         "run() is sequential; two cycles must never be live at once")

    def test_a_slow_cycle_does_not_build_a_backlog(self):
        """Five cycles requested, five cycles run -- not five plus catch-up."""
        r = self.drive(74.6, cycles=5)
        self.assertEqual(len(r.starts), 5)

    def test_an_overrunning_cycle_is_never_followed_by_a_zero_pause(self):
        """THE DEFECT: sleep(max(0, poll - elapsed)) == 0 for a 74.6s cycle."""
        r = self.drive(74.6)
        self.assertTrue(all(s > 0 for s in r.sleeps), r.sleeps)
        self.assertEqual(set(r.sleeps), {5.0})

    def test_the_realised_cadence_slows_instead_of_tightening(self):
        r = self.drive(74.6)
        for gap in r.gaps:
            self.assertAlmostEqual(gap, 79.6, places=1)
            self.assertGreater(gap, 30, "slower than the interval, not faster")

    def test_a_healthy_cycle_keeps_the_configured_interval(self):
        r = self.drive(8.0)
        for gap in r.gaps:
            self.assertAlmostEqual(gap, 30.0, places=1)

    def test_a_cycle_just_under_the_interval_still_gets_a_real_pause(self):
        """29.9s of work must not leave a 0.1s gap -- the floor applies always."""
        r = self.drive(29.9, floor=5.0)
        self.assertTrue(all(s >= 5.0 for s in r.sleeps), r.sleeps)

    def test_overruns_are_counted_and_reset(self):
        eng = paced_engine()
        self.assertEqual(eng.pause_after_cycle(74.6), 5.0)
        self.assertEqual(eng.pause_after_cycle(74.6), 5.0)
        self.assertEqual(eng.status.consecutive_overruns, 2)
        eng.pause_after_cycle(8.0)
        self.assertEqual(eng.status.consecutive_overruns, 0,
                         "recovery must clear the counter, not latch it")

    def test_the_next_cycle_time_is_recorded(self):
        """Observable WHILE running -- that is when an operator would look."""
        eng = paced_engine()
        r = Recorder(eng, 74.6)
        self.addCleanup(r.restore)
        seen = []
        real_sleep = r.sleep

        def watch(s):
            seen.append(eng.status.next_cycle_ms)
            real_sleep(s)

        eng.run(max_cycles=3, sleep=watch)
        self.assertTrue(seen and all(v > 0 for v in seen), seen)
        self.assertEqual(eng.status.next_cycle_ms, 0,
                         "and cleared once the run stops, not left dangling")

    def test_cycle_duration_is_recorded(self):
        r = self.drive(74.6, cycles=3)
        self.assertAlmostEqual(r.eng.status.last_cycle_seconds, 74.6, places=1)
        self.assertAlmostEqual(r.eng.status.slowest_cycle_seconds, 74.6, places=1)

    def test_a_failing_cycle_is_still_paced(self):
        """An exception must not turn the loop into a hot retry."""
        eng = paced_engine()
        r = Recorder(eng, 0.0)
        self.addCleanup(r.restore)

        def boom():
            r.starts.append(r.clock)
            raise RuntimeError("cycle failed")

        eng.cycle = boom
        eng.run(max_cycles=3, sleep=r.sleep)
        self.assertEqual(len(r.starts), 3)
        self.assertTrue(all(s >= 5.0 for s in r.sleeps), r.sleeps)

    def test_the_heartbeat_reports_the_cadence(self):
        from crypto_edge.notify import formatters as fmt
        msg = fmt.heartbeat(uptime_s=60, equity=10_000, today_pnl=0, total_pnl=0,
                            open_positions=0, btc_regime="RISK_ON", breadth=50,
                            signals_evaluated=11, last_data_ms=0, halted=False,
                            cycle_s=74.6, poll_s=30, overruns=3)
        self.assertIn("74.6s", msg)
        self.assertIn("BEHIND", msg)

    def test_a_healthy_heartbeat_does_not_cry_wolf(self):
        from crypto_edge.notify import formatters as fmt
        msg = fmt.heartbeat(uptime_s=60, equity=10_000, today_pnl=0, total_pnl=0,
                            open_positions=0, btc_regime="RISK_ON", breadth=50,
                            signals_evaluated=11, last_data_ms=0, halted=False,
                            cycle_s=8.0, poll_s=30, overruns=0)
        self.assertNotIn("BEHIND", msg)


# ------------------------------------------------------------------- cache
class CountingVenue:
    """Serves any timeframe, like a real venue, and counts requests."""

    def __init__(self, now_fn, step=HOUR, total=2000, cap=720):
        self.now_fn, self.step, self.cap = now_fn, step, cap
        self.calls = 0
        self.total = total
        self.fail = False

    def bars_for(self, step):
        newest = (self.now_fn() // step) * step
        return [[newest - (self.total - 1 - i) * step,
                 100.0, 101.0, 99.0, 100.5, 10.0] for i in range(self.total)]

    @property
    def bars(self):
        return self.bars_for(self.step)

    def fetch_ohlcv(self, symbol, timeframe=None, since=None, limit=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("venue unreachable")
        bars = self.bars_for(tf_ms(timeframe) if timeframe else self.step)
        rows = bars[-self.cap:] if since is None else \
            [r for r in bars if r[0] >= since][:self.cap]
        if limit is not None:
            rows = rows[:limit] if since is not None else rows[-limit:]
        return [list(r) for r in rows]


class CacheCase(unittest.TestCase):
    WANT = 405

    def setUp(self):
        import crypto_edge.data.ccxt_feed as mod
        self.mod = mod
        self._real_now = mod.now_ms
        self.addCleanup(lambda: setattr(mod, "now_ms", self._real_now))
        base = (self._real_now() // HOUR) * HOUR
        self.t = [base + 30 * 60_000]           # mid-bar, nothing pending
        mod.now_ms = lambda: self.t[0]
        self.venue = CountingVenue(lambda: self.t[0], HOUR)
        self.feed = self._feed(self.venue, cache_bars=2000)

    def _feed(self, venue, cache_bars):
        f = object.__new__(CCXTFeed)
        f.name, f.quote, f.rate_limit_ms = "kraken", "USD", 0
        f.client = venue
        f.page_limit = 300
        f.precision_mode = 4
        f.quote_ts_fallback = "local"
        f._markets = {}
        f.quote_ts_venue = f.quote_ts_local = 0
        f.close_buffer_ms = 20_000
        f.cache_bars = cache_bars
        f._ohlcv_cache = {}
        f.cache_hits = f.cache_refreshes = f.cache_bootstraps = 0
        f._sleep = lambda: None
        return f

    def get(self, tf="1h"):
        return self.feed.fetch_ohlcv("BTC/USD", tf, self.WANT)

    def closed(self, tf="1h"):
        return self.get(tf).drop_unclosed(self.t[0], 20_000)


class TestTheCacheAvoidsRedundantTraffic(CacheCase):
    def test_the_first_call_bootstraps_the_required_history(self):
        self.assertGreaterEqual(len(self.closed()), self.WANT)
        self.assertGreater(self.venue.calls, 0)

    def test_a_second_call_in_the_same_bar_makes_no_request_at_all(self):
        self.get()
        before = self.venue.calls
        self.get()
        self.assertEqual(self.venue.calls, before,
                         "a closed candle is immutable; refetching it is waste")

    def test_many_cycles_within_one_bar_cost_nothing(self):
        self.get()
        before = self.venue.calls
        for _ in range(20):                      # 10 minutes of 30s cycles
            self.t[0] += 30_000
            self.get()
        self.assertEqual(self.venue.calls, before)

    def test_a_new_candle_triggers_exactly_one_request(self):
        self.get()
        before = self.venue.calls
        self.t[0] += HOUR
        self.get()
        self.assertEqual(self.venue.calls - before, 1,
                         "one page refresh, not a full re-bootstrap")

    def test_the_refresh_actually_advances_the_series(self):
        first = int(self.closed().open_ms[-1])
        self.t[0] += HOUR
        second = int(self.closed().open_ms[-1])
        self.assertEqual(second - first, HOUR, "exactly one new closed bar")

    def test_depth_is_maintained_across_a_bar_boundary(self):
        self.get()
        for _ in range(4):
            self.t[0] += HOUR
            self.assertGreaterEqual(len(self.closed()), self.WANT)

    def test_disabling_the_cache_restores_full_refetching(self):
        feed = self._feed(CountingVenue(lambda: self.t[0], HOUR), cache_bars=0)
        feed.fetch_ohlcv("BTC/USD", "1h", self.WANT)
        before = feed.client.calls
        feed.fetch_ohlcv("BTC/USD", "1h", self.WANT)
        self.assertGreater(feed.client.calls, before,
                           "cache_bars=0 must genuinely disable the cache")

    def test_timeframes_are_cached_independently(self):
        self.get("1h")
        self.get("4h")
        stats = self.feed.ohlcv_cache_stats()
        self.assertEqual(stats["series"], 2)


class TestTheCacheCannotServeStaleDataAsCurrent(CacheCase):
    def test_the_cache_expires_the_instant_a_new_candle_closes(self):
        """Not a TTL: the deadline is the next close, computed exactly."""
        self.get()
        newest = last_closed_open_ms("1h", self.t[0], 20_000)
        before = self.venue.calls
        self.t[0] = newest + 2 * HOUR + 20_001    # one tick past the next close
        self.get()
        self.assertGreater(self.venue.calls, before,
                           "past the close it MUST go back to the venue")

    def test_a_refresh_failure_fails_closed_rather_than_serving_the_cache(self):
        self.get()
        self.t[0] += HOUR
        self.venue.fail = True
        with self.assertRaises(DataUnavailable):
            self.get()

    def test_a_bootstrap_failure_still_raises(self):
        self.venue.fail = True
        with self.assertRaises(DataUnavailable):
            self.get()

    def test_the_in_progress_candle_is_never_stored(self):
        self.get()
        cached = self.feed._ohlcv_cache[("BTC/USD", "1h")]
        newest_closed = last_closed_open_ms("1h", self.t[0], 20_000)
        self.assertLessEqual(cached[-1][0], newest_closed,
                             "a live bar in the cache is stale data by definition")

    def test_the_in_progress_candle_is_never_stored_after_a_refresh_either(self):
        """The refresh path merges a page that DOES contain the live bar.

        The bootstrap path filters it out; the refresh path has to filter it out
        again, separately. Testing only the bootstrap left the refresh free to
        store a live bar that a later cache hit would then serve as closed.
        """
        self.get()                       # bootstrap
        self.t[0] += HOUR
        self.get()                       # refresh -- merges a live bar
        cached = self.feed._ohlcv_cache[("BTC/USD", "1h")]
        newest_closed = last_closed_open_ms("1h", self.t[0], 20_000)
        self.assertLessEqual(cached[-1][0], newest_closed,
                             "the refresh must not cache the candle in progress")
        self.assertEqual(self.feed.cache_refreshes, 1, "the refresh path ran")

    def test_a_cache_hit_after_a_refresh_serves_only_closed_candles(self):
        self.get()
        self.t[0] += HOUR
        self.get()                       # refresh
        self.t[0] += 60_000
        raw = self.get()                 # hit, served from what the refresh stored
        self.assertEqual(
            len(raw), len(raw.drop_unclosed(self.t[0], 20_000)),
            "a cache hit must contain nothing that has not closed")

    def test_a_cache_hit_never_returns_an_unclosed_candle(self):
        self.get()
        self.t[0] += 60_000
        raw = self.get()
        newest_closed = last_closed_open_ms("1h", self.t[0], 20_000)
        self.assertLessEqual(int(raw.open_ms[-1]), newest_closed,
                             "no look-ahead can enter through the cache")

    def test_a_long_gap_rebootstraps_instead_of_splicing(self):
        """Away for days: the refresh page cannot join, so history is rebuilt."""
        self.get()
        before = self.venue.calls
        self.t[0] += 400 * HOUR
        closed = self.closed()
        self.assertGreater(self.venue.calls - before, 1, "a full rebuild")
        self.assertTrue(closed.is_sane(), "and no hole spliced into it")
        self.assertGreaterEqual(len(closed), self.WANT)

    def test_the_served_series_is_always_contiguous(self):
        for _ in range(6):
            self.assertTrue(self.closed().is_sane())
            self.t[0] += HOUR

    def test_asking_for_more_depth_than_is_cached_rebuilds(self):
        self.get()
        deeper = self.feed.fetch_ohlcv("BTC/USD", "1h", 900)
        self.assertGreaterEqual(len(deeper.drop_unclosed(self.t[0], 20_000)), 900)

    def test_a_restart_starts_from_a_cold_cache(self):
        self.get()
        fresh = self._feed(self.venue, cache_bars=2000)
        self.assertEqual(fresh.ohlcv_cache_stats()["series"], 0)
        self.assertGreaterEqual(
            len(fresh.fetch_ohlcv("BTC/USD", "1h", self.WANT)
                .drop_unclosed(self.t[0], 20_000)), self.WANT,
            "a cold start must rebuild full history, not run short")

    def test_the_cache_is_bounded(self):
        self.feed.cache_bars = 500
        self.feed._ohlcv_cache = {}
        self.feed.fetch_ohlcv("BTC/USD", "1h", 405)
        self.assertLessEqual(len(self.feed._ohlcv_cache[("BTC/USD", "1h")]), 500)

    def test_cached_and_uncached_series_agree(self):
        """The cache must be an optimisation, not a change in behaviour."""
        cached = self.closed()
        plain = self._feed(CountingVenue(lambda: self.t[0], HOUR), cache_bars=0)
        direct = plain.fetch_ohlcv("BTC/USD", "1h", self.WANT) \
                      .drop_unclosed(self.t[0], 20_000)
        self.assertEqual(int(cached.open_ms[-1]), int(direct.open_ms[-1]))
        self.assertEqual(len(cached), len(direct))


if __name__ == "__main__":
    unittest.main()
