"""Market age, established independently of indicator history.

WHAT WAS REPORTED
-----------------
On live Kraken, BTC/USDT and ETH/USDT both returned exactly 720 1h bars and were
rejected as "history truncated by venue". Nothing was eligible. Kraken caps
OHLCV at 720 candles, which at 1h is 30 days, and the age gate wanted 45.

THE INSIGHT
-----------
The cap is PER TIMEFRAME, not per market. The same 720 candles buy 30 days at
1h, 120 days at 4h, 720 days at 1d and over thirteen years at 1w. The venue can
answer "how old is this market" perfectly well -- it was simply being asked at
the one resolution where it cannot.

So indicator depth keeps asking at 1h, where it needs density, and age asks at
1d, where it needs reach. Neither threshold moved.

WHAT IS ENFORCED HERE
---------------------
  * age never comes from the span of the indicator window again
  * sources are tried in a defensible order and the one used is recorded
  * stored evidence only ever moves EARLIER, so a venue trimming its history
    cannot make a verified market look young again
  * an unverifiable market fails closed, and is reported differently from a
    market that is genuinely too new -- they are different facts
  * every verdict carries a timestamp, so a past decision can be reconstructed
"""
import unittest

import helpers  # noqa: F401  -- silences the engine's log handlers
from crypto_edge.data.feed import DataUnavailable
from crypto_edge.data.fixture_feed import make_series
from crypto_edge.data.market_age import (ASSET_HINT, CACHED_OBSERVATION,
                                         COARSE_OHLCV, EXCHANGE_METADATA,
                                         UNKNOWN, AgeVerdict, MarketAgeService,
                                         probe_bars_for)
from crypto_edge.models import MarketMeta
from crypto_edge.timeutils import now_ms
from helpers import open_repo, temp_repo

DAY = 86_400_000


def meta(symbol="BTC/USDT", created_ms=0):
    return MarketMeta(symbol, symbol.split("/")[0], "USDT", True,
                      4, 4, 0.0001, 10.0, 0.0001, 0.01, created_ms)


class DailyFeed:
    """A venue that serves daily candles, capped like Kraken."""

    def __init__(self, days=400, cap=720, fail=False):
        self.days = days
        self.cap = cap
        self.fail = fail
        self.calls = []

    def fetch_ohlcv(self, symbol, timeframe, limit):
        self.calls.append((symbol, timeframe, limit))
        if self.fail:
            raise DataUnavailable("no history endpoint")
        n = min(self.days, self.cap, limit)
        start = now_ms() - n * DAY
        return make_series(symbol, timeframe, [100.0] * n, start)


class Asset:
    def __init__(self, first_known_ms):
        self.first_known_ms = first_known_ms


class TestSourcePriority(unittest.TestCase):
    def setUp(self):
        self.repo, self.path = temp_repo()
        self.svc = MarketAgeService(self.repo, probe_timeframe="1d",
                                    probe_bars=400, cache_hours=0)

    def test_venue_metadata_wins_when_offered(self):
        created = now_ms() - 900 * DAY
        v = self.svc.age_of("BTC/USDT", meta=meta(created_ms=created),
                            feed=DailyFeed())
        self.assertEqual(v.source, EXCHANGE_METADATA)
        self.assertAlmostEqual(v.age_days, 900, delta=1)

    def test_coarse_ohlcv_is_used_when_the_venue_offers_no_listing_time(self):
        feed = DailyFeed(days=400)
        v = self.svc.age_of("BTC/USDT", meta=meta(), feed=feed)
        self.assertEqual(v.source, COARSE_OHLCV)
        self.assertAlmostEqual(v.age_days, 400, delta=2)
        self.assertEqual(feed.calls[0][1], "1d",
                         "age must be probed at the COARSE timeframe")

    def test_the_probe_never_asks_at_the_indicator_timeframe(self):
        feed = DailyFeed()
        self.svc.age_of("BTC/USDT", meta=meta(), feed=feed)
        self.assertNotIn("1h", [c[1] for c in feed.calls],
                         "asking at 1h is exactly the bug being fixed")

    def test_asset_hint_is_the_last_resort(self):
        v = self.svc.age_of("BTC/USDT", meta=meta(), feed=DailyFeed(fail=True),
                            asset=Asset(now_ms() - 1000 * DAY))
        self.assertEqual(v.source, ASSET_HINT)
        self.assertAlmostEqual(v.age_days, 1000, delta=1)
        self.assertIn("asset age, not listing age", v.detail,
                      "the weaker meaning of this source must be stated")

    def test_nothing_available_means_unknown_not_zero(self):
        v = self.svc.age_of("NEW/USDT", meta=meta(), feed=DailyFeed(fail=True))
        self.assertFalse(v.known)
        self.assertEqual(v.source, UNKNOWN)
        self.assertIsNone(v.age_days)

    def test_a_future_listing_timestamp_is_not_trusted(self):
        v = self.svc.age_of("X/USDT", meta=meta(created_ms=now_ms() + 10 * DAY),
                            feed=DailyFeed(fail=True))
        self.assertNotEqual(v.source, EXCHANGE_METADATA)


class TestTheKrakenCase(unittest.TestCase):
    """The reported failure, reproduced and then shown fixed."""

    def setUp(self):
        self.repo, self.path = temp_repo()

    def test_720_hourly_bars_cannot_evidence_45_days(self):
        """The precondition: why asking at 1h could never work."""
        span_days = 720 / 24
        self.assertLess(span_days, 45)

    def test_the_same_cap_at_1d_evidences_years(self):
        svc = MarketAgeService(self.repo, probe_timeframe="1d", probe_bars=400,
                               cache_hours=0)
        v = svc.age_of("BTC/USDT", meta=meta(), feed=DailyFeed(days=400, cap=720))
        self.assertTrue(v.meets(45), v.reason_if_blocked(45))
        self.assertGreater(v.age_days, 45 * 5)

    def test_a_mature_kraken_market_now_passes_the_age_gate(self):
        svc = MarketAgeService(self.repo, probe_timeframe="1d", probe_bars=400,
                               cache_hours=0)
        v = svc.age_of("ETH/USDT", meta=meta("ETH/USDT"), feed=DailyFeed(days=720))
        self.assertEqual(v.reason_if_blocked(45), "")

    def test_a_genuinely_new_listing_still_fails(self):
        svc = MarketAgeService(self.repo, probe_timeframe="1d", probe_bars=400,
                               cache_hours=0)
        v = svc.age_of("NEW/USDT", meta=meta("NEW/USDT"), feed=DailyFeed(days=10))
        self.assertFalse(v.meets(45))
        self.assertIn("too new", v.reason_if_blocked(45))

    def test_probe_bars_for_computes_enough_reach(self):
        self.assertGreaterEqual(probe_bars_for("1d", 45) * 1, 45)
        self.assertGreater(probe_bars_for("4h", 45), 45 * 6)


class TestUnverifiableIsNotYoung(unittest.TestCase):
    """Two different facts that must never be reported as one."""

    def setUp(self):
        self.repo, self.path = temp_repo()
        self.svc = MarketAgeService(self.repo, cache_hours=0)

    def test_unknown_age_blocks_the_entry(self):
        v = self.svc.age_of("X/USDT", meta=meta(), feed=DailyFeed(fail=True))
        self.assertNotEqual(v.reason_if_blocked(45), "", "must fail closed")

    def test_unknown_is_worded_differently_from_too_new(self):
        unknown = self.svc.age_of("X/USDT", meta=meta(), feed=DailyFeed(fail=True))
        young = self.svc.age_of("Y/USDT", meta=meta("Y/USDT"), feed=DailyFeed(days=5))
        self.assertIn("could not be established", unknown.reason_if_blocked(45))
        self.assertIn("too new", young.reason_if_blocked(45))
        self.assertNotIn("too new", unknown.reason_if_blocked(45))

    def test_a_known_old_market_is_not_blocked(self):
        v = self.svc.age_of("Z/USDT", meta=meta("Z/USDT"), feed=DailyFeed(days=300))
        self.assertEqual(v.reason_if_blocked(45), "")


class TestPersistenceAndMonotonicity(unittest.TestCase):
    def setUp(self):
        self.repo, self.path = temp_repo()

    def test_the_verdict_is_stored_with_its_provenance(self):
        svc = MarketAgeService(self.repo, cache_hours=0)
        svc.age_of("BTC/USDT", meta=meta(), feed=DailyFeed(days=300))
        row = self.repo.get_market_age("BTC/USDT")
        self.assertIsNotNone(row)
        self.assertEqual(row["source"], COARSE_OHLCV)
        self.assertGreater(row["observed_ms"], 0)
        self.assertGreater(row["first_ms"], 0)

    def test_evidence_only_ever_moves_earlier(self):
        """A venue that later serves less history must not un-age a market."""
        svc = MarketAgeService(self.repo, cache_hours=0)
        svc.age_of("BTC/USDT", meta=meta(), feed=DailyFeed(days=400))
        first_before = self.repo.get_market_age("BTC/USDT")["first_ms"]

        svc.age_of("BTC/USDT", meta=meta(), feed=DailyFeed(days=30))
        first_after = self.repo.get_market_age("BTC/USDT")["first_ms"]
        self.assertEqual(first_after, first_before,
                         "older evidence must win over a newer, shorter answer")

    def test_earlier_evidence_does_replace_later_evidence(self):
        """The counterpart of monotonicity: better evidence must be adopted.
        Both probes stay inside the service's 400-bar probe depth."""
        svc = MarketAgeService(self.repo, probe_bars=400, cache_hours=0)
        svc.age_of("BTC/USDT", meta=meta(), feed=DailyFeed(days=100))
        svc.age_of("BTC/USDT", meta=meta(), feed=DailyFeed(days=300))
        age = (now_ms() - self.repo.get_market_age("BTC/USDT")["first_ms"]) / DAY
        self.assertAlmostEqual(age, 300, delta=3)

    def test_the_probe_never_asks_for_more_than_its_configured_depth(self):
        svc = MarketAgeService(self.repo, probe_bars=400, cache_hours=0)
        feed = DailyFeed(days=5000)
        svc.age_of("BTC/USDT", meta=meta(), feed=feed)
        self.assertEqual(feed.calls[0][2], 400)

    def test_a_fresh_cache_avoids_re_probing(self):
        svc = MarketAgeService(self.repo, cache_hours=24)
        feed = DailyFeed(days=300)
        svc.age_of("BTC/USDT", meta=meta(), feed=feed)
        calls = len(feed.calls)
        v = svc.age_of("BTC/USDT", meta=meta(), feed=feed)
        self.assertEqual(len(feed.calls), calls, "age changes slowly; do not spam")
        self.assertTrue(v.known)

    def test_a_stored_answer_survives_a_restart(self):
        svc = MarketAgeService(self.repo, cache_hours=0)
        svc.age_of("BTC/USDT", meta=meta(), feed=DailyFeed(days=300))
        self.repo.conn.close()

        repo2 = open_repo(self.path)
        svc2 = MarketAgeService(repo2, cache_hours=0)
        v = svc2.age_of("BTC/USDT", meta=meta(), feed=DailyFeed(fail=True))
        self.assertEqual(v.source, CACHED_OBSERVATION)
        self.assertAlmostEqual(v.age_days, 300, delta=2)

    def test_the_cached_answer_rescues_a_venue_outage(self):
        svc = MarketAgeService(self.repo, cache_hours=0)
        svc.age_of("BTC/USDT", meta=meta(), feed=DailyFeed(days=300))
        v = svc.age_of("BTC/USDT", meta=meta(), feed=DailyFeed(fail=True))
        self.assertTrue(v.meets(45),
                        "an outage must not un-verify an already-aged market")

    def test_a_probe_failure_on_an_unknown_symbol_stores_nothing(self):
        svc = MarketAgeService(self.repo, cache_hours=0)
        svc.age_of("GHOST/USDT", meta=meta("GHOST/USDT"), feed=DailyFeed(fail=True))
        self.assertIsNone(self.repo.get_market_age("GHOST/USDT"))


class TestVerdictReporting(unittest.TestCase):
    def test_a_verdict_carries_when_it_was_established(self):
        v = AgeVerdict("X", 100.0, COARSE_OHLCV, now_ms() - 100 * DAY, now_ms())
        self.assertGreater(v.observed_ms, 0)
        self.assertIn("source", v.as_dict())
        self.assertIn("observed_ms", v.as_dict())

    def test_the_blocking_reason_names_the_source(self):
        v = AgeVerdict("X", 10.0, COARSE_OHLCV, 0, now_ms())
        self.assertIn(COARSE_OHLCV, v.reason_if_blocked(45))

    def test_meets_is_false_for_an_unknown_age(self):
        self.assertFalse(AgeVerdict("X", None, UNKNOWN).meets(0))

    def test_an_exception_in_the_probe_does_not_escape(self):
        repo, _ = temp_repo()

        class Exploding:
            def fetch_ohlcv(self, *a, **kw):
                raise RuntimeError("boom")

        v = MarketAgeService(repo, cache_hours=0).age_of(
            "X/USDT", meta=meta(), feed=Exploding())
        self.assertFalse(v.known)


if __name__ == "__main__":
    unittest.main()
