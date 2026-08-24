"""CCXT adapter correctness, verified against recorded CCXT response shapes.

DEFECTS UNDER TEST
------------------
1. **Precision was silently wrong on every major venue.** CCXT reports market
   granularity in three modes. `TICK_SIZE` -- which binance, kraken, coinbase,
   kucoin, okx, bybit and bitstamp all use in current CCXT -- gives an absolute
   tick as a *float* (0.00001). The adapter tested `isinstance(raw, int)`, which
   is False for a float, and fell back to 8 decimal places. A venue permitting
   5 decimals of quantity was therefore sized to 8, and one with a 0.05 tick
   could not be expressed at all.

2. **A missing ticker timestamp was silently replaced with local time.** CCXT
   normalises an absent venue timestamp to None; substituting `now_ms()` without
   saying so makes a quote of unknown vintage look 0 seconds old to the entry
   staleness check -- laundering an unknown into a pass.

These tests construct the adapter without touching ccxt or the network, by
exercising the pure parsing functions and by bypassing __init__ where a feed
instance is needed. Live behaviour is verified separately with
`python -m crypto_edge.cli verify-live`.
"""
import unittest

import helpers  # noqa: F401  -- silences the engine's log handlers
from crypto_edge.data.ccxt_feed import (DECIMAL_PLACES, SIGNIFICANT_DIGITS,
                                        TICK_SIZE, CCXTFeed, _precision_fields,
                                        _resolve_quote_ts, decimals_from_tick)
from crypto_edge.execution.paper_broker import (PaperBroker, decimals_for_step,
                                                round_amount, round_price)
from crypto_edge.models import TS_LOCAL, TS_VENUE, MarketMeta, Quote
from crypto_edge.timeutils import now_ms


class TestPrecisionParsing(unittest.TestCase):
    def test_tick_size_mode_is_no_longer_mistaken_for_missing(self):
        """The exact defect: a float tick used to fall through to 8 decimals."""
        dp, step = _precision_fields(0.00001, TICK_SIZE)
        self.assertEqual(step, 0.00001)
        self.assertEqual(dp, 5, "0.00001 is five decimal places, not eight")

    def test_tick_size_price_precision(self):
        dp, step = _precision_fields(0.01, TICK_SIZE)
        self.assertEqual((dp, step), (2, 0.01))

    def test_decimal_places_mode_still_works(self):
        dp, step = _precision_fields(4, DECIMAL_PLACES)
        self.assertEqual((dp, step), (4, 0.0))

    def test_significant_digits_mode_is_read_as_a_count(self):
        dp, step = _precision_fields(6, SIGNIFICANT_DIGITS)
        self.assertEqual((dp, step), (6, 0.0))

    def test_missing_precision_falls_back_safely(self):
        self.assertEqual(_precision_fields(None, TICK_SIZE), (8, 0.0))
        self.assertEqual(_precision_fields("nonsense", TICK_SIZE), (8, 0.0))

    def test_absurd_ticks_are_refused(self):
        for bad in (0.0, -1.0, float("inf"), float("nan")):
            self.assertEqual(_precision_fields(bad, TICK_SIZE), (8, 0.0))

    def test_a_float_tick_declared_under_decimal_places_is_still_handled(self):
        """Some venues declare DECIMAL_PLACES and hand back a tick anyway."""
        dp, step = _precision_fields(0.001, DECIMAL_PLACES)
        self.assertEqual((dp, step), (3, 0.001))

    def test_whole_unit_and_coarse_ticks(self):
        self.assertEqual(_precision_fields(1.0, TICK_SIZE), (0, 1.0))
        self.assertEqual(_precision_fields(10.0, TICK_SIZE), (0, 10.0))

    def test_decimals_from_tick(self):
        for tick, expected in ((0.001, 3), (0.05, 2), (1.0, 0), (1e-8, 8)):
            self.assertEqual(decimals_from_tick(tick), expected, f"tick={tick}")


class TestTickRounding(unittest.TestCase):
    """A tick is more expressive than a decimal count and must take priority."""

    def test_amount_rounds_down_to_the_tick(self):
        self.assertAlmostEqual(round_amount(0.123456789, 8, step=0.001), 0.123, places=12)
        self.assertAlmostEqual(round_amount(7.9, 8, step=1.0), 7.0, places=12)

    def test_non_power_of_ten_ticks_are_honoured(self):
        """0.05 cannot be expressed as a number of decimal places at all."""
        self.assertAlmostEqual(round_amount(0.17, 8, step=0.05), 0.15, places=12)
        self.assertAlmostEqual(round_amount(0.19999, 8, step=0.05), 0.15, places=12)
        self.assertAlmostEqual(round_amount(0.20001, 8, step=0.05), 0.20, places=12)

    def test_amount_never_rounds_up(self):
        for qty in (0.0999, 1.4999, 9.9999):
            for step in (0.001, 0.01, 0.1, 0.5, 1.0):
                self.assertLessEqual(round_amount(qty, 8, step=step), qty + 1e-12,
                                     f"qty={qty} step={step}")

    def test_price_rounds_to_the_nearest_tick(self):
        self.assertAlmostEqual(round_price(100.024, 8, step=0.01), 100.02, places=12)
        self.assertAlmostEqual(round_price(100.026, 8, step=0.01), 100.03, places=12)
        self.assertAlmostEqual(round_price(102.4, 8, step=0.5), 102.5, places=12)

    def test_results_carry_no_binary_float_dust(self):
        """A ledger must not accumulate 0.30000000000000004."""
        for step in (0.01, 0.1, 0.05, 0.001):
            v = round_amount(0.3, 8, step=step)
            self.assertEqual(v, round(v, decimals_for_step(step)))

    def test_decimal_precision_is_used_when_no_tick_is_given(self):
        self.assertAlmostEqual(round_amount(0.123456789, 3), 0.123, places=12)
        self.assertAlmostEqual(round_price(100.0249, 2), 100.02, places=12)

    def test_sizing_honours_the_tick(self):
        b = PaperBroker(7.5, 6.0, 15.0, use_book_spread=False)
        meta = MarketMeta("X/USDT", "X", "USDT", True, amount_precision=8,
                          price_precision=8, min_amount=0.0, min_cost=0.0,
                          amount_step=0.01, price_step=0.01)
        r = b.size_position(equity=10_000.0, cash=1e9, entry_price=100.0,
                            stop_price=95.0, risk_pct=0.5, max_position_pct=100.0,
                            current_exposure=0.0, max_exposure_pct=100.0, meta=meta)
        self.assertTrue(r.ok, r.reason)
        self.assertAlmostEqual(r.qty, round(r.qty, 2), places=12,
                               msg="quantity must land on the venue's tick")
        self.assertLessEqual(r.qty * 5.0, 50.0 + 1e-9, "risk budget still respected")


class TestQuoteTimestampResolution(unittest.TestCase):
    def test_a_plausible_venue_timestamp_is_trusted(self):
        ts = now_ms() - 1500
        got, source = _resolve_quote_ts(ts)
        self.assertEqual((got, source), (ts, TS_VENUE))

    def test_a_missing_timestamp_is_flagged_not_faked(self):
        got, source = _resolve_quote_ts(None)
        self.assertEqual(source, TS_LOCAL)
        self.assertAlmostEqual(got, now_ms(), delta=2000)

    def test_zero_and_negative_are_treated_as_absent(self):
        for bad in (0, -1):
            self.assertEqual(_resolve_quote_ts(bad)[1], TS_LOCAL)

    def test_non_numeric_is_treated_as_absent(self):
        for bad in ("", "abc", object()):
            self.assertEqual(_resolve_quote_ts(bad)[1], TS_LOCAL)

    def test_a_seconds_instead_of_milliseconds_timestamp_is_rejected(self):
        """A venue sending epoch SECONDS would otherwise look ~55 years stale."""
        seconds = int(now_ms() / 1000)
        got, source = _resolve_quote_ts(seconds)
        self.assertEqual(source, TS_LOCAL)
        self.assertAlmostEqual(got, now_ms(), delta=2000)

    def test_a_wildly_future_timestamp_is_rejected(self):
        self.assertEqual(_resolve_quote_ts(now_ms() + 5 * 86_400_000)[1], TS_LOCAL)


class _Bare(CCXTFeed):
    """A CCXTFeed with no ccxt client, for testing pure parsing paths."""

    def __init__(self, client, mode=TICK_SIZE, fallback="local"):
        self.name, self.quote, self.rate_limit_ms = "test", "USDT", 0
        self.client = client
        self.precision_mode = mode
        self.quote_ts_fallback = fallback
        self._markets = {}
        self.quote_ts_venue = self.quote_ts_local = 0


class FakeClient:
    precisionMode = TICK_SIZE

    def __init__(self, markets=None, ticker=None, ohlcv=None):
        self._markets = markets or {}
        self._ticker = ticker or {}
        self._ohlcv = ohlcv or []
        self.has = {"fetchTickers": True, "fetchTime": False}

    def load_markets(self):
        return self._markets

    def fetch_ticker(self, symbol):
        return dict(self._ticker)

    def fetch_ohlcv(self, symbol, timeframe, limit):
        return list(self._ohlcv)


def spot_market(symbol="BTC/USDT", amount=0.00001, price=0.01, active=True):
    base, quote = symbol.split("/")
    return {"symbol": symbol, "base": base, "quote": quote, "spot": True,
            "active": active, "precision": {"amount": amount, "price": price},
            "limits": {"amount": {"min": 0.00001}, "cost": {"min": 5.0}}}


class TestLoadMarkets(unittest.TestCase):
    def test_tick_sizes_reach_market_meta(self):
        feed = _Bare(FakeClient(markets={"BTC/USDT": spot_market()}))
        m = feed.load_markets()["BTC/USDT"]
        self.assertEqual(m.amount_step, 0.00001)
        self.assertEqual(m.price_step, 0.01)
        self.assertEqual(m.amount_precision, 5)
        self.assertEqual(m.price_precision, 2)
        self.assertEqual(m.min_amount, 0.00001)
        self.assertEqual(m.min_cost, 5.0)

    def test_non_spot_and_other_quotes_are_skipped(self):
        markets = {"BTC/USDT": spot_market(),
                   "ETH/EUR": spot_market("ETH/EUR"),
                   "BTC/USDT:USDT": {**spot_market(), "spot": False}}
        feed = _Bare(FakeClient(markets=markets))
        self.assertEqual(list(feed.load_markets()), ["BTC/USDT"])

    def test_a_venue_with_no_matching_markets_fails_loudly(self):
        from crypto_edge.data.feed import DataUnavailable
        feed = _Bare(FakeClient(markets={"ETH/EUR": spot_market("ETH/EUR")}))
        with self.assertRaises(DataUnavailable):
            feed.load_markets()

    def test_missing_limits_do_not_crash(self):
        m = spot_market()
        m["limits"] = {}
        feed = _Bare(FakeClient(markets={"BTC/USDT": m}))
        meta = feed.load_markets()["BTC/USDT"]
        self.assertEqual((meta.min_amount, meta.min_cost), (0.0, 0.0))


class TestFetchQuote(unittest.TestCase):
    def test_venue_timestamp_is_preserved_and_labelled(self):
        ts = now_ms() - 800
        feed = _Bare(FakeClient(ticker={"bid": 100.0, "ask": 100.1, "last": 100.05,
                                        "timestamp": ts}))
        q = feed.fetch_quote("BTC/USDT")
        self.assertEqual((q.ts_ms, q.ts_source), (ts, TS_VENUE))
        self.assertTrue(q.venue_timestamped)
        self.assertEqual(feed.quote_ts_venue, 1)

    def test_missing_timestamp_is_labelled_local_not_passed_off_as_venue(self):
        feed = _Bare(FakeClient(ticker={"bid": 100.0, "ask": 100.1, "last": 100.05,
                                        "timestamp": None}))
        q = feed.fetch_quote("BTC/USDT")
        self.assertEqual(q.ts_source, TS_LOCAL)
        self.assertFalse(q.venue_timestamped)
        self.assertEqual(feed.quote_ts_local, 1)

    def test_reject_policy_discards_an_unstamped_quote(self):
        feed = _Bare(FakeClient(ticker={"bid": 100.0, "ask": 100.1, "last": 100.05,
                                        "timestamp": None}), fallback="reject")
        self.assertIsNone(feed.fetch_quote("BTC/USDT"))

    def test_mid_is_used_when_last_is_absent(self):
        feed = _Bare(FakeClient(ticker={"bid": 100.0, "ask": 100.2,
                                        "timestamp": now_ms()}))
        self.assertAlmostEqual(feed.fetch_quote("BTC/USDT").last, 100.1, places=9)

    def test_an_unusable_ticker_yields_no_quote(self):
        feed = _Bare(FakeClient(ticker={"bid": None, "ask": None, "last": None}))
        self.assertIsNone(feed.fetch_quote("BTC/USDT"))

    def test_stats_report_the_timestamp_mix(self):
        feed = _Bare(FakeClient(ticker={"bid": 1.0, "ask": 1.1, "last": 1.05,
                                        "timestamp": now_ms()}))
        feed.fetch_quote("X")
        feed.client._ticker["timestamp"] = None
        feed.fetch_quote("X")
        stats = feed.quote_ts_stats()
        self.assertEqual((stats["venue_stamped"], stats["locally_stamped"]), (1, 1))
        self.assertAlmostEqual(stats["venue_pct"], 50.0, places=6)


class TestFetchOhlcv(unittest.TestCase):
    def _rows(self, n=5):
        base = now_ms() - n * 3_600_000
        return [[base + i * 3_600_000, 100.0, 101.0, 99.0, 100.5, 10.0]
                for i in range(n)]

    def test_normal_rows_parse(self):
        feed = _Bare(FakeClient(ohlcv=self._rows()))
        s = feed.fetch_ohlcv("BTC/USDT", "1h", 100)
        self.assertEqual(len(s), 5)
        self.assertTrue(s.is_sane())

    def test_rows_containing_none_are_dropped_not_propagated_as_nan(self):
        rows = self._rows()
        rows[2][3] = None
        feed = _Bare(FakeClient(ohlcv=rows))
        s = feed.fetch_ohlcv("BTC/USDT", "1h", 100)
        self.assertEqual(len(s), 4, "the malformed row must be removed")

    def test_short_rows_are_dropped(self):
        rows = self._rows()
        rows[1] = [rows[1][0], 100.0]
        feed = _Bare(FakeClient(ohlcv=rows))
        self.assertEqual(len(feed.fetch_ohlcv("BTC/USDT", "1h", 100)), 4)

    def test_extra_columns_are_ignored(self):
        rows = [r + ["junk"] for r in self._rows()]
        feed = _Bare(FakeClient(ohlcv=rows))
        self.assertEqual(len(feed.fetch_ohlcv("BTC/USDT", "1h", 100)), 5)

    def test_empty_and_all_malformed_raise_data_unavailable(self):
        from crypto_edge.data.feed import DataUnavailable
        with self.assertRaises(DataUnavailable):
            _Bare(FakeClient(ohlcv=[])).fetch_ohlcv("BTC/USDT", "1h", 100)
        with self.assertRaises(DataUnavailable):
            _Bare(FakeClient(ohlcv=[[None] * 6])).fetch_ohlcv("BTC/USDT", "1h", 100)

    def test_unsupported_fetch_tickers_is_reported_clearly(self):
        from crypto_edge.data.feed import DataUnavailable
        feed = _Bare(FakeClient())
        feed.client.has["fetchTickers"] = False
        with self.assertRaises(DataUnavailable) as ctx:
            feed.fetch_tickers()
        self.assertIn("fetchTickers", str(ctx.exception))


class TestEntryGateTimestampPolicy(unittest.TestCase):
    """A locally-stamped quote must not be mistaken for a verified-fresh one."""

    def setUp(self):
        self.b = PaperBroker(0.0, 0.0, 0.0, use_book_spread=True,
                             max_spread_bps_entry=25.0)

    def _local(self, age_ms=0):
        return Quote("X/USDT", 99.95, 100.05, 100.0, now_ms() - age_ms,
                     ts_source=TS_LOCAL)

    def test_locally_stamped_quote_is_accepted_but_marked_unverified(self):
        r = self.b.validate_entry_quote(self._local(), ref_price=100.0)
        self.assertTrue(r.ok, r.reason)
        self.assertFalse(r.age_verified,
                         "we must not claim to have verified freshness we cannot")
        self.assertEqual(r.ts_source, TS_LOCAL)
        self.assertEqual(r.age_s, 0.0)

    def test_the_age_check_is_skipped_rather_than_silently_passed(self):
        """An old LOCAL stamp is a clock artefact, not evidence of staleness --
        but neither is it evidence of freshness, so it must not be scored."""
        r = self.b.validate_entry_quote(self._local(age_ms=10 * 60_000),
                                        ref_price=100.0, max_age_s=90)
        self.assertTrue(r.ok)
        self.assertFalse(r.age_verified)

    def test_policy_can_demand_a_venue_timestamp(self):
        r = self.b.validate_entry_quote(self._local(), ref_price=100.0,
                                        require_venue_timestamp=True)
        self.assertFalse(r.ok)
        self.assertIn("no quote timestamp", r.reason)

    def test_venue_stamped_quotes_are_still_age_checked(self):
        stale = Quote("X/USDT", 99.95, 100.05, 100.0, now_ms() - 600_000,
                      ts_source=TS_VENUE)
        r = self.b.validate_entry_quote(stale, ref_price=100.0, max_age_s=90)
        self.assertFalse(r.ok)
        self.assertIn("stale", r.reason)
        self.assertTrue(r.age_verified)

    def test_deviation_check_still_protects_an_unstamped_venue(self):
        """This is what carries the load when the age check cannot run."""
        far = Quote("X/USDT", 139.9, 140.1, 140.0, now_ms(), ts_source=TS_LOCAL)
        r = self.b.validate_entry_quote(far, ref_price=100.0, max_deviation_pct=10.0)
        self.assertFalse(r.ok)
        self.assertIn("deviates", r.reason)

    def test_spread_check_still_applies_to_an_unstamped_quote(self):
        wide = Quote("X/USDT", 99.0, 101.0, 100.0, now_ms(), ts_source=TS_LOCAL)
        r = self.b.validate_entry_quote(wide, ref_price=100.0)
        self.assertFalse(r.ok)
        self.assertIn("spread", r.reason)

    def test_quotes_default_to_venue_sourced_for_backwards_compatibility(self):
        q = Quote("X/USDT", 99.95, 100.05, 100.0, now_ms())
        self.assertEqual(q.ts_source, TS_VENUE)
        self.assertTrue(self.b.validate_entry_quote(q, ref_price=100.0).age_verified)


if __name__ == "__main__":
    unittest.main()
