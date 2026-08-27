"""24h volume: quote-currency semantics, audited rather than assumed.

WHAT WAS REPORTED
-----------------
25 of 29 Kraken markets that intersect the broad universe were rejected for
`24h volume`. That ratio is high enough that the filter had to be proven correct
before being believed.

WHAT THE AUDIT FOUND
--------------------
CCXT's kraken.parse_ticker sets `quoteVolume = baseVolume x vwap`, where
baseVolume is `v[1]` (24h base volume) and vwap is `p[1]` (24h vwap). That is
already the 24h notional in the QUOTE currency, correctly converted. There is no
base/quote confusion and no unit error to fix in the Kraken path.

What WAS wrong was our own fallback. The old code only computed a replacement
when `quoteVolume` was exactly `None`:

    qv = t.get("quoteVolume")
    if qv is None:
        ...derive from baseVolume...
    qv = float(qv or 0.0)

so a venue publishing `0`, `"0"`, `""` or an unparseable value skipped the
fallback entirely and was recorded as zero volume -- rejecting a market for
illiquidity it might not have. A missing measurement is not a measurement of
zero, and the two must not collapse into the same number.

The threshold itself is unchanged.
"""
import unittest

import helpers  # noqa: F401  -- silences the engine's log handlers
from crypto_edge.data.universe import UniverseBuilder
from crypto_edge.models import MarketMeta
from helpers import test_config


def market(symbol="SOL/USDT"):
    return MarketMeta(symbol, symbol.split("/")[0], "USDT", True,
                      4, 4, 0.0001, 10.0)


class TestQuoteVolumeDerivation(unittest.TestCase):
    """Each source, in the order of accuracy it deserves."""

    def test_quote_volume_is_used_when_published(self):
        v, how = UniverseBuilder.quote_volume(
            {"quoteVolume": 12_345_678.0, "baseVolume": 100.0, "last": 1.0})
        self.assertEqual(v, 12_345_678.0)
        self.assertEqual(how, "quoteVolume")

    def test_kraken_semantics_reproduce_the_ccxt_calculation(self):
        """base x vwap is what ccxt computes for kraken; we must agree."""
        base, vwap = 1_500.0, 62_000.0
        v, how = UniverseBuilder.quote_volume(
            {"quoteVolume": base * vwap, "baseVolume": base, "vwap": vwap})
        self.assertAlmostEqual(v, 93_000_000.0, places=2)
        self.assertEqual(how, "quoteVolume")

    def test_vwap_is_preferred_over_last_when_quote_volume_is_absent(self):
        """vwap is the 24h average; last is a point estimate of a 24h figure."""
        v, how = UniverseBuilder.quote_volume(
            {"baseVolume": 1_000.0, "vwap": 50.0, "last": 80.0})
        self.assertEqual(v, 50_000.0)
        self.assertEqual(how, "baseVolume x vwap")

    def test_last_price_is_the_final_fallback(self):
        v, how = UniverseBuilder.quote_volume({"baseVolume": 1_000.0, "last": 80.0})
        self.assertEqual(v, 80_000.0)
        self.assertEqual(how, "baseVolume x last")

    def test_close_substitutes_for_last(self):
        v, how = UniverseBuilder.quote_volume({"baseVolume": 10.0, "close": 7.0})
        self.assertEqual(v, 70.0)

    def test_base_volume_is_never_mistaken_for_quote_volume(self):
        """The specific confusion asked about: 1000 BTC is not 1000 dollars."""
        v, how = UniverseBuilder.quote_volume({"baseVolume": 1_000.0, "last": 60_000.0})
        self.assertEqual(v, 60_000_000.0)
        self.assertNotEqual(v, 1_000.0, "base units must be converted, not copied")


class TestMissingIsNotZero(unittest.TestCase):
    """The defect: a falsy-but-not-None quoteVolume skipped the fallback."""

    def test_a_zero_quote_volume_falls_through_to_base_volume(self):
        v, how = UniverseBuilder.quote_volume(
            {"quoteVolume": 0, "baseVolume": 1_000.0, "last": 50.0})
        self.assertEqual(v, 50_000.0)
        self.assertEqual(how, "baseVolume x last")

    def test_a_zero_string_falls_through_too(self):
        v, _ = UniverseBuilder.quote_volume(
            {"quoteVolume": "0", "baseVolume": 1_000.0, "last": 50.0})
        self.assertEqual(v, 50_000.0)

    def test_an_empty_string_falls_through(self):
        v, _ = UniverseBuilder.quote_volume(
            {"quoteVolume": "", "baseVolume": 1_000.0, "last": 50.0})
        self.assertEqual(v, 50_000.0)

    def test_ccxt_string_numbers_are_accepted(self):
        """ccxt's safe_string leaves numeric strings in the ticker."""
        v, how = UniverseBuilder.quote_volume({"quoteVolume": "12345678.9"})
        self.assertAlmostEqual(v, 12_345_678.9, places=2)

    def test_a_genuinely_empty_ticker_reports_unavailable_not_zero_volume(self):
        v, how = UniverseBuilder.quote_volume({})
        self.assertEqual(v, 0.0)
        self.assertEqual(how, "unavailable",
                         "the report must distinguish 'no data' from 'no volume'")

    def test_nonsense_values_do_not_crash(self):
        for bad in ({"quoteVolume": "n/a"}, {"quoteVolume": None},
                    {"baseVolume": "x", "last": "y"}, {"quoteVolume": float("nan")},
                    {"quoteVolume": float("inf")}, {"baseVolume": -5.0, "last": 2.0}):
            v, how = UniverseBuilder.quote_volume(bad)
            self.assertGreaterEqual(v, 0.0)

    def test_negative_volume_is_not_trusted(self):
        v, _ = UniverseBuilder.quote_volume({"quoteVolume": -100.0,
                                             "baseVolume": 10.0, "last": 3.0})
        self.assertEqual(v, 30.0, "a negative figure must not be used as-is")


class TestRankingAndFiltering(unittest.TestCase):
    def setUp(self):
        self.cfg = test_config().universe
        self.b = UniverseBuilder(self.cfg)

    def test_ranking_uses_the_derived_notional(self):
        markets = {s: market(s) for s in ("A/USDT", "B/USDT", "C/USDT")}
        tickers = {
            "A/USDT": {"baseVolume": 1_000.0, "last": 10.0},        # 10k
            "B/USDT": {"quoteVolume": 90_000.0},                     # 90k
            "C/USDT": {"baseVolume": 100.0, "vwap": 500.0},          # 50k
        }
        order = [s for s, _ in self.b.rank_by_volume(tickers, markets)]
        self.assertEqual(order, ["B/USDT", "C/USDT", "A/USDT"])

    def test_a_market_with_only_base_volume_is_no_longer_zeroed(self):
        """Previously this market was rejected as having no volume at all."""
        markets = {"SOL/USDT": market()}
        tickers = {"SOL/USDT": {"quoteVolume": 0,
                                "baseVolume": 1_000_000.0, "last": 100.0}}
        keep, audit = self.b.build_candidates(markets, tickers)
        self.assertEqual(keep, ["SOL/USDT"],
                         "100m of notional must not read as zero")

    def test_a_genuinely_thin_market_is_still_rejected(self):
        """The fix must not admit low-volume markets -- that would be loosening."""
        markets = {"THIN/USDT": market("THIN/USDT")}
        tickers = {"THIN/USDT": {"baseVolume": 10.0, "last": 2.0}}   # $20
        keep, audit = self.b.build_candidates(markets, tickers)
        self.assertEqual(keep, [])
        self.assertIn("volume", audit[0]["reject_reason"])

    def test_a_market_with_no_volume_data_at_all_still_fails_closed(self):
        markets = {"GHOST/USDT": market("GHOST/USDT")}
        keep, audit = self.b.build_candidates(markets, {"GHOST/USDT": {}})
        self.assertEqual(keep, [], "unmeasurable liquidity is not tradable")

    def test_the_threshold_itself_is_unchanged(self):
        from crypto_edge.config import load_config
        cfg = load_config("config/config.toml", "/nonexistent")
        self.assertEqual(cfg.universe.min_dollar_volume_24h, 5_000_000.0)
        self.assertEqual(cfg.universe.max_spread_bps, 15.0)


if __name__ == "__main__":
    unittest.main()
