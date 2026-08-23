"""The broad asset universe replaces the exchange-volume Top-200.

DEFECT UNDER TEST
-----------------
"Top 200" used to mean "the 200 markets with the highest 24h volume on our
exchange". That is a snapshot of where speculation is pointed today, not a
definition of a legitimate asset: it admits whatever is being churned hardest
right now, changes with listing promotions and fee campaigns, and makes any
backtest run against it irreproducible.

Corrected pipeline:

    BROAD TOP ~200 ASSETS (market cap, venue-independent)
      -> intersect with this exchange's markets
      -> drop stablecoins / wrapped / staked / leveraged / blacklisted
      -> liquidity, spread, history, age, volatility filters
      -> strategy ranking

with a cache carrying source + timestamp + content hash, fallback to the last
valid cached snapshot during an outage, and FAIL CLOSED for new entries when
neither exists -- while open positions keep being managed regardless.

The live CoinGecko provider REQUIRES VERIFICATION ON YOUR MACHINE (no network
here). Everything below runs offline against injected providers.
"""
import unittest

from crypto_edge.data.broad_universe import (NON_ASSET_BASES, BroadUniverse,
                                             BroadUniverseService, RankedAsset,
                                             StaticBroadUniverseProvider,
                                             content_hash)
from crypto_edge.data.universe import UniverseBuilder
from crypto_edge.models import MarketMeta
from crypto_edge.timeutils import now_ms
from helpers import (breakout_closes, engine_config, temp_repo, test_config,
                     trend_closes)
from test_engine import build_engine

HOUR = 3_600_000


class ExplodingProvider:
    name = "exploding"

    def __init__(self, error="provider is down"):
        self.error = error
        self.calls = 0

    def fetch(self, limit):
        self.calls += 1
        raise RuntimeError(self.error)


class ThinProvider:
    """Returns far fewer assets than the floor -- a broken provider, not an
    empty market."""
    name = "thin"

    def fetch(self, limit):
        return [RankedAsset(1, "BTC"), RankedAsset(2, "ETH")]


def market(symbol, base=None, active=True):
    return MarketMeta(symbol, base or symbol.split("/")[0], "USDT", active,
                      4, 4, 0.0001, 10.0)


def ticker(vol=50_000_000.0, spread_bps=5.0, last=100.0):
    half = spread_bps / 2e4
    return {"quoteVolume": vol, "last": last,
            "bid": last * (1 - half), "ask": last * (1 + half)}


def service(repo, provider, **kw):
    kw.setdefault("min_assets", 3)
    kw.setdefault("limit", 200)
    return BroadUniverseService(repo, provider, **kw)


class TestProviderAndCleaning(unittest.TestCase):
    def setUp(self):
        self.repo, self.path = temp_repo()

    def test_ranking_is_by_market_cap_rank_not_volume(self):
        prov = StaticBroadUniverseProvider(["BTC", "ETH", "SOL", "ADA"])
        u = service(self.repo, prov).get()
        self.assertEqual([a.symbol for a in u.assets], ["BTC", "ETH", "SOL", "ADA"])
        self.assertEqual(u.rank_of("BTC"), 1)
        self.assertEqual(u.rank_of("ADA"), 4)
        self.assertIsNone(u.rank_of("DOGE"))

    def test_stablecoins_and_wrapped_never_enter_the_broad_list(self):
        prov = StaticBroadUniverseProvider(
            ["BTC", "USDT", "ETH", "WBTC", "USDC", "STETH", "SOL"])
        u = service(self.repo, prov).get()
        self.assertEqual([a.symbol for a in u.assets], ["BTC", "ETH", "SOL"])
        for junk in ("USDT", "WBTC", "USDC", "STETH"):
            self.assertIn(junk, NON_ASSET_BASES)
            self.assertIsNone(u.rank_of(junk))

    def test_duplicates_are_collapsed(self):
        prov = StaticBroadUniverseProvider(["BTC", "btc", "ETH", "BTC", "SOL"])
        u = service(self.repo, prov).get()
        self.assertEqual([a.symbol for a in u.assets], ["BTC", "ETH", "SOL"])

    def test_the_list_is_capped_at_the_configured_limit(self):
        prov = StaticBroadUniverseProvider([f"A{i}" for i in range(500)])
        u = service(self.repo, prov, limit=200).get()
        self.assertEqual(len(u), 200)

    def test_a_provider_returning_too_few_assets_is_treated_as_broken(self):
        u = service(self.repo, ThinProvider(), min_assets=50).get()
        self.assertIsNone(u, "a two-asset 'universe' is a broken provider")

    def test_lookup_is_case_insensitive(self):
        u = service(self.repo, StaticBroadUniverseProvider(["btc", "eth", "sol"])).get()
        self.assertEqual(u.rank_of("BtC"), 1)
        self.assertEqual(u.bases, {"BTC", "ETH", "SOL"})


class TestCachingAndProvenance(unittest.TestCase):
    def setUp(self):
        self.repo, self.path = temp_repo()
        self.prov = StaticBroadUniverseProvider(["BTC", "ETH", "SOL", "ADA"],
                                                name="fixture_cap")

    def test_source_and_timestamp_are_stored(self):
        now = now_ms()
        u = service(self.repo, self.prov).get(now=now)
        row = self.repo.latest_broad_universe()
        self.assertEqual(row["source"], "fixture_cap")
        self.assertEqual(row["fetched_ms"], now)
        self.assertEqual(row["as_of_ms"], now)
        self.assertEqual(row["n_assets"], 4)
        self.assertEqual(row["content_hash"], u.content_hash)

    def test_content_hash_is_reproducible_and_content_sensitive(self):
        a = [RankedAsset(1, "BTC"), RankedAsset(2, "ETH")]
        b = [RankedAsset(1, "BTC"), RankedAsset(2, "SOL")]
        self.assertEqual(content_hash(a), content_hash(list(a)))
        self.assertNotEqual(content_hash(a), content_hash(b))

    def test_a_fresh_cache_is_reused_without_refetching(self):
        svc = service(self.repo, self.prov, refresh_hours=12)
        now = now_ms()
        svc.get(now=now)
        calls_before = len(self.repo.latest_broad_universe()["assets"])
        again = svc.get(now=now + HOUR)
        self.assertTrue(again.from_cache)
        self.assertFalse(again.stale)
        self.assertEqual(len(again), calls_before)

    def test_a_stale_cache_triggers_a_refetch(self):
        svc = service(self.repo, self.prov, refresh_hours=12)
        svc.get(now=now_ms())
        later = svc.get(now=now_ms() + 24 * HOUR)
        self.assertFalse(later.from_cache, "a stale cache must be refreshed")

    def test_history_is_append_only_so_past_decisions_reproduce(self):
        svc = service(self.repo, self.prov)
        t0 = now_ms()
        svc.get(now=t0, force=True)
        svc.provider = StaticBroadUniverseProvider(["BTC", "ETH", "SOL", "XRP"],
                                                   name="fixture_cap")
        svc.get(now=t0 + 24 * HOUR, force=True)
        rows = self.repo.conn.execute(
            "SELECT fetched_ms, content_hash FROM broad_universe_cache "
            "ORDER BY fetched_ms").fetchall()
        self.assertEqual(len(rows), 2, "both snapshots must be retained")
        self.assertNotEqual(rows[0]["content_hash"], rows[1]["content_hash"])
        self.assertEqual(self.repo.latest_broad_universe()["fetched_ms"],
                         t0 + 24 * HOUR)


class TestOutageFallback(unittest.TestCase):
    def setUp(self):
        self.repo, self.path = temp_repo()

    def test_outage_falls_back_to_the_last_valid_cache(self):
        good = StaticBroadUniverseProvider(["BTC", "ETH", "SOL"], name="good")
        svc = service(self.repo, good, refresh_hours=12)
        svc.get(now=now_ms())

        svc.provider = ExplodingProvider()
        later = svc.get(now=now_ms() + 24 * HOUR)      # cache is stale -> refetch
        self.assertIsNotNone(later, "an outage must fall back, not fail")
        self.assertTrue(later.from_cache)
        self.assertTrue(later.stale, "the fallback must be flagged as stale")
        self.assertEqual([a.symbol for a in later.assets], ["BTC", "ETH", "SOL"])

    def test_fallback_survives_a_restart(self):
        good = StaticBroadUniverseProvider(["BTC", "ETH", "SOL"], name="good")
        service(self.repo, good).get(now=now_ms())
        self.repo.conn.close()

        from helpers import open_repo
        repo2 = open_repo(self.path)
        svc2 = service(repo2, ExplodingProvider(), refresh_hours=12)
        u = svc2.get(now=now_ms() + 24 * HOUR)
        self.assertIsNotNone(u)
        self.assertTrue(u.from_cache)

    def test_a_cache_older_than_the_limit_stops_being_usable(self):
        good = StaticBroadUniverseProvider(["BTC", "ETH", "SOL"], name="good")
        svc = service(self.repo, good, max_cache_age_hours=168)
        svc.get(now=now_ms())
        svc.provider = ExplodingProvider()
        u = svc.get(now=now_ms() + 200 * HOUR)
        self.assertIsNone(u, "a month-old ranking is not a fallback, it is a guess")

    def test_no_provider_and_no_cache_returns_nothing(self):
        svc = service(self.repo, None)
        self.assertIsNone(svc.get())
        self.assertIn("no broad-universe provider", svc.last_error)

    def test_the_outage_reason_is_recorded(self):
        svc = service(self.repo, ExplodingProvider("HTTP 429 rate limited"))
        self.assertIsNone(svc.get())
        self.assertIn("429", svc.last_error)


class TestIntersectionWithExchangeMarkets(unittest.TestCase):
    def setUp(self):
        self.cfg = test_config().universe
        self.b = UniverseBuilder(self.cfg)

    def _broad(self, symbols):
        assets = [RankedAsset(i, s) for i, s in enumerate(symbols, start=1)]
        return BroadUniverse(assets, "test", now_ms(), now_ms(), content_hash(assets))

    def test_markets_outside_the_broad_universe_are_rejected(self):
        markets = {"BTC/USDT": market("BTC/USDT"), "SOL/USDT": market("SOL/USDT"),
                   "SCAMCOIN/USDT": market("SCAMCOIN/USDT")}
        tickers = {s: ticker() for s in markets}
        keep, audit = self.b.build_candidates(markets, tickers,
                                              self._broad(["BTC", "SOL"]))
        self.assertEqual(sorted(keep), ["BTC/USDT", "SOL/USDT"])
        rejected = [r for r in audit if r["symbol"] == "SCAMCOIN/USDT"][0]
        self.assertIn("not in the broad top-N asset universe",
                      rejected["reject_reason"])

    def test_a_huge_volume_market_outside_the_broad_list_is_still_rejected(self):
        """The exact defect: venue volume no longer buys membership."""
        markets = {"BTC/USDT": market("BTC/USDT"), "HYPEDOG/USDT": market("HYPEDOG/USDT")}
        tickers = {"BTC/USDT": ticker(vol=1_000_000.0 * 50),
                   "HYPEDOG/USDT": ticker(vol=1_000_000_000_000.0)}   # #1 by volume
        keep, audit = self.b.build_candidates(markets, tickers, self._broad(["BTC"]))
        self.assertEqual(keep, ["BTC/USDT"])
        self.assertIn("not in the broad", [r for r in audit
                                           if r["symbol"] == "HYPEDOG/USDT"][0]["reject_reason"])

    def test_pipeline_order_follows_market_cap_not_volume(self):
        markets = {s: market(s) for s in ("BTC/USDT", "ETH/USDT", "SOL/USDT")}
        # SOL has by far the most volume, but is third by market cap
        tickers = {"BTC/USDT": ticker(vol=50_000_000.0),
                   "ETH/USDT": ticker(vol=60_000_000.0),
                   "SOL/USDT": ticker(vol=900_000_000.0)}
        keep, _ = self.b.build_candidates(markets, tickers,
                                          self._broad(["BTC", "ETH", "SOL"]))
        self.assertEqual(keep, ["BTC/USDT", "ETH/USDT", "SOL/USDT"])

    def test_liquidity_and_spread_filters_still_apply_after_the_intersection(self):
        # deliberately avoids always_include symbols, which bypass these gates
        markets = {s: market(s) for s in ("BTC/USDT", "ADA/USDT", "SOL/USDT")}
        tickers = {"BTC/USDT": ticker(vol=50_000_000.0, spread_bps=3.0),
                   "ADA/USDT": ticker(vol=1_000.0, spread_bps=3.0),        # illiquid
                   "SOL/USDT": ticker(vol=50_000_000.0, spread_bps=500.0)}  # wide
        keep, audit = self.b.build_candidates(markets, tickers,
                                              self._broad(["BTC", "ADA", "SOL"]))
        self.assertEqual(keep, ["BTC/USDT"])
        reasons = {r["symbol"]: r["reject_reason"] for r in audit}
        self.assertIn("volume", reasons["ADA/USDT"])
        self.assertIn("spread", reasons["SOL/USDT"])

    def test_always_include_survives_a_missing_broad_ranking(self):
        """BTC and ETH are the strategy's regime reference; losing them to a
        ranking hiccup would be worse than including them."""
        markets = {"ETH/USDT": market("ETH/USDT"), "JUNK/USDT": market("JUNK/USDT")}
        tickers = {s: ticker() for s in markets}
        keep, _ = self.b.build_candidates(markets, tickers, self._broad(["BTC"]))
        self.assertIn("ETH/USDT", keep)
        self.assertNotIn("JUNK/USDT", keep)

    def test_static_junk_filters_still_apply_after_the_intersection(self):
        markets = {"BTC/USDT": market("BTC/USDT"),
                   "BTC3L/USDT": market("BTC3L/USDT"),
                   "DEAD/USDT": market("DEAD/USDT", active=False)}
        tickers = {s: ticker() for s in markets}
        broad = self._broad(["BTC", "BTC3L", "DEAD"])
        keep, audit = self.b.build_candidates(markets, tickers, broad)
        self.assertEqual(keep, ["BTC/USDT"])
        reasons = {r["symbol"]: r["reject_reason"] for r in audit}
        self.assertEqual(reasons["BTC3L/USDT"], "leveraged token")
        self.assertIn("inactive", reasons["DEAD/USDT"])

    def test_top_n_cut_applies_to_the_broad_rank(self):
        cfg = test_config().universe
        cfg.top_n = 2
        b = UniverseBuilder(cfg)
        markets = {s: market(s) for s in ("BTC/USDT", "ETH/USDT", "SOL/USDT")}
        tickers = {s: ticker() for s in markets}
        keep, audit = b.build_candidates(markets, tickers,
                                         self._broad(["BTC", "ETH", "SOL"]))
        self.assertEqual(keep, ["BTC/USDT", "ETH/USDT"])
        self.assertIn("outside top 2 by market cap",
                      [r for r in audit if r["symbol"] == "SOL/USDT"][0]["reject_reason"])

    def test_every_symbol_considered_is_still_audited(self):
        markets = {s: market(s) for s in ("BTC/USDT", "JUNK/USDT")}
        tickers = {s: ticker() for s in markets}
        keep, audit = self.b.build_candidates(markets, tickers, self._broad(["BTC"]))
        self.assertEqual(len(audit), 2)
        self.assertTrue(all("reject_reason" in r for r in audit))

    def test_the_audit_records_the_market_cap_rank(self):
        markets = {s: market(s) for s in ("BTC/USDT", "SOL/USDT")}
        tickers = {s: ticker() for s in markets}
        _, audit = self.b.build_candidates(markets, tickers,
                                           self._broad(["BTC", "ETH", "SOL"]))
        ranks = {r["symbol"]: r["rank"] for r in audit}
        self.assertEqual(ranks["BTC/USDT"], 1)
        self.assertEqual(ranks["SOL/USDT"], 3, "SOL keeps its market-cap rank of 3")


class TestEngineFailsClosedWithoutAUniverse(unittest.TestCase):
    def setUp(self):
        self.closes = {"BTC/USDT": trend_closes(400, seed=4),
                       "SOL/USDT": breakout_closes(400, seed=11)}

    def _engine(self, source="none", static=None, provider=None):
        cfg = engine_config()
        cfg.universe.broad_source = source
        cfg.universe.broad_static_assets = static or []
        engine, feed, transport, repo = build_engine(self.closes, cfg=cfg)
        if provider is not None:
            engine.broad_universe.provider = provider
        return engine, feed, transport, repo

    def test_no_universe_means_no_entries(self):
        engine, feed, transport, repo = self._engine(source="none")
        engine.cycle()
        self.assertEqual(engine.status.universe, [])
        self.assertEqual(len(engine.account.positions()), 0,
                         "with no legitimate-asset list, nothing may be opened")
        allowed, why = engine.entries_allowed()
        self.assertFalse(allowed)
        self.assertIn("no valid broad asset universe", why)

    def test_the_suspension_is_announced(self):
        engine, feed, transport, repo = self._engine(source="none")
        engine.cycle()
        self.assertTrue(any("UNIVERSE UNAVAILABLE" in m for m in transport.sent),
                        "an operator must be told entries are suspended")

    def test_a_provider_outage_with_a_valid_cache_keeps_trading(self):
        engine, feed, transport, repo = self._engine(
            source="static", static=["BTC", "ETH", "SOL"])
        engine.cycle()
        self.assertEqual(len(engine.account.positions()), 1, "control case")

        # provider dies; the cached universe must carry us
        engine.broad_universe.provider = ExplodingProvider()
        engine.status.last_universe_refresh_ms = 0        # force a refresh
        universe = engine.refresh_universe(force=True)
        self.assertTrue(universe, "a cached universe must keep the pipeline alive")
        self.assertTrue(engine.entries_allowed()[0])

    def test_open_positions_are_managed_even_with_no_universe(self):
        """The provider must never strand risk already on the books."""
        engine, feed, transport, repo = self._engine(
            source="static", static=["BTC", "ETH", "SOL"])
        engine.cycle()
        self.assertEqual(len(engine.account.positions()), 1)
        pos = engine.account.positions()[0]

        # the entire broad-universe layer disappears, cache included
        engine.broad_universe.provider = None
        repo.conn.execute("DELETE FROM broad_universe_cache")
        engine.status.last_universe_refresh_ms = 0
        engine.cycle()

        self.assertEqual(engine.status.universe, [], "universe must be empty")
        self.assertFalse(engine.entries_allowed()[0])
        self.assertIn(pos.symbol, engine._series_1h,
                      "an open position must keep receiving candles, or its stop "
                      "can never be evaluated")

    def test_a_held_symbol_outside_the_universe_still_gets_data(self):
        engine, feed, transport, repo = self._engine(
            source="static", static=["BTC", "ETH", "SOL"])
        engine.cycle()
        pos = engine.account.positions()[0]
        engine.status.universe = []            # symbol drops out of the scan list
        engine._series_1h.clear()
        engine.fetch_data([])
        self.assertIn(pos.symbol, engine._series_1h)

    def test_circuit_breakers_still_trip_when_the_universe_is_unavailable(self):
        """Entry gating must not short-circuit the safety layer: a drawdown
        halt has to trip and persist even on a cycle that could not have
        entered anything anyway."""
        engine, feed, transport, repo = self._engine(source="none")
        repo.update_account(cash=7_000.0, peak_equity=10_000.0,
                            daily_start_equity=7_000.0)
        engine.cycle()
        acct = repo.get_account()
        self.assertTrue(int(acct["halted"]),
                        "the drawdown breaker must still trip")
        self.assertIn("DRAWDOWN", acct["halt_reason"])

    def test_disabling_the_requirement_is_an_explicit_opt_out(self):
        engine, feed, transport, repo = self._engine(source="none")
        engine.cfg.universe.require_broad_universe = False
        allowed, why = engine.entries_allowed()
        self.assertTrue(allowed, "the fail-closed gate must be explicitly waivable")
        self.assertEqual(why, "")


class TestConfigValidation(unittest.TestCase):
    def test_static_source_requires_a_list(self):
        cfg = test_config()
        cfg.universe.broad_source = "static"
        cfg.universe.broad_static_assets = []
        self.assertTrue(any("broad_static_assets" in e for e in cfg.validate()))

    def test_unknown_source_is_rejected(self):
        cfg = test_config()
        cfg.universe.broad_source = "vibes"
        self.assertTrue(any("broad_source" in e for e in cfg.validate()))

    def test_requiring_a_universe_with_no_source_is_rejected(self):
        cfg = test_config()
        cfg.universe.broad_source = "none"
        cfg.universe.require_broad_universe = True
        self.assertTrue(any("require_broad_universe" in e for e in cfg.validate()))

    def test_the_shipped_defaults_are_valid(self):
        from crypto_edge.config import load_config
        cfg = load_config("config/config.toml", "/nonexistent")
        cfg.telegram.enabled = False        # secrets are not present in CI
        self.assertEqual(cfg.validate(), [])
        self.assertEqual(cfg.universe.broad_source, "coingecko")
        self.assertTrue(cfg.universe.require_broad_universe)


if __name__ == "__main__":
    unittest.main()
