"""Ticker symbols are not globally unique -- do not guess which asset is meant.

DEFECT UNDER TEST
-----------------
The broad universe mapped a market-cap asset to an exchange market by TICKER
alone, and `_clean()` de-duplicated by ticker, keeping whichever entry ranked
higher. Two independent failure modes followed:

  * **wrongly admitted** -- an exchange lists a small token whose ticker happens
    to match a top-200 asset. The bot treats it as the top-200 asset, passes it
    through every liquidity filter on the small token's own numbers, and trades
    something nobody chose.
  * **evidence destroyed** -- because the duplicate was collapsed silently,
    nothing anywhere recorded that the ticker was ambiguous at all.

Corrected: assets carry the provider's stable id, the scan reaches further down
the ranking than the trading cut-off so collisions are visible, an ambiguous
ticker is REFUSED rather than guessed, and an explicit override can pin one
deterministically. The ambiguity map is cached with the ranking, so a universe
served during a provider outage is exactly as careful as a live one.
"""
import unittest

from crypto_edge.data.broad_universe import (BroadUniverse, BroadUniverseService,
                                             RankedAsset, content_hash)
from crypto_edge.data.universe import UniverseBuilder
from crypto_edge.models import MarketMeta
from crypto_edge.timeutils import now_ms
from helpers import open_repo, temp_repo, test_config

HOUR = 3_600_000

# The Graph (top-200, real) and an unrelated token reusing GRT far below the
# trading cut-off. This is the shape of the real-world hazard.
THE_GRAPH = RankedAsset(3, "GRT", "The Graph", 2.0e9, "the-graph")
IMPOSTOR = RankedAsset(640, "GRT", "Golden Ratio Token", 3.0e6, "golden-ratio-token")


class ScriptedProvider:
    name = "scripted"

    def __init__(self, assets):
        self.assets = assets
        self.last_limit = None

    def fetch(self, limit):
        self.last_limit = limit
        return list(self.assets)[:limit]


def universe(assets=(), limit=10, **kw):
    """A ranking with the standard four majors plus whatever is passed in."""
    base = [RankedAsset(1, "BTC", "Bitcoin", 1e12, "bitcoin"),
            RankedAsset(2, "ETH", "Ethereum", 4e11, "ethereum"),
            RankedAsset(4, "SOL", "Solana", 8e10, "solana")]
    return list(base) + list(assets)


def market(symbol, base=None):
    return MarketMeta(symbol, base or symbol.split("/")[0], "USDT", True,
                      4, 4, 0.0001, 10.0)


def ticker(vol=50_000_000.0, spread_bps=5.0, last=100.0):
    half = spread_bps / 2e4
    return {"quoteVolume": vol, "last": last,
            "bid": last * (1 - half), "ask": last * (1 + half)}


class TestCollisionDetection(unittest.TestCase):
    def setUp(self):
        self.repo, self.path = temp_repo()

    def _service(self, assets, **kw):
        kw.setdefault("limit", 10)
        kw.setdefault("min_assets", 3)
        kw.setdefault("collision_scan_limit", 1000)
        return BroadUniverseService(self.repo, ScriptedProvider(assets), **kw)

    def test_a_collision_below_the_trading_cutoff_is_still_seen(self):
        """The impostor ranks 640th; the tradable list stops at 10. Seeing the
        clash at all requires scanning past the cut-off."""
        svc = self._service(universe([THE_GRAPH, IMPOSTOR]), limit=10)
        u = svc.get()
        self.assertIn("GRT", u.ambiguous)
        self.assertEqual(sorted(u.ambiguous["GRT"]),
                         ["golden-ratio-token", "the-graph"])
        self.assertEqual(svc.provider.last_limit, 1000,
                         "the provider must be asked for the wider scan")

    def test_an_ambiguous_ticker_is_refused_not_guessed(self):
        u = self._service(universe([THE_GRAPH, IMPOSTOR])).get()
        res = u.resolve("GRT")
        self.assertFalse(res.ok, "we must not pick a winner by rank")
        self.assertIn("claimed by 2 distinct assets", res.reason)
        self.assertIsNone(u.rank_of("GRT"))

    def test_the_old_behaviour_would_have_admitted_it(self):
        """Regression guard stated as the defect it replaces: keyed on ticker
        alone, GRT resolves to a rank and sails through."""
        u = self._service(universe([THE_GRAPH, IMPOSTOR])).get()
        by_ticker_only = {a.symbol.upper(): a.rank for a in u.assets}
        self.assertIn("GRT", by_ticker_only,
                      "precondition: a ticker-keyed map would have resolved it")
        self.assertIsNone(u.rank_of("GRT"),
                          "identity-aware resolution must refuse it")

    def test_unaffected_tickers_are_untouched(self):
        u = self._service(universe([THE_GRAPH, IMPOSTOR])).get()
        for sym, rank in (("BTC", 1), ("ETH", 2), ("SOL", 4)):
            self.assertEqual(u.rank_of(sym), rank)

    def test_the_same_asset_listed_twice_is_a_plain_duplicate(self):
        """Two rows with the SAME identity are a provider hiccup, not a clash."""
        dup = RankedAsset(300, "SOL", "Solana", 8e10, "solana")
        u = self._service(universe([dup])).get()
        self.assertNotIn("SOL", u.ambiguous)
        self.assertEqual(u.rank_of("SOL"), 4, "the better-ranked row wins")

    def test_collisions_outside_the_tradable_set_are_not_carried(self):
        """Two obscure assets clashing on a ticker we cannot trade anyway is
        noise; carrying it would bloat every cached snapshot."""
        a = RankedAsset(700, "ZZZ", "Zed One", 1e6, "zed-one")
        b = RankedAsset(800, "ZZZ", "Zed Two", 1e6, "zed-two")
        # limit=3 keeps only BTC/ETH/SOL tradable; both ZZZ rows are scanned
        u = self._service(universe([a, b]), limit=3).get()
        self.assertEqual(len(u), 3)
        self.assertNotIn("ZZZ", u.ambiguous)

    def test_assets_without_a_provider_id_fall_back_to_ticker_identity(self):
        """A provider that supplies no ids must not make everything ambiguous."""
        u = self._service([RankedAsset(1, "BTC"), RankedAsset(2, "ETH"),
                           RankedAsset(3, "SOL")]).get()
        self.assertEqual(u.ambiguous, {})
        self.assertEqual(u.rank_of("BTC"), 1)


class TestOverrides(unittest.TestCase):
    def setUp(self):
        self.repo, self.path = temp_repo()

    def _service(self, overrides):
        return BroadUniverseService(
            self.repo, ScriptedProvider(universe([THE_GRAPH, IMPOSTOR])),
            limit=10, min_assets=3, collision_scan_limit=1000,
            symbol_overrides=overrides)

    def test_an_override_resolves_an_ambiguous_ticker(self):
        u = self._service({"GRT": "the-graph"}).get()
        res = u.resolve("GRT")
        self.assertTrue(res.ok, res.reason)
        self.assertEqual((res.rank, res.asset_id), (3, "the-graph"))

    def test_an_override_can_select_the_lower_ranked_asset(self):
        """Deterministic means deterministic: the operator decides, not rank."""
        svc = BroadUniverseService(
            self.repo, ScriptedProvider(universe([THE_GRAPH, IMPOSTOR])),
            limit=1000, min_assets=3, collision_scan_limit=1000,
            symbol_overrides={"GRT": "golden-ratio-token"})
        res = svc.get().resolve("GRT")
        self.assertTrue(res.ok, res.reason)
        self.assertEqual(res.asset_id, "golden-ratio-token")

    def test_an_override_pointing_at_an_unknown_id_fails_closed(self):
        res = self._service({"GRT": "not-a-real-asset"}).get().resolve("GRT")
        self.assertFalse(res.ok)
        self.assertIn("not in the broad universe", res.reason)

    def test_overrides_are_case_insensitive_on_the_ticker(self):
        u = self._service({"grt": "the-graph"}).get()
        self.assertTrue(u.resolve("GRT").ok)

    def test_an_override_cannot_invent_membership(self):
        """Pinning a ticker to an asset outside the top-N must not admit it."""
        svc = BroadUniverseService(
            self.repo, ScriptedProvider(universe([THE_GRAPH, IMPOSTOR])),
            limit=2, min_assets=1, collision_scan_limit=1000,
            symbol_overrides={"GRT": "the-graph"})
        res = svc.get().resolve("GRT")
        self.assertFalse(res.ok, "The Graph is outside the top 2 here")


class TestCollisionsSurviveCaching(unittest.TestCase):
    def setUp(self):
        self.repo, self.path = temp_repo()

    def test_ambiguity_is_stored_and_reloaded(self):
        # limit=4 makes the tradable list BTC/ETH/GRT/SOL while the scan also
        # sees the rank-640 impostor -- the realistic shape.
        svc = BroadUniverseService(
            self.repo, ScriptedProvider(universe([THE_GRAPH, IMPOSTOR])),
            limit=4, min_assets=3, collision_scan_limit=1000)
        svc.get(now=now_ms())
        row = self.repo.latest_broad_universe()
        self.assertIn("GRT", row["ambiguous"])
        self.assertEqual(row["n_assets"], 4)
        self.assertGreater(row["scanned"], row["n_assets"],
                           "the scan must reach past the trading cut-off")

    def test_a_cached_universe_is_as_careful_as_a_live_one(self):
        """The failure this prevents: an outage falls back to a cache that has
        forgotten the ticker was ambiguous, and starts trading the impostor."""
        class Dead:
            name = "dead"

            def fetch(self, limit):
                raise RuntimeError("provider down")

        svc = BroadUniverseService(
            self.repo, ScriptedProvider(universe([THE_GRAPH, IMPOSTOR])),
            limit=10, min_assets=3, collision_scan_limit=1000, refresh_hours=12)
        svc.get(now=now_ms())
        svc.provider = Dead()
        fallback = svc.get(now=now_ms() + 24 * HOUR)
        self.assertTrue(fallback.from_cache)
        self.assertIn("GRT", fallback.ambiguous)
        self.assertFalse(fallback.resolve("GRT").ok)

    def test_ambiguity_survives_a_restart(self):
        svc = BroadUniverseService(
            self.repo, ScriptedProvider(universe([THE_GRAPH, IMPOSTOR])),
            limit=10, min_assets=3, collision_scan_limit=1000)
        svc.get()
        self.repo.conn.close()
        repo2 = open_repo(self.path)
        svc2 = BroadUniverseService(repo2, None, limit=10, min_assets=3)
        reloaded = svc2.get()
        self.assertIn("GRT", reloaded.ambiguous)
        self.assertFalse(reloaded.resolve("GRT").ok)

    def test_overrides_apply_to_a_cached_universe_too(self):
        svc = BroadUniverseService(
            self.repo, ScriptedProvider(universe([THE_GRAPH, IMPOSTOR])),
            limit=10, min_assets=3, collision_scan_limit=1000)
        svc.get()
        self.repo.conn.close()
        repo2 = open_repo(self.path)
        svc2 = BroadUniverseService(repo2, None, limit=10, min_assets=3,
                                    symbol_overrides={"GRT": "the-graph"})
        self.assertTrue(svc2.get().resolve("GRT").ok)

    def test_a_legacy_list_payload_still_loads(self):
        """Snapshots written before ambiguity was tracked must not break."""
        import json
        self.repo.conn.execute(
            "INSERT INTO broad_universe_cache"
            "(fetched_ms, source, as_of_ms, n_assets, content_hash, payload)"
            " VALUES(?,?,?,?,?,?)",
            (now_ms(), "legacy", now_ms(), 3, "abc",
             json.dumps([{"rank": 1, "symbol": "BTC"}, {"rank": 2, "symbol": "ETH"},
                         {"rank": 3, "symbol": "SOL"}])))
        row = self.repo.latest_broad_universe()
        self.assertEqual(len(row["assets"]), 3)
        self.assertEqual(row["ambiguous"], {})


class TestContentHashIncludesIdentity(unittest.TestCase):
    def test_same_tickers_different_assets_hash_differently(self):
        a = [RankedAsset(1, "GRT", "The Graph", 0, "the-graph")]
        b = [RankedAsset(1, "GRT", "Golden Ratio", 0, "golden-ratio-token")]
        self.assertNotEqual(content_hash(a), content_hash(b),
                            "identity must be part of what makes a universe")

    def test_identical_rankings_hash_identically(self):
        a = [RankedAsset(1, "BTC", "Bitcoin", 1e12, "bitcoin")]
        self.assertEqual(content_hash(a), content_hash(list(a)))


class TestIntersectionRefusesAmbiguousMarkets(unittest.TestCase):
    def setUp(self):
        self.cfg = test_config().universe
        self.b = UniverseBuilder(self.cfg)

    def _broad(self, assets, ambiguous=None, overrides=None):
        return BroadUniverse(list(assets), "test", now_ms(), now_ms(),
                             content_hash(list(assets)),
                             ambiguous=ambiguous or {}, overrides=overrides or {})

    def test_an_ambiguous_market_is_excluded_with_a_specific_reason(self):
        markets = {"BTC/USDT": market("BTC/USDT"), "GRT/USDT": market("GRT/USDT")}
        tickers = {s: ticker() for s in markets}
        broad = self._broad(universe([THE_GRAPH]),
                            ambiguous={"GRT": ["the-graph", "golden-ratio-token"]})
        keep, audit = self.b.build_candidates(markets, tickers, broad)
        self.assertEqual(keep, ["BTC/USDT"])
        reason = [r for r in audit if r["symbol"] == "GRT/USDT"][0]["reject_reason"]
        self.assertIn("claimed by", reason)
        self.assertNotIn("not in the broad top-N", reason,
                         "the audit must distinguish ambiguity from absence")

    def test_an_override_lets_the_market_through_the_intersection(self):
        markets = {"GRT/USDT": market("GRT/USDT")}
        tickers = {s: ticker() for s in markets}
        broad = self._broad(universe([THE_GRAPH]),
                            ambiguous={"GRT": ["the-graph", "golden-ratio-token"]},
                            overrides={"GRT": "the-graph"})
        keep, _ = self.b.build_candidates(markets, tickers, broad)
        self.assertEqual(keep, ["GRT/USDT"])

    def test_absence_and_ambiguity_produce_different_audit_reasons(self):
        markets = {"GRT/USDT": market("GRT/USDT"), "NOPE/USDT": market("NOPE/USDT")}
        tickers = {s: ticker() for s in markets}
        broad = self._broad(universe([THE_GRAPH]),
                            ambiguous={"GRT": ["the-graph", "golden-ratio-token"]})
        _, audit = self.b.build_candidates(markets, tickers, broad)
        reasons = {r["symbol"]: r["reject_reason"] for r in audit}
        self.assertIn("claimed by", reasons["GRT/USDT"])
        self.assertIn("not in the broad top-N", reasons["NOPE/USDT"])

    def test_bases_property_excludes_ambiguous_tickers(self):
        broad = self._broad(universe([THE_GRAPH]),
                            ambiguous={"GRT": ["the-graph", "golden-ratio-token"]})
        self.assertNotIn("GRT", broad.bases)
        self.assertIn("BTC", broad.bases)


class TestConfigValidation(unittest.TestCase):
    def test_scan_limit_below_the_trading_limit_is_rejected(self):
        cfg = test_config()
        cfg.universe.broad_limit = 200
        cfg.universe.broad_collision_scan_limit = 50
        self.assertTrue(any("collision_scan_limit" in e for e in cfg.validate()))

    def test_overrides_must_map_to_non_empty_ids(self):
        cfg = test_config()
        cfg.universe.broad_symbol_overrides = {"GRT": ""}
        self.assertTrue(any("broad_symbol_overrides" in e for e in cfg.validate()))

    def test_valid_overrides_pass(self):
        cfg = test_config()
        cfg.universe.broad_symbol_overrides = {"GRT": "the-graph"}
        self.assertEqual(cfg.validate(), [])


if __name__ == "__main__":
    unittest.main()
