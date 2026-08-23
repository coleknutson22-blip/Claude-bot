"""Research database, news/event architecture and universe filtering.

The centrepiece here is the `event_time` + `observed_at` separation: a future
backtest must never be able to consume information before the system actually
received it. `event_time` is when the thing happened in the world;
`observed_at` is when THIS system learned about it. Every visibility query
filters on `observed_at`.
"""
import unittest

import numpy as np

from crypto_edge.data.fixture_feed import make_series
from crypto_edge.data.universe import UniverseBuilder
from crypto_edge.intel.events import TokenEventCalendar
from crypto_edge.intel.news import (NewsEngine, NewsItem, classify,
                                    confidence_for, event_id)
from crypto_edge.models import MarketMeta
from crypto_edge.research.counterfactual import CounterfactualTracker
from crypto_edge.research.journal import ResearchJournal
from crypto_edge.strategy.base import Signal
from helpers import default_meta, flat_closes, open_repo, temp_repo, test_config

HOUR = 3_600_000
NOW = 1_760_000_000_000


def news(headline, *, event_time, observed_at, tier=1, assets=("SOL",),
         source="reuters"):
    return NewsItem(headline=headline, source=source, source_tier=tier,
                    event_time=event_time, observed_at=observed_at,
                    assets=assets)


# ===================================================== look-ahead protection
class TestNewsLookAhead(unittest.TestCase):
    def setUp(self):
        self.repo, self.path = temp_repo()
        self.engine = NewsEngine(self.repo)

    def test_event_not_yet_observed_is_invisible(self):
        # Happened an hour ago in the world, but we only learn of it in 6h.
        self.engine.ingest([news("SOL bridge exploit drained funds",
                                 event_time=NOW - HOUR,
                                 observed_at=NOW + 6 * HOUR)])
        self.assertEqual(self.repo.news_visible_at(NOW), [],
                         "an unobserved event must not be visible")

    def test_event_becomes_visible_only_after_observed_at(self):
        obs = NOW + 6 * HOUR
        self.engine.ingest([news("SOL bridge exploit drained funds",
                                 event_time=NOW - HOUR, observed_at=obs)])
        self.assertEqual(len(self.repo.news_visible_at(obs - 1)), 0)
        self.assertEqual(len(self.repo.news_visible_at(obs)), 1)
        self.assertEqual(len(self.repo.news_visible_at(obs + HOUR)), 1)

    def test_visibility_uses_observed_at_not_event_time(self):
        # An OLD event we only just heard about IS actionable now...
        self.engine.ingest([news("SOL network halt", event_time=NOW - 90 * HOUR,
                                 observed_at=NOW - HOUR, assets=("SOL",))])
        # ...while a RECENT event we have not heard about yet is not.
        self.engine.ingest([news("ETH chain halted", event_time=NOW - HOUR,
                                 observed_at=NOW + HOUR, assets=("ETH",))])
        visible = self.repo.news_visible_at(NOW, since_ms=NOW - 200 * HOUR)
        headlines = [v["headline"] for v in visible]
        self.assertEqual(len(headlines), 1)
        self.assertIn("SOL", headlines[0])

    def test_context_for_respects_as_of(self):
        self.engine.ingest([news("SOL exploit", event_time=NOW - 2 * HOUR,
                                 observed_at=NOW + HOUR)])
        self.assertEqual(self.engine.context_for(["SOL/USDT"], NOW), {})
        later = self.engine.context_for(["SOL/USDT"], NOW + 2 * HOUR)
        self.assertIn("SOL/USDT", later)

    def test_context_returns_the_worst_visible_event(self):
        self.engine.ingest([
            news("SOL listing on major venue", event_time=NOW - 3 * HOUR,
                 observed_at=NOW - 3 * HOUR),
            news("SOL exploit drained funds", event_time=NOW - 2 * HOUR,
                 observed_at=NOW - 2 * HOUR),
        ])
        ctx = self.engine.context_for(["SOL/USDT"], NOW)
        self.assertEqual(ctx["SOL/USDT"]["category"], "exploit")
        self.assertEqual(ctx["SOL/USDT"]["severity"], 5)

    def test_visibility_survives_restart(self):
        obs = NOW + 6 * HOUR
        self.engine.ingest([news("SOL exploit", event_time=NOW,
                                 observed_at=obs)])
        reopened = open_repo(self.path)
        self.assertEqual(reopened.news_visible_at(NOW), [])
        self.assertEqual(len(reopened.news_visible_at(obs)), 1)


class TestNewsClassification(unittest.TestCase):
    def test_classify_is_deterministic(self):
        h = "Major DeFi protocol hacked, funds drained"
        self.assertEqual(classify(h), classify(h))

    def test_severity_ordering(self):
        exploit = classify("protocol exploit drained funds")[1]
        regulatory = classify("SEC charges exchange")[1]
        listing = classify("exchange will list token")[1]
        self.assertGreater(exploit, regulatory)
        self.assertGreater(regulatory, listing)

    def test_unknown_headline_is_neutral(self):
        cat, sev, sent = classify("token price moved today")
        self.assertEqual(cat, "other")
        self.assertEqual(sev, 0)
        self.assertEqual(sent, "neutral")

    def test_highest_severity_match_wins(self):
        cat, sev, _ = classify("mainnet upgrade delayed after exploit")
        self.assertEqual(cat, "exploit")
        self.assertEqual(sev, 5)

    def test_event_id_is_stable_so_refetching_never_duplicates(self):
        item = news("SOL exploit", event_time=NOW, observed_at=NOW)
        self.assertEqual(event_id(item), event_id(item))
        repo, _ = temp_repo()
        eng = NewsEngine(repo)
        eng.ingest([item])
        eng.ingest([item])
        self.assertEqual(len(repo.news_visible_at(NOW + HOUR)), 1)

    def test_corroboration_raises_confidence_but_tier_dominates(self):
        lone_social = confidence_for(4, 1)
        many_social = confidence_for(4, 5)
        lone_tier1 = confidence_for(1, 1)
        self.assertGreater(many_social, lone_social)
        self.assertGreater(lone_tier1, many_social)


class TestNewsBlocking(unittest.TestCase):
    def setUp(self):
        self.repo, _ = temp_repo()
        self.engine = NewsEngine(self.repo, block_severity=4)

    def test_severe_event_from_trusted_source_blocks_the_asset(self):
        self.engine.ingest([news("SOL bridge exploit drained funds",
                                 event_time=NOW, observed_at=NOW, tier=1)])
        self.assertIn("SOL", self.repo.blocked_symbols(NOW + HOUR))

    def test_rumour_on_social_never_blocks(self):
        self.engine.ingest([news("SOL bridge exploit drained funds",
                                 event_time=NOW, observed_at=NOW, tier=4,
                                 source="anon_account")])
        self.assertNotIn("SOL", self.repo.blocked_symbols(NOW + HOUR))

    def test_low_severity_news_never_blocks(self):
        self.engine.ingest([news("SOL partnership announced", event_time=NOW,
                                 observed_at=NOW, tier=1)])
        self.assertNotIn("SOL", self.repo.blocked_symbols(NOW + HOUR))

    def test_block_expires(self):
        eng = NewsEngine(self.repo, block_severity=4, block_hours=24)
        eng.ingest([news("SOL chain halted", event_time=NOW, observed_at=NOW,
                         tier=1)])
        self.assertIn("SOL", self.repo.blocked_symbols(NOW + 23 * HOUR))
        self.assertNotIn("SOL", self.repo.blocked_symbols(NOW + 25 * HOUR))


class TestTokenEvents(unittest.TestCase):
    def setUp(self):
        self.repo, _ = temp_repo()
        self.cal = TokenEventCalendar(self.repo, block_hours=12)

    def test_upcoming_unlock_blocks_entry(self):
        self.cal.add("SOL/USDT", "unlock", event_time=NOW + 6 * HOUR,
                     observed_at=NOW - HOUR, severity=3)
        self.assertIsNotNone(self.cal.blocking_event("SOL/USDT", NOW))

    def test_event_outside_the_forward_window_does_not_block(self):
        self.cal.add("SOL/USDT", "unlock", event_time=NOW + 48 * HOUR,
                     observed_at=NOW - HOUR, severity=3)
        self.assertIsNone(self.cal.blocking_event("SOL/USDT", NOW))

    def test_unobserved_event_does_not_block(self):
        # scheduled soon, but we have not learned about it yet
        self.cal.add("SOL/USDT", "unlock", event_time=NOW + 6 * HOUR,
                     observed_at=NOW + 5 * HOUR, severity=3)
        self.assertIsNone(self.cal.blocking_event("SOL/USDT", NOW))
        self.assertIsNotNone(self.cal.blocking_event("SOL/USDT", NOW + 5 * HOUR))

    def test_trivial_event_does_not_block(self):
        self.cal.add("SOL/USDT", "ama", event_time=NOW + 2 * HOUR,
                     observed_at=NOW - HOUR, severity=1)
        self.assertIsNone(self.cal.blocking_event("SOL/USDT", NOW))

    def test_other_symbols_unaffected(self):
        self.cal.add("SOL/USDT", "unlock", event_time=NOW + 2 * HOUR,
                     observed_at=NOW - HOUR, severity=3)
        self.assertIsNone(self.cal.blocking_event("ETH/USDT", NOW))


# ============================================================ research DB
def make_signal(symbol="SOL/USDT", ts=NOW, score=61.0, reject=""):
    return Signal(symbol=symbol, strategy="trend_breakout", version="v1",
                  candle_id=f"{symbol}|1h|{ts}", ts_ms=ts, ref_price=100.0,
                  stop_price=95.0, atr=2.5, score=score, passed=not reject,
                  reject_reason=reject,
                  features={"adx": np.float64(27.3), "rsi": 61.0,
                            "atr": 2.5, "htf_regime": "bull",
                            "nan_field": float("nan")},
                  components={"adx": 8.0, "breakout": 12.0})


class TestResearchJournal(unittest.TestCase):
    def setUp(self):
        self.repo, self.path = temp_repo()
        self.journal = ResearchJournal(self.repo)

    def test_records_accepted_and_rejected_decisions(self):
        self.journal.record(make_signal(), "ENTERED")
        self.journal.record(make_signal(symbol="ETH/USDT", reject="adx too low"),
                            "REJECTED_STRATEGY")
        obs = self.repo.get_observations()
        self.assertEqual({o["decision"] for o in obs},
                         {"ENTERED", "REJECTED_STRATEGY"})

    def test_rejection_reason_is_persisted(self):
        self.journal.record(make_signal(reject="rel_volume 0.8 < 1.2"),
                            "REJECTED_STRATEGY")
        obs = self.repo.get_observations()[0]
        self.assertIn("rel_volume", obs["reject_reason"])

    def test_numpy_and_nan_are_sanitised_for_storage(self):
        # sqlite/JSON cannot take numpy scalars or NaN; storing must not raise
        self.journal.record(make_signal(), "ENTERED")
        obs = self.repo.get_observations()[0]
        feats = obs["features"]
        self.assertIsInstance(feats, dict, "repo should return features decoded")
        self.assertEqual(feats["adx"], 27.3)
        self.assertIsNone(feats["nan_field"])

    def test_ranked_out_decision_is_recorded(self):
        self.journal.record(make_signal(), "RANKED_OUT", rank=7)
        self.assertEqual(self.repo.get_observations()[0]["decision"], "RANKED_OUT")

    def test_observations_survive_restart(self):
        self.journal.record(make_signal(), "ENTERED")
        self.assertEqual(len(open_repo(self.path).get_observations()), 1)

    def test_entry_journal_captures_the_full_snapshot(self):
        snap = self.journal.entry_journal(make_signal(), {"cash": 10_000.0})
        for k in ("symbol", "signal_ts", "candle_id", "strategy_version",
                  "entry_ref_price", "adx", "rsi", "htf_regime",
                  "signal_score", "initial_stop", "cash"):
            self.assertIn(k, snap)


class TestCounterfactuals(unittest.TestCase):
    def setUp(self):
        self.repo, _ = temp_repo()
        self.journal = ResearchJournal(self.repo)
        self.tracker = CounterfactualTracker(self.repo, horizons_h=[4])

    def _series(self, final_price):
        closes = np.linspace(100.0, final_price, 12)
        return make_series("SOL/USDT", "1h", closes, NOW - 6 * HOUR)

    def test_counterfactual_never_creates_a_position(self):
        self.journal.record(make_signal(ts=NOW - 6 * HOUR, reject="adx too low"),
                            "REJECTED_STRATEGY")
        self.tracker.evaluate_pending(lambda s: self._series(110.0))
        self.assertEqual(len(self.repo.get_positions()), 0)
        self.assertEqual(len(self.repo.get_trades()), 0)

    def test_hypothetical_return_is_computed_and_flagged(self):
        self.journal.record(make_signal(ts=NOW - 6 * HOUR, reject="adx too low"),
                            "REJECTED_STRATEGY")
        n = self.tracker.evaluate_pending(lambda s: self._series(110.0))
        self.assertEqual(n, 1)
        rows = self.repo.conn.execute(
            "SELECT * FROM counterfactuals").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertGreater(dict(rows[0])["return_pct"], 0)

    def test_report_flags_insufficient_sample(self):
        for i in range(3):
            self.journal.record(
                make_signal(ts=NOW - 6 * HOUR + i, reject="adx too low"),
                "REJECTED_STRATEGY")
        self.tracker.evaluate_pending(lambda s: self._series(110.0))
        rep = self.tracker.filter_report(min_sample=20)
        self.assertTrue(rep)
        self.assertFalse(rep[0]["sufficient_sample"])
        self.assertEqual(rep[0]["label"], "HYPOTHETICAL / NOT EXECUTED")

    def test_missing_data_is_skipped_not_guessed(self):
        self.journal.record(make_signal(ts=NOW - 6 * HOUR, reject="adx too low"),
                            "REJECTED_STRATEGY")
        self.assertEqual(self.tracker.evaluate_pending(lambda s: None), 0)


# ============================================================ universe
def meta(symbol, active=True):
    base = symbol.split("/")[0]
    return MarketMeta(symbol, base, "USDT", active, 4, 4, 0.0001, 10.0)


def ticker(vol, spread_bps=5.0, last=100.0):
    half = spread_bps / 2e4
    return {"quoteVolume": vol, "last": last,
            "bid": last * (1 - half), "ask": last * (1 + half)}


class TestUniverseFiltering(unittest.TestCase):
    def setUp(self):
        self.cfg = test_config().universe
        self.b = UniverseBuilder(self.cfg)

    def test_stablecoins_rejected(self):
        self.assertEqual(self.b.static_reject(meta("USDC/USDT")), "stablecoin")

    def test_leveraged_tokens_rejected(self):
        for sym in ("BTC3L/USDT", "ETH3S/USDT", "BTCUP/USDT"):
            self.assertEqual(self.b.static_reject(meta(sym)), "leveraged token",
                             f"{sym} should be rejected as leveraged")

    def test_wrapped_duplicates_rejected(self):
        self.assertIn("wrapped", self.b.static_reject(meta("WBTC/USDT")))

    def test_inactive_market_rejected(self):
        self.assertIn("inactive", self.b.static_reject(meta("FOO/USDT", active=False)))

    def test_ordinary_market_passes_static_filters(self):
        self.assertEqual(self.b.static_reject(meta("SOL/USDT")), "")

    def test_thin_volume_rejected_with_a_reason(self):
        markets = {"SOL/USDT": meta("SOL/USDT")}
        tickers = {"SOL/USDT": ticker(self.cfg.min_dollar_volume_24h / 10)}
        keep, audit = self.b.build_candidates(markets, tickers)
        self.assertEqual(keep, [])
        self.assertIn("volume", audit[0]["reject_reason"])

    def test_wide_spread_rejected(self):
        markets = {"SOL/USDT": meta("SOL/USDT")}
        tickers = {"SOL/USDT": ticker(self.cfg.min_dollar_volume_24h * 10,
                                      spread_bps=self.cfg.max_spread_bps * 5)}
        keep, audit = self.b.build_candidates(markets, tickers)
        self.assertEqual(keep, [])
        self.assertIn("spread", audit[0]["reject_reason"])

    def test_liquid_market_survives(self):
        markets = {"SOL/USDT": meta("SOL/USDT")}
        tickers = {"SOL/USDT": ticker(self.cfg.min_dollar_volume_24h * 10,
                                      spread_bps=3.0)}
        keep, _ = self.b.build_candidates(markets, tickers)
        self.assertEqual(keep, ["SOL/USDT"])

    def test_ranking_is_by_dollar_volume_descending(self):
        markets = {s: meta(s) for s in ("A/USDT", "B/USDT", "C/USDT")}
        v = self.cfg.min_dollar_volume_24h * 10
        tickers = {"A/USDT": ticker(v), "B/USDT": ticker(v * 3),
                   "C/USDT": ticker(v * 2)}
        keep, _ = self.b.build_candidates(markets, tickers)
        self.assertEqual(keep, ["B/USDT", "C/USDT", "A/USDT"])

    def test_outside_top_n_is_cut(self):
        cfg = test_config().universe
        cfg.top_n = 2
        b = UniverseBuilder(cfg)
        markets = {s: meta(s) for s in ("A/USDT", "B/USDT", "C/USDT")}
        v = cfg.min_dollar_volume_24h * 10
        tickers = {"A/USDT": ticker(v * 3), "B/USDT": ticker(v * 2),
                   "C/USDT": ticker(v)}
        keep, audit = b.build_candidates(markets, tickers)
        self.assertEqual(keep, ["A/USDT", "B/USDT"])
        cut = [r for r in audit if r["symbol"] == "C/USDT"][0]
        self.assertIn("outside top 2", cut["reject_reason"])

    def test_every_rejection_is_audited(self):
        markets = {s: meta(s) for s in ("USDC/USDT", "SOL/USDT")}
        v = self.cfg.min_dollar_volume_24h * 10
        tickers = {"USDC/USDT": ticker(v * 5), "SOL/USDT": ticker(v)}
        keep, audit = self.b.build_candidates(markets, tickers)
        self.assertEqual(len(audit), 2, "every symbol considered must be logged")
        self.assertTrue(any(r["reject_reason"] for r in audit))

    def test_short_history_rejected(self):
        s = make_series("SOL/USDT", "1h", flat_closes(50), NOW - 50 * HOUR)
        self.assertIn("1h candles", self.b.filter_by_history("SOL/USDT", s))

    def test_missing_data_rejected(self):
        self.assertIn("no candle data",
                      self.b.filter_by_history("SOL/USDT", None))

    def test_too_quiet_and_too_volatile_both_rejected(self):
        cfg = test_config().universe
        cfg.min_candles_1h = 10
        cfg.min_market_age_days = 0
        b = UniverseBuilder(cfg)
        s = make_series("SOL/USDT", "1h", flat_closes(400), NOW - 400 * HOUR)
        self.assertIn("too quiet",
                      b.filter_by_history("SOL/USDT", s, atr_pct=cfg.min_atr_pct / 2))
        self.assertIn("too volatile",
                      b.filter_by_history("SOL/USDT", s, atr_pct=cfg.max_atr_pct * 2))
        self.assertEqual(
            b.filter_by_history("SOL/USDT", s,
                                atr_pct=(cfg.min_atr_pct + cfg.max_atr_pct) / 2), "")


if __name__ == "__main__":
    unittest.main()
