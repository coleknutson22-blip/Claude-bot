"""Entry quote validation: NEW ENTRIES FAIL CLOSED.

DEFECT UNDER TEST
-----------------
`spread_acceptable(None)` returned `(True, 0.0)` -- a missing quote was read as
a perfect zero-bps spread. Combined with `_safe_quote()` swallowing every fetch
error and returning None, a total ticker outage silently became "the book looks
great", and the engine sized and entered against a price nobody had confirmed.

Corrected rule: a new position may only be opened against a quote we can
actually trust. Missing, invalid, stale, malformed, crossed, wide or
implausible -> no entry, with the reason written to the research journal.

Exits are deliberately NOT gated this way. An open position must stay
manageable whether or not a quote is available; that is asserted here too.
"""
import unittest

import numpy as np

from crypto_edge.execution.paper_broker import PaperBroker
from crypto_edge.models import Quote
from crypto_edge.timeutils import now_ms
from helpers import breakout_closes, trend_closes
from test_engine import build_engine, rebuild_feed

NOW = 1_700_000_000_000


def broker(**kw):
    return PaperBroker(kw.pop("taker_fee_bps", 0.0), kw.pop("slippage_bps", 0.0),
                       kw.pop("stop_slippage_bps", 0.0),
                       use_book_spread=kw.pop("use_book_spread", True),
                       max_spread_bps_entry=kw.pop("max_spread_bps_entry", 25.0))


def good_quote(now=NOW, bid=99.95, ask=100.05, last=100.0):
    return Quote("SOL/USDT", bid, ask, last, now)


class TestQuoteValidation(unittest.TestCase):
    def setUp(self):
        self.b = broker()

    # ---------------------------------------------------------- happy path
    def test_a_normal_quote_passes(self):
        r = self.b.validate_entry_quote(good_quote(), ref_price=100.0, now=NOW)
        self.assertTrue(r.ok, r.reason)
        self.assertAlmostEqual(r.spread_bps, 10.0, places=3)
        self.assertAlmostEqual(r.age_s, 0.0, places=6)

    # ------------------------------------------------------------- missing
    def test_missing_quote_is_rejected(self):
        r = self.b.validate_entry_quote(None, ref_price=100.0, now=NOW)
        self.assertFalse(r.ok)
        self.assertIn("unavailable", r.reason)

    def test_spread_acceptable_no_longer_fails_open_on_none(self):
        """The exact original defect, pinned at the broker level."""
        ok, bps = self.b.spread_acceptable(None)
        self.assertFalse(ok, "a missing book must never read as an acceptable spread")
        self.assertEqual(bps, float("inf"))

    # ------------------------------------------------------------- invalid
    def test_zero_and_negative_prices_are_rejected(self):
        for bid, ask, last in ((0.0, 100.05, 100.0), (99.95, 0.0, 100.0),
                               (99.95, 100.05, 0.0), (-1.0, 100.05, 100.0)):
            r = self.b.validate_entry_quote(Quote("S", bid, ask, last, NOW),
                                            ref_price=100.0, now=NOW)
            self.assertFalse(r.ok, f"bid={bid} ask={ask} last={last} must be rejected")
            self.assertIn("invalid", r.reason)

    def test_non_finite_prices_are_rejected(self):
        for bad in (float("nan"), float("inf")):
            r = self.b.validate_entry_quote(Quote("S", bad, 100.05, 100.0, NOW),
                                            ref_price=100.0, now=NOW)
            self.assertFalse(r.ok)

    def test_crossed_book_is_rejected(self):
        r = self.b.validate_entry_quote(Quote("S", 101.0, 99.0, 100.0, NOW),
                                        ref_price=100.0, now=NOW)
        self.assertFalse(r.ok)
        self.assertIn("crossed", r.reason)

    def test_missing_timestamp_is_rejected(self):
        r = self.b.validate_entry_quote(Quote("S", 99.95, 100.05, 100.0, 0),
                                        ref_price=100.0, now=NOW)
        self.assertFalse(r.ok)
        self.assertIn("timestamp", r.reason)

    # --------------------------------------------------------------- stale
    def test_stale_quote_is_rejected(self):
        stale = good_quote(now=NOW - 600_000)          # 10 minutes old
        r = self.b.validate_entry_quote(stale, ref_price=100.0, now=NOW,
                                        max_age_s=90)
        self.assertFalse(r.ok)
        self.assertIn("stale", r.reason)
        self.assertAlmostEqual(r.age_s, 600.0, places=3)

    def test_quote_just_inside_the_age_limit_passes(self):
        r = self.b.validate_entry_quote(good_quote(now=NOW - 80_000),
                                        ref_price=100.0, now=NOW, max_age_s=90)
        self.assertTrue(r.ok, r.reason)

    def test_future_stamped_quote_is_rejected(self):
        r = self.b.validate_entry_quote(good_quote(now=NOW + 600_000),
                                        ref_price=100.0, now=NOW,
                                        max_future_skew_s=30)
        self.assertFalse(r.ok)
        self.assertIn("future", r.reason)

    def test_small_forward_skew_is_tolerated(self):
        r = self.b.validate_entry_quote(good_quote(now=NOW + 5_000),
                                        ref_price=100.0, now=NOW,
                                        max_future_skew_s=30)
        self.assertTrue(r.ok, r.reason)

    # ----------------------------------------------------------- malformed
    def test_object_without_quote_fields_is_rejected(self):
        class NotAQuote:
            bid = 99.0
        r = self.b.validate_entry_quote(NotAQuote(), ref_price=100.0, now=NOW)
        self.assertFalse(r.ok)
        self.assertIn("malformed", r.reason)

    def test_non_numeric_fields_are_rejected(self):
        class Weird:
            bid, ask, last, ts_ms = "n/a", "n/a", "n/a", "n/a"
        r = self.b.validate_entry_quote(Weird(), ref_price=100.0, now=NOW)
        self.assertFalse(r.ok)
        self.assertIn("malformed", r.reason)

    # -------------------------------------------------------------- spread
    def test_wide_spread_is_rejected_with_the_measured_value(self):
        r = self.b.validate_entry_quote(Quote("S", 99.0, 101.0, 100.0, NOW),
                                        ref_price=100.0, now=NOW)
        self.assertFalse(r.ok)
        self.assertIn("spread", r.reason)
        self.assertAlmostEqual(r.spread_bps, 200.0, places=3)

    # ----------------------------------------------------------- deviation
    def test_quote_far_from_the_signal_reference_is_rejected(self):
        """A quote 40% away from the closed-candle reference is bad data, or a
        market that has already made the move without us. Either way: no entry."""
        r = self.b.validate_entry_quote(good_quote(bid=139.9, ask=140.1, last=140.0),
                                        ref_price=100.0, now=NOW,
                                        max_deviation_pct=10.0)
        self.assertFalse(r.ok)
        self.assertIn("deviates", r.reason)

    def test_ordinary_drift_from_the_reference_is_allowed(self):
        r = self.b.validate_entry_quote(good_quote(bid=100.9, ask=101.1, last=101.0),
                                        ref_price=100.0, now=NOW,
                                        max_deviation_pct=10.0)
        self.assertTrue(r.ok, r.reason)


class TestEngineFailsClosedWithoutAQuote(unittest.TestCase):
    """End-to-end: the gate is actually wired into the entry path."""

    def setUp(self):
        self.closes = {"BTC/USDT": trend_closes(400, seed=4),
                       "SOL/USDT": breakout_closes(400, seed=11)}

    def _run(self, patch_feed):
        engine, feed, transport, repo = build_engine(self.closes)
        patch_feed(feed)
        engine.cycle()
        return engine, repo

    def test_baseline_entry_happens_with_a_healthy_quote(self):
        engine, repo = self._run(lambda feed: None)
        self.assertEqual(len(engine.account.positions()), 1,
                         "control case: a good quote must still trade")

    def test_no_entry_when_the_quote_is_unavailable(self):
        def kill(feed):
            feed.fetch_quote = lambda symbol: None
        engine, repo = self._run(kill)
        self.assertEqual(len(engine.account.positions()), 0)
        self.assertTrue(self._rejected_for(repo, "quote unavailable"))

    def test_no_entry_when_the_quote_fetch_raises(self):
        def boom(feed):
            def raiser(symbol):
                raise RuntimeError("exchange 502")
            feed.fetch_quote = raiser
        engine, repo = self._run(boom)
        self.assertEqual(len(engine.account.positions()), 0)
        self.assertTrue(self._rejected_for(repo, "quote unavailable"))

    def test_no_entry_when_the_quote_is_stale(self):
        def stale(feed):
            feed.fetch_quote = lambda symbol: Quote(symbol, 99.0, 99.1, 99.05,
                                                    now_ms() - 3_600_000)
        engine, repo = self._run(stale)
        self.assertEqual(len(engine.account.positions()), 0)
        self.assertTrue(self._rejected_for(repo, "stale"))

    def test_no_entry_when_the_quote_is_invalid(self):
        def invalid(feed):
            feed.fetch_quote = lambda symbol: Quote(symbol, 0.0, 0.0, 0.0, now_ms())
        engine, repo = self._run(invalid)
        self.assertEqual(len(engine.account.positions()), 0)
        self.assertTrue(self._rejected_for(repo, "invalid"))

    def test_no_entry_when_the_quote_is_malformed(self):
        class Broken:
            bid = 1.0
        def malformed(feed):
            feed.fetch_quote = lambda symbol: Broken()
        engine, repo = self._run(malformed)
        self.assertEqual(len(engine.account.positions()), 0)
        self.assertTrue(self._rejected_for(repo, "malformed"))

    def test_no_entry_when_the_spread_is_blown_out(self):
        def wide(feed):
            feed.fetch_quote = lambda symbol: Quote(symbol, 90.0, 110.0, 100.0, now_ms())
        engine, repo = self._run(wide)
        self.assertEqual(len(engine.account.positions()), 0)
        self.assertTrue(self._rejected_for(repo, "spread"))

    def test_the_rejection_reason_is_journalled_not_just_logged(self):
        def kill(feed):
            feed.fetch_quote = lambda symbol: None
        engine, repo = self._run(kill)
        rejects = [o for o in repo.get_observations()
                   if o["decision"] == "REJECTED_RISK"]
        self.assertTrue(rejects, "the rejection must be recorded as an observation")
        self.assertIn("entry blocked", rejects[0]["reject_reason"])
        self.assertEqual(rejects[0]["symbol"], "SOL/USDT")

    @staticmethod
    def _rejected_for(repo, fragment):
        return any(fragment in (o["reject_reason"] or "")
                   for o in repo.get_observations())


class TestPositionManagementSurvivesQuoteLoss(unittest.TestCase):
    """A quote outage must block NEW risk, never trap EXISTING risk."""

    def test_a_stop_still_fires_with_no_quote_available(self):
        closes = {"BTC/USDT": trend_closes(401, seed=4),
                  "SOL/USDT": breakout_closes(400, seed=11)}
        engine, feed, transport, repo = build_engine(
            {"BTC/USDT": closes["BTC/USDT"][:400], "SOL/USDT": closes["SOL/USDT"]})
        engine.cycle()
        self.assertEqual(len(engine.account.positions()), 1)
        pos = engine.account.positions()[0]

        # the ticker endpoint dies AFTER we are already in the trade
        feed.fetch_quote = lambda symbol: None
        rebuild_feed(feed, {
            "BTC/USDT": closes["BTC/USDT"][:401],
            "SOL/USDT": np.append(closes["SOL/USDT"], pos.current_stop * 0.90),
        }, vol_spike_idx=-2)
        engine.cycle()

        self.assertEqual(len(engine.account.positions()), 0,
                         "the stop must still execute without a quote")
        trades = repo.get_trades()
        self.assertEqual(len(trades), 1)
        self.assertIn(trades[0]["exit_reason"], ("stop_loss", "stop_gap"))

    def test_a_structurally_broken_quote_does_not_price_an_exit(self):
        """An exit is never blocked for want of a quote -- but a nonsense bid
        must not become the fill price either. It falls back to the reference."""
        closes = {"BTC/USDT": trend_closes(400, seed=4),
                  "SOL/USDT": breakout_closes(400, seed=11)}
        engine, feed, transport, repo = build_engine(closes)
        engine.cycle()
        pos = engine.account.positions()[0]

        # a bid of 0.01 on a ~140 asset: garbage the exchange should never send
        feed.fetch_quote = lambda symbol: Quote(symbol, -1.0, 0.0, 0.0, now_ms())
        exit_fill = engine.broker.sell(
            pos.symbol, pos.qty, 140.0,
            None if engine.broker.quote_structure_error(feed.fetch_quote(pos.symbol))
            else feed.fetch_quote(pos.symbol), None)
        self.assertGreater(exit_fill.fill_price, 100.0,
                           "the exit must price off the reference, not the junk bid")

    def test_structure_check_accepts_a_good_quote(self):
        b = broker()
        self.assertEqual(b.quote_structure_error(good_quote()), "")
        self.assertIn("unavailable", b.quote_structure_error(None))

    def test_marks_and_equity_still_update_without_a_quote(self):
        closes = {"BTC/USDT": trend_closes(400, seed=4),
                  "SOL/USDT": breakout_closes(400, seed=11)}
        engine, feed, transport, repo = build_engine(closes)
        engine.cycle()
        self.assertEqual(len(engine.account.positions()), 1)
        feed.fetch_quote = lambda symbol: None
        engine.cycle()                       # must not raise
        self.assertGreater(engine.account.equity(engine.marks()), 0.0)


if __name__ == "__main__":
    unittest.main()
