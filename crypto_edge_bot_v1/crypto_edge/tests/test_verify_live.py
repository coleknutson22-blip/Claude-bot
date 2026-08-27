"""The live-verification harness, exercised offline against fixtures.

`verify-live` is the tool an operator runs before letting the bot run
continuously, so it must not be the one untested thing in the repository. The
network-facing steps are driven here through the deterministic FixtureFeed and
a fake Telegram transport: the harness logic (what it checks, what it calls a
failure, what it reports, and that it never prints a credential) is verified
offline, while the actual network behaviour is what running it for real proves.
"""
import io
import unittest
from contextlib import redirect_stdout

from crypto_edge.execution.paper_broker import PaperBroker
from crypto_edge.models import TS_LOCAL, Quote
from crypto_edge.notify.telegram import TelegramNotifier
from crypto_edge.timeutils import now_ms
from crypto_edge.verify_live import (VerifyReport, _mask, verify_cycle,
                                     verify_exchange, verify_pending_first,
                                     verify_quotes, verify_telegram,
                                     verify_universe)
from helpers import (breakout_closes, broad_provider, engine_config,
                     recent_start_ms, temp_repo, trend_closes)
from test_engine import build_engine


def quiet(fn, *a, **kw):
    """Run a verification step, capturing its printed report."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = fn(*a, **kw)
    return result, buf.getvalue()


class RecordingTransport:
    def __init__(self, ok=True):
        self.ok = ok
        self.sent = []

    def send(self, token, chat_id, text, timeout):
        if not self.ok:
            return False, "simulated outage"
        self.sent.append(text)
        return True, "ok"


class TestMasking(unittest.TestCase):
    """A verification tool that leaks the token is a liability, not a tool."""

    def test_a_token_is_never_reproduced_in_full(self):
        secret = "1234567890:AAHfakefakefakefakefakefakefakefake"
        masked = _mask(secret)
        self.assertNotIn(secret, masked)
        self.assertNotIn(secret[:12], masked)
        self.assertIn(str(len(secret)), masked)

    def test_short_secrets_reveal_nothing_at_all(self):
        self.assertEqual(_mask("abc123"), "<set>")

    def test_absent_secret_is_reported_as_unset(self):
        self.assertEqual(_mask(""), "<unset>")

    def test_telegram_step_does_not_print_the_credentials(self):
        repo, _ = temp_repo()
        cfg = engine_config()
        cfg.telegram.enabled = True
        cfg.telegram_token = "1234567890:AAHsupersecrettokenvalue000000000000"
        cfg.telegram_chat_id = "-1001234567890"
        notifier = TelegramNotifier(cfg.telegram_token, cfg.telegram_chat_id, repo,
                                    enabled=True, transport=RecordingTransport(),
                                    sleep=lambda _: None)
        rep = VerifyReport()
        _, out = quiet(verify_telegram, cfg, repo, notifier, rep)
        self.assertNotIn(cfg.telegram_token, out)
        self.assertNotIn(cfg.telegram_chat_id, out)
        self.assertIn("<set,", out)


class TestExchangeStep(unittest.TestCase):
    def setUp(self):
        self.closes = {"BTC/USDT": trend_closes(400, seed=4),
                       "SOL/USDT": breakout_closes(400, seed=11)}

    def test_a_healthy_feed_passes_every_critical_check(self):
        engine, feed, _, _ = build_engine(self.closes)
        rep = VerifyReport()
        out, _ = quiet(verify_exchange, engine.cfg, feed, rep)
        self.assertTrue(rep.passed, [s for s in rep.steps if not s.ok])
        self.assertIn("markets", out)
        self.assertIn("btc_1h", out)
        self.assertIn("btc_4h", out)

    def test_it_reports_both_timeframes(self):
        engine, feed, _, _ = build_engine(self.closes)
        rep = VerifyReport()
        quiet(verify_exchange, engine.cfg, feed, rep)
        self.assertIn("OK", rep.facts["btc_1h"])
        self.assertIn("OK", rep.facts["btc_4h"])

    def test_stale_candles_are_reported_as_a_failure(self):
        """A venue serving week-old candles must not read as verified."""
        from crypto_edge.data.fixture_feed import FixtureFeed, make_series
        from helpers import default_meta
        old = 1_600_000_000_000
        series, markets = {}, {}
        for sym, c in self.closes.items():
            series[(sym, "1h")] = make_series(sym, "1h", c, old)
            series[(sym, "4h")] = make_series(sym, "4h", c[::4], old)
            markets[sym] = default_meta(sym)
        engine, _, _, _ = build_engine(self.closes)
        rep = VerifyReport()
        quiet(verify_exchange, engine.cfg, FixtureFeed(series, markets), rep)
        self.assertFalse(rep.passed, "ancient candles must fail verification")

    def test_a_dead_feed_fails_loudly(self):
        from crypto_edge.data.feed import DataUnavailable

        class Dead:
            name = "dead"

            def load_markets(self):
                raise DataUnavailable("exchange unreachable")

        engine, _, _, _ = build_engine(self.closes)
        rep = VerifyReport()
        quiet(verify_exchange, engine.cfg, Dead(), rep)
        self.assertFalse(rep.passed)
        self.assertIn("FAILED", rep.facts["ccxt_result"])


class TestQuoteCalibrationStep(unittest.TestCase):
    def setUp(self):
        self.closes = {"BTC/USDT": trend_closes(400, seed=4),
                       "SOL/USDT": breakout_closes(400, seed=11)}

    def test_it_samples_and_reports_quote_age(self):
        engine, feed, _, _ = build_engine(self.closes)
        broker = PaperBroker(7.5, 6.0, 15.0)
        rep = VerifyReport()
        quiet(verify_quotes, engine.cfg, feed, broker, 3, 0.0, rep)
        self.assertTrue(rep.passed, [s for s in rep.steps if not s.ok])
        self.assertIn("median age", rep.facts["ts_findings"])
        self.assertIn("spread", rep.facts["quote_status"])

    def test_a_venue_without_timestamps_is_reported_honestly(self):
        """The report must say the age check was skipped, not that it passed."""
        engine, feed, _, _ = build_engine(self.closes)
        feed.fetch_quote = lambda s: Quote(s, 99.9, 100.1, 100.0, now_ms(),
                                           ts_source=TS_LOCAL)
        rep = VerifyReport()
        quiet(verify_quotes, engine.cfg, feed, PaperBroker(7.5, 6.0, 15.0), 2, 0.0, rep)
        findings = rep.facts["ts_findings"]
        self.assertIn("no ticker timestamp", findings)
        self.assertIn("skipped", findings)
        self.assertIn("do NOT set", findings,
                      "the operator needs to be told what NOT to configure")

    def test_a_venue_lagging_past_the_limit_is_flagged_not_excused(self):
        engine, feed, _, _ = build_engine(self.closes)
        feed.fetch_quote = lambda s: Quote(s, 99.9, 100.1, 100.0,
                                           now_ms() - 600_000)
        rep = VerifyReport()
        quiet(verify_quotes, engine.cfg, feed, PaperBroker(7.5, 6.0, 15.0), 2, 0.0, rep)
        self.assertFalse(rep.passed)
        self.assertIn("EXCEEDS", rep.facts["ts_findings"].upper() + "".join(
            s.detail for s in rep.steps))

    def test_an_absent_quote_fails_the_step(self):
        engine, feed, _, _ = build_engine(self.closes)
        feed.fetch_quote = lambda s: None
        rep = VerifyReport()
        quiet(verify_quotes, engine.cfg, feed, PaperBroker(7.5, 6.0, 15.0), 1, 0.0, rep)
        self.assertFalse(rep.passed)


class TestUniverseStep(unittest.TestCase):
    def setUp(self):
        self.closes = {"BTC/USDT": trend_closes(400, seed=4),
                       "SOL/USDT": breakout_closes(400, seed=11)}

    def test_it_verifies_cache_provenance_and_fallback(self):
        engine, feed, _, repo = build_engine(self.closes)
        rep = VerifyReport()
        quiet(verify_universe, engine.cfg, repo, engine.broad_universe,
              feed.load_markets(), feed.fetch_tickers(), rep)
        self.assertTrue(rep.passed, [s for s in rep.steps if not s.ok])
        names = [s.name for s in rep.steps]
        for expected in ("broad universe fetch", "cache written",
                         "source + timestamp stored", "content hash stored",
                         "stablecoins / wrapped removed", "ticker collisions",
                         "exchange intersection",
                         "provider outage falls back to cache"):
            self.assertIn(expected, names)
        self.assertGreaterEqual(rep.facts["intersecting"], 1)
        self.assertGreaterEqual(rep.facts["after_filter"], 1)

    def test_the_outage_probe_restores_the_real_provider(self):
        engine, feed, _, repo = build_engine(self.closes)
        original = engine.broad_universe.provider
        rep = VerifyReport()
        quiet(verify_universe, engine.cfg, repo, engine.broad_universe,
              feed.load_markets(), feed.fetch_tickers(), rep)
        self.assertIs(engine.broad_universe.provider, original,
                      "a diagnostic must not leave the system altered")

    def test_a_cold_start_with_no_provider_and_no_cache_is_reported(self):
        engine, feed, _, repo = build_engine(self.closes)
        engine.broad_universe.provider = None
        repo.conn.execute("DELETE FROM broad_universe_cache")
        rep = VerifyReport()
        quiet(verify_universe, engine.cfg, repo, engine.broad_universe,
              feed.load_markets(), feed.fetch_tickers(), rep)
        self.assertFalse(rep.passed)
        self.assertIn("FAILED", rep.facts["broad_result"])


class TestTelegramSteps(unittest.TestCase):
    def _notifier(self, repo, ok=True):
        return TelegramNotifier("tok", "chat", repo, enabled=True,
                                transport=RecordingTransport(ok),
                                sleep=lambda _: None, outbox_lease_s=0)

    def test_delivery_dedupe_and_outbox_are_all_checked(self):
        repo, _ = temp_repo()
        cfg = engine_config()
        cfg.telegram.enabled = True
        cfg.telegram_token, cfg.telegram_chat_id = "tok", "chat"
        notifier = self._notifier(repo)
        rep = VerifyReport()
        quiet(verify_telegram, cfg, repo, notifier, rep)
        self.assertTrue(rep.passed, [s for s in rep.steps if not s.ok])
        self.assertIn("OK", rep.facts["telegram"])

    def test_pending_then_recovered_is_proven_not_assumed(self):
        repo, _ = temp_repo()
        notifier = self._notifier(repo)
        rep = VerifyReport()
        quiet(verify_pending_first, notifier, repo, rep)
        self.assertTrue(rep.passed, [s for s in rep.steps if not s.ok])
        names = [s.name for s in rep.steps]
        self.assertIn("failed send stays PENDING", names)
        self.assertIn("pending message recovers to SENT", names)

    def test_the_probe_restores_the_real_transport(self):
        repo, _ = temp_repo()
        notifier = self._notifier(repo)
        original = notifier.transport
        rep = VerifyReport()
        quiet(verify_pending_first, notifier, repo, rep)
        self.assertIs(notifier.transport, original)

    def test_missing_credentials_fail_the_step(self):
        repo, _ = temp_repo()
        cfg = engine_config()
        cfg.telegram.enabled = True
        notifier = TelegramNotifier("", "", repo, enabled=True,
                                    transport=RecordingTransport())
        rep = VerifyReport()
        quiet(verify_telegram, cfg, repo, notifier, rep)
        self.assertFalse(rep.passed)
        self.assertIn("FAILED", rep.facts["telegram"])

    def test_disabled_telegram_is_skipped_not_failed(self):
        repo, _ = temp_repo()
        cfg = engine_config()
        cfg.telegram.enabled = False
        rep = VerifyReport()
        quiet(verify_telegram, cfg, repo, self._notifier(repo), rep)
        self.assertTrue(rep.passed)
        self.assertIn("SKIPPED", rep.facts["telegram"])


class TestCycleStep(unittest.TestCase):
    def test_a_full_cycle_is_driven_and_reported(self):
        closes = {"BTC/USDT": trend_closes(400, seed=4),
                  "SOL/USDT": breakout_closes(400, seed=11)}
        engine, feed, transport, repo = build_engine(closes)
        notifier = TelegramNotifier("tok", "chat", repo, enabled=True,
                                    transport=RecordingTransport(),
                                    sleep=lambda _: None)
        rep = VerifyReport()
        quiet(verify_cycle, engine.cfg, repo, feed, notifier, rep)
        self.assertTrue(rep.passed, [s for s in rep.steps if not s.ok])
        self.assertIn("OK", rep.facts["cycle"])
        names = [s.name for s in rep.steps]
        for expected in ("engine cycle completes", "universe resolved",
                         "BTC context built", "HTF context built",
                         "signals evaluated", "open positions managed",
                         "circuit breakers evaluated", "journal writes"):
            self.assertIn(expected, names)

    def test_zero_entries_is_not_a_failure(self):
        """Explicitly: a cycle that trades nothing is a valid live verification."""
        closes = {"BTC/USDT": trend_closes(400, seed=4),
                  "SOL/USDT": trend_closes(400, seed=9)}   # no breakout
        engine, feed, _, repo = build_engine(closes)
        notifier = TelegramNotifier("tok", "chat", repo, enabled=True,
                                    transport=RecordingTransport(),
                                    sleep=lambda _: None)
        rep = VerifyReport()
        quiet(verify_cycle, engine.cfg, repo, feed, notifier, rep)
        self.assertTrue(rep.passed)
        self.assertEqual(len(repo.get_positions()), 0)

    def test_a_crashing_cycle_is_reported_not_swallowed(self):
        closes = {"BTC/USDT": trend_closes(400, seed=4)}
        engine, feed, _, repo = build_engine(closes)

        def boom(*a, **kw):
            raise RuntimeError("exchange returned nonsense")

        feed.load_markets = boom
        notifier = TelegramNotifier("tok", "chat", repo, enabled=True,
                                    transport=RecordingTransport(),
                                    sleep=lambda _: None)
        rep = VerifyReport()
        quiet(verify_cycle, engine.cfg, repo, feed, notifier, rep)
        self.assertFalse(rep.passed)
        self.assertIn("FAILED", rep.facts["cycle"])


class TestReportRendering(unittest.TestCase):
    def test_summary_contains_every_requested_field(self):
        rep = VerifyReport()
        quiet(rep.add, "something", True, "fine")
        text = rep.render_summary()
        for field in ("EXCHANGE USED", "LIVE CCXT RESULT", "BROAD UNIVERSE RESULT",
                      "NUMBER OF ASSETS RETURNED", "NUMBER INTERSECTING EXCHANGE",
                      "NUMBER AFTER FILTERING", "BTC 1H DATA STATUS",
                      "BTC 4H DATA STATUS", "QUOTE/SPREAD STATUS",
                      "QUOTE TIMESTAMP FINDINGS", "TELEGRAM DELIVERY RESULT",
                      "COMPLETE ENGINE CYCLE RESULT"):
            self.assertIn(field, text)

    def test_an_info_failure_does_not_fail_the_run(self):
        rep = VerifyReport()
        quiet(rep.add, "informational", False, "meh", severity="INFO")
        self.assertTrue(rep.passed)

    def test_a_critical_failure_does(self):
        rep = VerifyReport()
        quiet(rep.add, "critical", False, "broken")
        self.assertFalse(rep.passed)
        self.assertIn("do not run continuously", rep.render_summary())


class TestSummaryCannotContradictItsOwnChecks(unittest.TestCase):
    """DEFECT: the detail section said FAIL and the summary said OK.

    A real Kraken/USD run printed

        [FAIL] data freshness (1h)
        ...
        BTC 1H DATA STATUS             OK (300 rows, 1 unclosed dropped, ...)

    because `rep.facts["btc_1h"]` was assigned the string "OK (...)"
    unconditionally, further down the same function that had just recorded the
    failure. The summary is the part of the report written to be trusted -- an
    operator scrolls to it precisely so they do not have to audit every line --
    so a summary that disagrees with the checks is worse than no summary.

    The fix is structural: summary lines are DERIVED from the recorded checks
    via `VerifyReport.fact()`, so the two cannot drift apart again. These tests
    hold that structure in place rather than the one call site that was wrong.
    """

    def test_a_failed_critical_check_overrules_an_ok_fact(self):
        rep = VerifyReport()
        rep.facts["btc_1h"] = "OK (405 rows, 1 unclosed dropped)"
        quiet(rep.add, "data freshness (1h)", False, "118.5 min behind",
              topic="btc_1h")
        line = [ln for ln in rep.render_summary().splitlines()
                if "BTC 1H DATA STATUS" in ln][0]
        self.assertNotIn("OK", line, "the summary must not claim OK")
        self.assertIn("FAILED", line)
        self.assertIn("data freshness (1h)", line, "and must name the check")

    def test_the_summary_names_every_failed_check_for_that_dataset(self):
        rep = VerifyReport()
        rep.facts["btc_4h"] = "OK"
        quiet(rep.add, "data freshness (4h)", False, "", topic="btc_4h")
        quiet(rep.add, "timestamps sane (4h)", False, "", topic="btc_4h")
        line = [ln for ln in rep.render_summary().splitlines()
                if "BTC 4H DATA STATUS" in ln][0]
        self.assertIn("data freshness (4h)", line)
        self.assertIn("timestamps sane (4h)", line)

    def test_a_passing_dataset_still_reports_its_fact(self):
        rep = VerifyReport()
        rep.facts["btc_1h"] = "OK (405 rows)"
        quiet(rep.add, "data freshness (1h)", True, "", topic="btc_1h")
        line = [ln for ln in rep.render_summary().splitlines()
                if "BTC 1H DATA STATUS" in ln][0]
        self.assertIn("OK (405 rows)", line)

    def test_a_failure_on_another_dataset_does_not_leak_across(self):
        rep = VerifyReport()
        rep.facts["btc_1h"] = "OK (1h fine)"
        rep.facts["btc_4h"] = "OK (4h fine)"
        quiet(rep.add, "data freshness (4h)", False, "", topic="btc_4h")
        summary = rep.render_summary()
        one = [ln for ln in summary.splitlines() if "BTC 1H DATA" in ln][0]
        four = [ln for ln in summary.splitlines() if "BTC 4H DATA" in ln][0]
        self.assertIn("OK (1h fine)", one, "1h passed and must still say so")
        self.assertIn("FAILED", four)

    def test_a_warning_does_not_overrule_the_fact(self):
        """Only CRITICAL failures invalidate a dataset; warnings are advisory."""
        from crypto_edge.verify_live import INFO
        rep = VerifyReport()
        rep.facts["btc_1h"] = "OK (405 rows)"
        quiet(rep.add, "something advisory", False, "", severity=INFO,
              topic="btc_1h")
        line = [ln for ln in rep.render_summary().splitlines()
                if "BTC 1H DATA STATUS" in ln][0]
        self.assertIn("OK (405 rows)", line)

    def test_the_verdict_and_the_summary_lines_agree(self):
        """If any line says FAILED, the verdict must not say all checks passed."""
        rep = VerifyReport()
        rep.facts["btc_1h"] = "OK"
        quiet(rep.add, "data freshness (1h)", False, "", topic="btc_1h")
        summary = rep.render_summary()
        self.assertIn("FAILED", summary)
        self.assertNotIn("ALL CRITICAL CHECKS PASSED", summary)
        self.assertFalse(rep.passed)

    def test_every_summary_topic_is_reachable_from_a_check(self):
        """A topic string typo would silently disable this protection.

        Each fact key rendered in the summary must be one that some check in
        verify_live actually tags, otherwise `fact()` can never overrule it.
        """
        import inspect

        import crypto_edge.verify_live as vl
        src = inspect.getsource(vl)
        for key in ("btc_1h", "btc_4h"):
            self.assertIn(f'topic=label', src,
                          "candle checks must be tagged with their fact key")
            self.assertIn(key, src)


if __name__ == "__main__":
    unittest.main()
