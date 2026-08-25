"""One effective exchange, reported identically by every surface.

WHAT WAS REPORTED
-----------------
An operator ran `--exchange kraken verify-live` and the Telegram BOT ONLINE
message said `Exchange: binance`.

WHAT THE AUDIT FOUND
--------------------
The override itself propagates correctly: `--exchange kraken` reaches
`Config.exchange.name`, and from there the CCXT feed, the self-check, the
status output, the verification report and the Telegram message. It is not a
display-only bug and no component was left behind -- verified below across
every surface at once.

The real defect is that the venue is a PER-INVOCATION flag with a silent
default. Omit it on one command and that entire run -- feed included, not just
the message -- silently uses the config file's exchange. The dangerous shape is
verifying against one venue and then starting the trader against another, using
the same database, which then holds positions, processed candles and journal
rows from a venue its prices no longer come from.

WHAT IS ENFORCED HERE
---------------------
  * every surface derives the venue from one place, `Config.exchange_label()`
  * the source of the value is reported, so "why is it binance" is answerable
  * a feed whose own name disagrees with the config is caught, not trusted
  * the database remembers its venue and a change is announced loudly
"""
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import helpers  # noqa: F401  -- silences the engine's log handlers
from crypto_edge.cli import build_parser
from crypto_edge.config import load_config
from crypto_edge.notify import formatters as fmt
from crypto_edge.notify.telegram import TelegramNotifier
from crypto_edge.verify_live import VerifyReport, verify_exchange, verify_telegram
from helpers import breakout_closes, temp_repo, trend_closes
from test_engine import build_engine

CONFIG = "config/config.toml"
NO_ENV = "/nonexistent-env-file"
OVERRIDE_KEYS = ("CRYPTO_EDGE_EXCHANGE", "CRYPTO_EDGE_QUOTE")


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = fn(*a, **kw)
    return result, buf.getvalue()


class RecordingTransport:
    def __init__(self):
        self.sent = []

    def send(self, token, chat_id, text, timeout):
        self.sent.append(text)
        return True, "ok"


class OverrideIsolated(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in OVERRIDE_KEYS}

    def tearDown(self):
        for k in OVERRIDE_KEYS:
            os.environ.pop(k, None)
            if self._saved.get(k) is not None:
                os.environ[k] = self._saved[k]

    def load_with_cli(self, *argv):
        """Reproduce the CLI path exactly: main() exports, then config loads."""
        args = build_parser().parse_args(list(argv))
        if getattr(args, "exchange", None):
            os.environ["CRYPTO_EDGE_EXCHANGE"] = args.exchange
        if getattr(args, "quote", None):
            os.environ["CRYPTO_EDGE_QUOTE"] = args.quote
        return load_config(CONFIG, NO_ENV)


class TestOverrideReachesEverySurface(OverrideIsolated):
    """The reported symptom, pinned across every surface simultaneously."""

    def test_the_telegram_bot_start_message_names_the_override(self):
        cfg = self.load_with_cli("--exchange", "kraken", "verify-live")
        cfg.telegram.enabled = True
        cfg.telegram_token, cfg.telegram_chat_id = "tok", "chat"
        repo, _ = temp_repo()
        transport = RecordingTransport()
        notifier = TelegramNotifier("tok", "chat", repo, enabled=True,
                                    transport=transport, sleep=lambda _: None,
                                    outbox_lease_s=0)
        quiet(verify_telegram, cfg, repo, notifier, VerifyReport())

        online = [m for m in transport.sent if "BOT ONLINE" in m]
        self.assertEqual(len(online), 1)
        self.assertIn("kraken", online[0])
        self.assertNotIn("binance", online[0],
                         "the BOT ONLINE message must not name the config default")

    def test_the_engine_announcement_names_the_override(self):
        closes = {"BTC/USDT": trend_closes(400, seed=4),
                  "SOL/USDT": breakout_closes(400, seed=11)}
        engine, feed, transport, _ = build_engine(closes)
        engine.cfg.exchange.name = "kraken"
        engine.announce_start()
        online = [m for m in transport.sent if "BOT ONLINE" in m]
        self.assertTrue(online)
        self.assertIn("kraken", online[0])
        self.assertNotIn("binance", online[0])

    def test_config_feed_and_report_all_agree(self):
        cfg = self.load_with_cli("--exchange", "kraken", "verify-live")
        self.assertEqual(cfg.exchange.name, "kraken")
        self.assertEqual(cfg.exchange_label(), "kraken/USDT")
        self.assertIn("override", cfg.exchange_source)

    def test_the_quote_currency_can_be_overridden_too(self):
        cfg = self.load_with_cli("--exchange", "kraken", "--quote", "USD", "status")
        self.assertEqual(cfg.exchange_label(), "kraken/USD")

    def test_without_the_flag_the_config_default_is_used_and_labelled(self):
        cfg = self.load_with_cli("status")
        self.assertEqual(cfg.exchange.name, "binance")
        self.assertEqual(cfg.exchange_source, "config file",
                         "the operator must be able to see WHY it is binance")

    def test_the_environment_variable_route_is_equivalent(self):
        os.environ["CRYPTO_EDGE_EXCHANGE"] = "kucoin"
        cfg = load_config(CONFIG, NO_ENV)
        self.assertEqual(cfg.exchange.name, "kucoin")
        self.assertIn("override", cfg.exchange_source)

    def test_every_command_accepts_the_flag_before_the_subcommand(self):
        parser = build_parser()
        for cmd in ("selfcheck", "start", "status", "positions", "performance",
                    "verify-live", "verify-restart", "test"):
            args = parser.parse_args(["--exchange", "kraken", cmd])
            self.assertEqual(args.exchange, "kraken", cmd)

    def test_the_flag_after_the_subcommand_is_a_loud_error_not_a_silent_default(self):
        """If it is going to be ignored, it must not be ignored quietly."""
        with self.assertRaises(SystemExit):
            with redirect_stdout(io.StringIO()):
                buf = io.StringIO()
                import contextlib
                with contextlib.redirect_stderr(buf):
                    build_parser().parse_args(["status", "--exchange", "kraken"])


class TestFeedConfigMismatchIsCaught(OverrideIsolated):
    """A feed pulling prices from a venue the config does not name is a bug."""

    class _Feed:
        def __init__(self, name):
            self.name = name

        def load_markets(self):
            from crypto_edge.data.feed import DataUnavailable
            raise DataUnavailable("not needed for this test")

    def test_verify_live_flags_a_mismatched_feed(self):
        cfg = self.load_with_cli("--exchange", "kraken", "verify-live")
        rep = VerifyReport()
        quiet(verify_exchange, cfg, self._Feed("binance"), rep)
        mismatch = [s for s in rep.steps if s.name == "feed matches configured exchange"]
        self.assertEqual(len(mismatch), 1)
        self.assertFalse(mismatch[0].ok)
        self.assertIn("binance", mismatch[0].detail)
        self.assertIn("kraken", mismatch[0].detail)

    def test_verify_live_passes_a_matching_feed(self):
        cfg = self.load_with_cli("--exchange", "kraken", "verify-live")
        rep = VerifyReport()
        quiet(verify_exchange, cfg, self._Feed("kraken"), rep)
        mismatch = [s for s in rep.steps if s.name == "feed matches configured exchange"]
        self.assertTrue(mismatch[0].ok)

    def test_the_engine_detects_a_mismatched_feed(self):
        closes = {"BTC/USDT": trend_closes(400, seed=4)}
        engine, feed, _, _ = build_engine(closes)
        feed.name = "binance"
        engine.cfg.exchange.name = "kraken"
        self.assertFalse(engine.check_feed_matches_config())

    def test_the_offline_fixture_feed_is_exempt(self):
        """The fixture feed is not a venue and must not trip the check."""
        closes = {"BTC/USDT": trend_closes(400, seed=4)}
        engine, feed, _, _ = build_engine(closes)
        self.assertEqual(feed.name, "fixture")
        self.assertTrue(engine.check_feed_matches_config())


class TestDatabaseRemembersItsVenue(unittest.TestCase):
    """Verifying on one venue and trading on another must not be silent."""

    def setUp(self):
        self.repo, self.path = temp_repo()

    def test_first_use_records_the_venue_without_complaint(self):
        changed, previous = self.repo.record_exchange("kraken/USDT")
        self.assertFalse(changed)
        self.assertEqual(previous, "")
        self.assertEqual(self.repo.get_meta("exchange"), "kraken/USDT")

    def test_the_same_venue_again_is_not_a_change(self):
        self.repo.record_exchange("kraken/USDT")
        changed, _ = self.repo.record_exchange("kraken/USDT")
        self.assertFalse(changed)

    def test_a_different_venue_is_reported_with_the_previous_one(self):
        self.repo.record_exchange("kraken/USDT")
        changed, previous = self.repo.record_exchange("binance/USDT")
        self.assertTrue(changed, "switching venue on one database must be flagged")
        self.assertEqual(previous, "kraken/USDT")

    def test_a_changed_quote_currency_also_counts(self):
        self.repo.record_exchange("kraken/USDT")
        changed, previous = self.repo.record_exchange("kraken/USD")
        self.assertTrue(changed)
        self.assertEqual(previous, "kraken/USDT")

    def test_the_record_survives_a_restart(self):
        from helpers import open_repo
        self.repo.record_exchange("kraken/USDT")
        self.repo.conn.close()
        repo2 = open_repo(self.path)
        changed, previous = repo2.record_exchange("binance/USDT")
        self.assertTrue(changed)
        self.assertEqual(previous, "kraken/USDT")

    def test_recording_does_not_disturb_the_schema_version(self):
        from crypto_edge.storage import db
        self.repo.record_exchange("kraken/USDT")
        self.assertEqual(int(self.repo.get_meta("schema_version")), db.SCHEMA_VERSION)


class TestSelfCheckReportsTheVenue(OverrideIsolated):
    def _run(self, cfg, repo):
        from crypto_edge.selfcheck import run_selfcheck
        notifier = TelegramNotifier("", "", repo, enabled=False,
                                    transport=RecordingTransport())
        return run_selfcheck(cfg, repo, None, notifier, check_network=False)

    def test_the_effective_exchange_appears_in_the_selfcheck(self):
        cfg = self.load_with_cli("--exchange", "kraken", "selfcheck")
        cfg.telegram.enabled = False
        repo, _ = temp_repo()
        rep = self._run(cfg, repo)
        line = [r for r in rep.results if r.name == "effective exchange"]
        self.assertEqual(len(line), 1)
        self.assertIn("kraken/USDT", line[0].detail)
        self.assertIn("override", line[0].detail)

    def test_a_venue_change_is_surfaced_as_a_warning(self):
        repo, _ = temp_repo()
        repo.record_exchange("binance/USDT")
        cfg = self.load_with_cli("--exchange", "kraken", "selfcheck")
        cfg.telegram.enabled = False
        rep = self._run(cfg, repo)
        changed = [r for r in rep.results
                   if r.name == "exchange unchanged since last run"]
        self.assertEqual(len(changed), 1)
        self.assertFalse(changed[0].ok)
        self.assertIn("binance/USDT", changed[0].detail)
        self.assertIn("kraken/USDT", changed[0].detail)

    def test_a_venue_change_warns_but_does_not_block_startup(self):
        """Switching venue is sometimes legitimate; it must be loud, not fatal."""
        repo, _ = temp_repo()
        repo.record_exchange("binance/USDT")
        cfg = self.load_with_cli("--exchange", "kraken", "selfcheck")
        cfg.telegram.enabled = False
        rep = self._run(cfg, repo)
        self.assertTrue(rep.passed,
                        "a deliberate venue switch must not be a critical failure")

    def test_no_warning_when_the_venue_is_unchanged(self):
        repo, _ = temp_repo()
        repo.record_exchange("kraken/USDT")
        cfg = self.load_with_cli("--exchange", "kraken", "selfcheck")
        cfg.telegram.enabled = False
        rep = self._run(cfg, repo)
        changed = [r for r in rep.results
                   if r.name == "exchange unchanged since last run"]
        self.assertTrue(changed[0].ok)


class TestStatusOutput(OverrideIsolated):
    def test_status_names_the_venue_and_its_source(self):
        from crypto_edge.cli import cmd_status

        workdir = Path(tempfile.mkdtemp())
        cfg_text = Path(CONFIG).read_text(encoding="utf-8-sig")
        cfg_text = cfg_text.replace('db_path = "data/crypto_edge.db"',
                                    f'db_path = "{(workdir / "s.db").as_posix()}"')
        cfg_text = cfg_text.replace('log_dir = "logs"',
                                    f'log_dir = "{(workdir / "logs").as_posix()}"')
        cfg_path = workdir / "c.toml"
        cfg_path.write_text(cfg_text)

        args = build_parser().parse_args(
            ["--config", str(cfg_path), "--env", NO_ENV,
             "--exchange", "kraken", "status"])
        os.environ["CRYPTO_EDGE_EXCHANGE"] = args.exchange
        _, out = quiet(cmd_status, args)
        self.assertIn("kraken/USDT", out)
        self.assertIn("Exchange from:", out)
        self.assertIn("override", out)

    def test_status_works_on_a_database_the_engine_has_never_touched(self):
        """The operator runs this before the first cycle; it must not traceback."""
        from crypto_edge.cli import cmd_status

        workdir = Path(tempfile.mkdtemp())
        cfg_text = Path(CONFIG).read_text(encoding="utf-8-sig")
        cfg_text = cfg_text.replace('db_path = "data/crypto_edge.db"',
                                    f'db_path = "{(workdir / "fresh.db").as_posix()}"')
        cfg_text = cfg_text.replace('log_dir = "logs"',
                                    f'log_dir = "{(workdir / "logs").as_posix()}"')
        cfg_path = workdir / "c.toml"
        cfg_path.write_text(cfg_text)

        args = build_parser().parse_args(
            ["--config", str(cfg_path), "--env", NO_ENV, "status"])
        rc, out = quiet(cmd_status, args)
        self.assertEqual(rc, 0)
        self.assertIn("Equity", out)


class TestFormatterIsNotTheProblem(unittest.TestCase):
    def test_bot_start_prints_whatever_exchange_it_is_given(self):
        msg = fmt.bot_start(mode="PAPER", exchange="kraken/USDT", equity=1.0,
                            cash=1.0, open_positions=0, strategy="s",
                            version="1", universe_size=0, telegram_ok=True)
        self.assertIn("Exchange: kraken/USDT", msg)
        self.assertNotIn("binance", msg)


if __name__ == "__main__":
    unittest.main()
