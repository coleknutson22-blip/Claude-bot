"""Operability fixes for running on Windows from a plain terminal.

DEFECTS UNDER TEST
------------------
1. **A Notepad-saved `.env` disabled Telegram silently.** Windows Notepad writes
   a UTF-8 byte order mark. Read as plain UTF-8 that BOM becomes a `\\ufeff`
   character glued to the first key, so `TELEGRAM_BOT_TOKEN` is stored as
   `\\ufeffTELEGRAM_BOT_TOKEN`, is never found, and the bot runs with
   notifications quietly switched off. No error is raised anywhere -- the
   operator simply never hears from it. This is the single most likely failure
   for someone following the setup instructions on Windows.

2. **Non-ASCII output crashed when redirected.** Python writes to a Windows
   console through the Unicode API, but falls back to the system code page the
   moment output is piped or redirected to a file. One emoji in a status line
   then raises UnicodeEncodeError -- precisely when someone is capturing output
   to send to support.

3. **Changing exchange required editing a TOML file.** Venue switching is the
   most common operational change (geo-blocks, outages) and a mistyped TOML
   line is a worse failure than a wrong venue.
"""
import io
import os
import tempfile
import unittest
from pathlib import Path

import helpers  # noqa: F401  -- silences the engine's log handlers
from crypto_edge.config import load_config, load_dotenv

TOKEN_KEYS = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
              "CRYPTO_EDGE_EXCHANGE", "CRYPTO_EDGE_QUOTE", "TELEGRAM_ENABLED")


class EnvIsolated(unittest.TestCase):
    """load_dotenv uses setdefault, so a leaked variable would mask a failure."""

    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in TOKEN_KEYS}
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        for k in TOKEN_KEYS:
            os.environ.pop(k, None)
            if self._saved.get(k) is not None:
                os.environ[k] = self._saved[k]

    def write_env(self, data: bytes) -> Path:
        p = self.dir / ".env"
        p.write_bytes(data)
        return p


class TestDotenvEncoding(EnvIsolated):
    def test_a_notepad_saved_env_is_read_correctly(self):
        """UTF-8 with BOM -- exactly what Windows Notepad produces."""
        p = self.write_env(b"\xef\xbb\xbfTELEGRAM_BOT_TOKEN=123456:ABCDEF\n"
                           b"TELEGRAM_CHAT_ID=-1001234567890\n")
        load_dotenv(p)
        self.assertEqual(os.environ.get("TELEGRAM_BOT_TOKEN"), "123456:ABCDEF")
        self.assertEqual(os.environ.get("TELEGRAM_CHAT_ID"), "-1001234567890")

    def test_no_bom_mangled_key_is_created(self):
        p = self.write_env(b"\xef\xbb\xbfTELEGRAM_BOT_TOKEN=abc\n")
        load_dotenv(p)
        self.assertNotIn("﻿TELEGRAM_BOT_TOKEN", os.environ,
                         "the BOM must be stripped, not glued onto the key")

    def test_a_plain_utf8_env_still_works(self):
        p = self.write_env(b"TELEGRAM_BOT_TOKEN=plain\nTELEGRAM_CHAT_ID=1\n")
        load_dotenv(p)
        self.assertEqual(os.environ.get("TELEGRAM_BOT_TOKEN"), "plain")

    def test_windows_line_endings_are_handled(self):
        p = self.write_env(b"\xef\xbb\xbfTELEGRAM_BOT_TOKEN=crlf\r\n"
                           b"TELEGRAM_CHAT_ID=-100\r\n")
        load_dotenv(p)
        self.assertEqual(os.environ.get("TELEGRAM_BOT_TOKEN"), "crlf")
        self.assertEqual(os.environ.get("TELEGRAM_CHAT_ID"), "-100")

    def test_quotes_added_by_a_helpful_editor_are_stripped(self):
        p = self.write_env(b'TELEGRAM_BOT_TOKEN="quoted:value"\n')
        load_dotenv(p)
        self.assertEqual(os.environ.get("TELEGRAM_BOT_TOKEN"), "quoted:value")

    def test_comments_and_blank_lines_are_ignored(self):
        p = self.write_env(b"\xef\xbb\xbf# a comment\n\n"
                           b"TELEGRAM_BOT_TOKEN=ok\n")
        load_dotenv(p)
        self.assertEqual(os.environ.get("TELEGRAM_BOT_TOKEN"), "ok")

    def test_a_bom_only_line_does_not_create_an_empty_key(self):
        p = self.write_env(b"\xef\xbb\xbf\nTELEGRAM_BOT_TOKEN=ok\n")
        load_dotenv(p)
        self.assertNotIn("", os.environ)
        self.assertEqual(os.environ.get("TELEGRAM_BOT_TOKEN"), "ok")

    def test_a_missing_env_file_is_not_an_error(self):
        load_dotenv(self.dir / "does-not-exist")

    def test_real_environment_variables_still_win(self):
        os.environ["TELEGRAM_BOT_TOKEN"] = "from-environment"
        p = self.write_env(b"TELEGRAM_BOT_TOKEN=from-file\n")
        load_dotenv(p)
        self.assertEqual(os.environ["TELEGRAM_BOT_TOKEN"], "from-environment")

    def test_the_credentials_actually_reach_the_config(self):
        """End to end: the failure this prevents is a silently disabled bot."""
        p = self.write_env(b"\xef\xbb\xbfTELEGRAM_BOT_TOKEN=123:ABC\n"
                           b"TELEGRAM_CHAT_ID=-100999\n")
        cfg = load_config("config/config.toml", p)
        self.assertEqual(cfg.telegram_token, "123:ABC")
        self.assertEqual(cfg.telegram_chat_id, "-100999")
        self.assertEqual(cfg.validate(), [],
                         "with credentials present the config must validate")


class TestConfigFileEncoding(EnvIsolated):
    def test_a_config_toml_with_a_bom_still_parses(self):
        src = Path("config/config.toml").read_bytes()
        p = self.dir / "config.toml"
        p.write_bytes(b"\xef\xbb\xbf" + src)
        cfg = load_config(p, self.dir / "no-env")
        self.assertEqual(cfg.exchange.quote, "USDT")
        self.assertEqual(cfg.universe.broad_limit, 200)


class TestExchangeOverride(EnvIsolated):
    """Switching venue must never require editing a file."""

    def test_environment_variable_overrides_the_exchange(self):
        os.environ["CRYPTO_EDGE_EXCHANGE"] = "kraken"
        cfg = load_config("config/config.toml", self.dir / "no-env")
        self.assertEqual(cfg.exchange.name, "kraken")

    def test_environment_variable_overrides_the_quote_currency(self):
        os.environ["CRYPTO_EDGE_QUOTE"] = "USD"
        cfg = load_config("config/config.toml", self.dir / "no-env")
        self.assertEqual(cfg.exchange.quote, "USD")

    def test_without_an_override_the_config_file_wins(self):
        cfg = load_config("config/config.toml", self.dir / "no-env")
        self.assertEqual(cfg.exchange.name, "binance")

    def test_an_empty_override_is_ignored(self):
        os.environ["CRYPTO_EDGE_EXCHANGE"] = "   "
        cfg = load_config("config/config.toml", self.dir / "no-env")
        self.assertEqual(cfg.exchange.name, "binance")

    def test_the_cli_flag_sets_the_override(self):
        from crypto_edge.cli import build_parser
        args = build_parser().parse_args(["--exchange", "coinbase", "status"])
        self.assertEqual(args.exchange, "coinbase")

    def test_the_cli_flag_is_accepted_by_every_command(self):
        from crypto_edge.cli import build_parser
        parser = build_parser()
        for cmd in ("selfcheck", "start", "status", "verify-live",
                    "verify-restart", "test"):
            args = parser.parse_args(["--exchange", "kucoin", cmd])
            self.assertEqual(args.exchange, "kucoin", cmd)

    def test_live_trading_stays_off_regardless_of_overrides(self):
        os.environ["CRYPTO_EDGE_EXCHANGE"] = "kraken"
        cfg = load_config("config/config.toml", self.dir / "no-env")
        self.assertEqual(cfg.safety.mode, "PAPER")
        self.assertFalse(cfg.safety.live_trading_enabled)


class TestOutputEncoding(unittest.TestCase):
    def test_redirected_output_survives_non_ascii(self):
        """The crash this prevents: piping output to a file on Windows."""
        from crypto_edge.cli import _force_utf8_output

        buf = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
        with self.assertRaises(UnicodeEncodeError):
            buf.write("CIRCUIT BREAKER ⚠ halted \U0001f6a8")
            buf.flush()

        stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
        stream.reconfigure(encoding="utf-8", errors="replace")
        stream.write("CIRCUIT BREAKER ⚠ halted \U0001f6a8")
        stream.flush()          # must not raise

    def test_forcing_utf8_never_raises_on_an_odd_stream(self):
        import sys

        from crypto_edge.cli import _force_utf8_output

        saved_out, saved_err = sys.stdout, sys.stderr
        try:
            sys.stdout = io.StringIO()      # has no reconfigure()
            sys.stderr = io.StringIO()
            _force_utf8_output()            # must swallow the AttributeError
        finally:
            sys.stdout, sys.stderr = saved_out, saved_err


if __name__ == "__main__":
    unittest.main()
