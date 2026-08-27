"""Quote currency is configuration, not an assumption baked into strings.

WHAT WAS REPORTED
-----------------
Kraken's USDT books are thin: 0 of 29 broad-universe intersections cleared the
$5M 24h threshold, with BTC/USDT at ~$3.94M. Kraken's depth is in USD. But the
system could not be pointed at kraken/USD, because the quote currency was
configuration in name only -- three places hardcoded a USDT pair.

THE ONE THAT MATTERED
---------------------
`strategy.btc_symbol = "BTC/USDT"`. On a USD venue that market may not even be
listed, so the BTC reference series would come back empty, `require_btc_data`
would fire, and the market-regime filter would be silently disabled -- or worse,
the regime would be read from a thin USDT market while every position traded in
USD. `universe.always_include` had the same shape.

THE RULE ENFORCED HERE
----------------------
Pairs are never written down. Bases are configured; `Config.market_for()`
completes them from `Config.quote_currency`, once, after the TOML and any
override have settled. Anything an operator writes explicitly is honoured and
then VALIDATED against the quote, so a mismatch is a startup error rather than a
silently different market.
"""
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import helpers  # noqa: F401  -- silences the engine's log handlers
from crypto_edge.cli import build_parser
from crypto_edge.config import Config, load_config
from crypto_edge.data.universe import UniverseBuilder
from crypto_edge.models import MarketMeta
from crypto_edge.notify import formatters as fmt
from crypto_edge.notify.telegram import TelegramNotifier
from crypto_edge.timeutils import now_ms
from helpers import open_repo, temp_repo

CONFIG = "config/config.toml"
NO_ENV = "/nonexistent-env-file"
KEYS = ("CRYPTO_EDGE_EXCHANGE", "CRYPTO_EDGE_QUOTE")


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = fn(*a, **kw)
    return result, buf.getvalue()


class Recording:
    def __init__(self):
        self.sent = []

    def send(self, token, chat_id, text, timeout):
        self.sent.append(text)
        return True, "ok"


class QuoteIsolated(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in KEYS}

    def tearDown(self):
        for k in KEYS:
            os.environ.pop(k, None)
            if self._saved.get(k) is not None:
                os.environ[k] = self._saved[k]

    def load(self, *argv):
        """Exactly the CLI path: main() exports, then the config resolves."""
        args = build_parser().parse_args(list(argv))
        if getattr(args, "exchange", None):
            os.environ["CRYPTO_EDGE_EXCHANGE"] = args.exchange
        if getattr(args, "quote", None):
            os.environ["CRYPTO_EDGE_QUOTE"] = args.quote
        return load_config(CONFIG, NO_ENV)


class TestBtcReferenceFollowsQuote(QuoteIsolated):
    def test_kraken_usd_uses_btc_usd(self):
        cfg = self.load("--exchange", "kraken", "--quote", "USD", "diagnose")
        self.assertEqual(cfg.strategy.btc_symbol, "BTC/USD")

    def test_no_usdt_symbol_survives_anywhere_on_a_usd_venue(self):
        """The blunt version of the requirement: grep the resolved config."""
        cfg = self.load("--exchange", "kraken", "--quote", "USD", "diagnose")
        surfaces = [cfg.strategy.btc_symbol, *cfg.universe.always_include,
                    cfg.exchange_label(), cfg.market_for("SOL")]
        for value in surfaces:
            self.assertNotIn("USDT", value,
                             f"{value!r} still carries a USDT assumption")

    def test_always_include_follows_the_quote(self):
        cfg = self.load("--exchange", "kraken", "--quote", "USD", "diagnose")
        self.assertEqual(cfg.universe.always_include, ["BTC/USD", "ETH/USD"])

    def test_the_default_venue_is_unchanged(self):
        cfg = self.load("status")
        self.assertEqual(cfg.strategy.btc_symbol, "BTC/USDT")
        self.assertEqual(cfg.universe.always_include, ["BTC/USDT", "ETH/USDT"])

    def test_market_for_is_the_single_construction_point(self):
        cfg = self.load("--quote", "EUR", "status")
        self.assertEqual(cfg.market_for("SOL"), "SOL/EUR")
        self.assertEqual(cfg.market_for("sol"), "SOL/EUR", "case-insensitive")
        self.assertEqual(cfg.quote_currency, "EUR")

    def test_a_directly_constructed_config_resolves_too(self):
        """Tests and the smoke test build Config() directly; it must be coherent."""
        cfg = Config()
        self.assertEqual(cfg.strategy.btc_symbol, "BTC/USDT")
        cfg.exchange.quote = "USD"
        cfg.resolve_symbols()
        self.assertEqual(cfg.strategy.btc_symbol, "BTC/USD")

    def test_changing_the_btc_base_is_respected(self):
        cfg = Config()
        cfg.strategy.btc_base = "ETH"
        cfg.resolve_symbols()
        self.assertEqual(cfg.strategy.btc_symbol, "ETH/USDT")


class TestExplicitOverridesAreValidated(QuoteIsolated):
    """An operator may pin a symbol -- but not to a different currency."""

    def _config_with(self, table: str, line: str) -> Path:
        """Set one key inside an EXISTING table.

        Appending a second `[strategy]` header would be a TOML error, not an
        operator override -- the key has to land in the table that is already
        there, exactly as a human editing the file would write it.
        """
        workdir = Path(tempfile.mkdtemp())
        out, seen = [], False
        for row in Path(CONFIG).read_text(encoding="utf-8-sig").splitlines():
            out.append(row)
            if row.strip() == f"[{table}]":
                out.append(line)
                seen = True
        self.assertTrue(seen, f"[{table}] not found in the shipped config")
        path = workdir / "c.toml"
        path.write_text("\n".join(out) + "\n")
        return path

    def test_an_explicit_btc_symbol_is_honoured(self):
        path = self._config_with("strategy", 'btc_symbol = "BTC/USD"')
        os.environ["CRYPTO_EDGE_QUOTE"] = "USD"
        cfg = load_config(path, NO_ENV)
        self.assertEqual(cfg.strategy.btc_symbol, "BTC/USD")

    def test_a_mismatched_explicit_symbol_is_a_startup_error(self):
        """The silent-disable failure, turned into a loud one."""
        path = self._config_with("strategy", 'btc_symbol = "BTC/USDT"')
        os.environ["CRYPTO_EDGE_QUOTE"] = "USD"
        cfg = load_config(path, NO_ENV)
        cfg.telegram.enabled = False
        errs = cfg.validate()
        self.assertTrue(any("btc_symbol" in e for e in errs), errs)
        self.assertTrue(any("USD" in e for e in errs))

    def test_a_mismatched_always_include_is_a_startup_error(self):
        path = self._config_with("universe", 'always_include = ["BTC/USDT"]')
        os.environ["CRYPTO_EDGE_QUOTE"] = "USD"
        cfg = load_config(path, NO_ENV)
        cfg.telegram.enabled = False
        self.assertTrue(any("always_include" in e for e in cfg.validate()))

    def test_a_pair_in_the_quote_field_is_rejected(self):
        cfg = Config()
        cfg.exchange.quote = "BTC/USD"
        cfg.telegram.enabled = False
        self.assertTrue(any("bare currency code" in e for e in cfg.validate()))


class TestVolumeIsEvaluatedInTheSelectedQuote(unittest.TestCase):
    def setUp(self):
        cfg = Config()
        cfg.exchange.quote = "USD"
        cfg.resolve_symbols()
        self.cfg = cfg
        self.b = UniverseBuilder(cfg.universe)

    def _market(self, symbol):
        return MarketMeta(symbol, symbol.split("/")[0], "USD", True,
                          4, 4, 0.0001, 10.0)

    def test_the_threshold_is_applied_to_quote_currency_notional(self):
        markets = {"SOL/USD": self._market("SOL/USD")}
        tickers = {"SOL/USD": {"baseVolume": 100_000.0, "last": 100.0}}   # 10M USD
        keep, _ = self.b.build_candidates(markets, tickers)
        self.assertEqual(keep, ["SOL/USD"])

    def test_the_rejection_message_names_the_quote_currency(self):
        markets = {"SOL/USD": self._market("SOL/USD")}
        tickers = {"SOL/USD": {"baseVolume": 10.0, "last": 2.0}}
        _, audit = self.b.build_candidates(markets, tickers)
        reason = audit[0]["reject_reason"]
        self.assertIn("USD", reason)
        self.assertNotIn("$", reason,
                         "a currency symbol would be wrong on a non-USD venue")

    def test_a_eur_venue_reports_eur(self):
        cfg = Config()
        cfg.exchange.quote = "EUR"
        cfg.resolve_symbols()
        b = UniverseBuilder(cfg.universe)
        markets = {"SOL/EUR": MarketMeta("SOL/EUR", "SOL", "EUR", True,
                                         4, 4, 0.0001, 10.0)}
        _, audit = b.build_candidates(markets, {"SOL/EUR": {"baseVolume": 1.0,
                                                            "last": 1.0}})
        self.assertIn("EUR", audit[0]["reject_reason"])

    def test_the_threshold_value_itself_is_unchanged(self):
        from crypto_edge.config import load_config as lc
        self.assertEqual(lc(CONFIG, NO_ENV).universe.min_dollar_volume_24h,
                         5_000_000.0)


class TestEverySurfaceShowsTheSameVenue(QuoteIsolated):
    def test_status_selfcheck_and_telegram_agree(self):
        cfg = self.load("--exchange", "kraken", "--quote", "USD", "status")
        cfg.telegram.enabled = True
        cfg.telegram_token, cfg.telegram_chat_id = "tok", "chat"
        label = cfg.exchange_label()
        self.assertEqual(label, "kraken/USD")

        # selfcheck
        from crypto_edge.selfcheck import run_selfcheck
        repo, _ = temp_repo()
        notifier = TelegramNotifier("tok", "chat", repo, enabled=True,
                                    transport=Recording(), sleep=lambda _: None,
                                    outbox_lease_s=0)
        rep = run_selfcheck(cfg, repo, None, notifier, check_network=False)
        line = [r for r in rep.results if r.name == "effective exchange"][0]
        self.assertIn(label, line.detail)

        # telegram
        from crypto_edge.verify_live import VerifyReport, verify_telegram
        transport = Recording()
        n2 = TelegramNotifier("tok", "chat", repo, enabled=True,
                              transport=transport, sleep=lambda _: None,
                              outbox_lease_s=0)
        quiet(verify_telegram, cfg, repo, n2, VerifyReport())
        online = [m for m in transport.sent if "BOT ONLINE" in m][0]
        self.assertIn(label, online)
        self.assertNotIn("USDT", online)

    def test_the_engine_announcement_uses_the_same_label(self):
        from helpers import breakout_closes, trend_closes
        from test_engine import build_engine
        engine, feed, transport, _ = build_engine(
            {"BTC/USDT": trend_closes(400, seed=4),
             "SOL/USDT": breakout_closes(400, seed=11)})
        engine.cfg.exchange.name = "kraken"
        engine.cfg.exchange.quote = "USD"
        engine.announce_start()
        online = [m for m in transport.sent if "BOT ONLINE" in m][0]
        self.assertIn("kraken/USD", online)

    def test_status_prints_the_label_and_source(self):
        from crypto_edge.cli import cmd_status
        workdir = Path(tempfile.mkdtemp())
        text = Path(CONFIG).read_text(encoding="utf-8-sig")
        text = text.replace('db_path = "data/crypto_edge.db"',
                            f'db_path = "{(workdir / "s.db").as_posix()}"')
        text = text.replace('log_dir = "logs"',
                            f'log_dir = "{(workdir / "logs").as_posix()}"')
        path = workdir / "c.toml"
        path.write_text(text)
        args = build_parser().parse_args(
            ["--config", str(path), "--env", NO_ENV,
             "--exchange", "kraken", "--quote", "USD", "status"])
        os.environ["CRYPTO_EDGE_EXCHANGE"] = "kraken"
        os.environ["CRYPTO_EDGE_QUOTE"] = "USD"
        _, out = quiet(cmd_status, args)
        self.assertIn("kraken/USD", out)
        self.assertNotIn("USDT", out)


class TestQuoteChangeIsVisibleInProvenance(unittest.TestCase):
    """USD and USDT account state must never be silently mixed."""

    def setUp(self):
        self.repo, self.path = temp_repo()

    def test_a_quote_change_is_recorded_as_a_change(self):
        self.repo.record_exchange("kraken/USDT")
        changed, previous = self.repo.record_exchange("kraken/USD")
        self.assertTrue(changed)
        self.assertEqual(previous, "kraken/USDT")

    def test_the_selfcheck_names_the_quote_currency_change(self):
        from crypto_edge.selfcheck import run_selfcheck
        self.repo.record_exchange("kraken/USDT")
        cfg = Config()
        cfg.exchange.name, cfg.exchange.quote = "kraken", "USD"
        cfg.resolve_symbols()
        cfg.telegram.enabled = False
        rep = run_selfcheck(cfg, self.repo, None,
                            TelegramNotifier("", "", self.repo, enabled=False,
                                             transport=Recording()),
                            check_network=False)
        warn = [r for r in rep.results
                if r.name == "exchange unchanged since last run"][0]
        self.assertFalse(warn.ok)
        self.assertIn("QUOTE CURRENCY", warn.detail)
        self.assertIn("USDT", warn.detail)
        self.assertIn("denominated", warn.detail,
                      "the operator must be told the equity units differ")

    def test_an_exchange_change_is_worded_differently_from_a_quote_change(self):
        from crypto_edge.selfcheck import run_selfcheck

        def warn_for(previous, name, quote):
            repo, _ = temp_repo()
            repo.record_exchange(previous)
            cfg = Config()
            cfg.exchange.name, cfg.exchange.quote = name, quote
            cfg.resolve_symbols()
            cfg.telegram.enabled = False
            rep = run_selfcheck(cfg, repo, None,
                                TelegramNotifier("", "", repo, enabled=False,
                                                 transport=Recording()),
                                check_network=False)
            return [r for r in rep.results
                    if r.name == "exchange unchanged since last run"][0].detail

        self.assertIn("QUOTE CURRENCY", warn_for("kraken/USDT", "kraken", "USD"))
        self.assertIn("EXCHANGE", warn_for("binance/USDT", "kraken", "USDT"))

    def test_the_same_venue_and_quote_is_not_a_change(self):
        self.repo.record_exchange("kraken/USD")
        changed, _ = self.repo.record_exchange("kraken/USD")
        self.assertFalse(changed)

    def test_the_record_survives_a_restart(self):
        self.repo.record_exchange("kraken/USD")
        self.repo.conn.close()
        repo2 = open_repo(self.path)
        changed, previous = repo2.record_exchange("kraken/USDT")
        self.assertTrue(changed)
        self.assertEqual(previous, "kraken/USD")


class TestIndependentMarketIdentities(QuoteIsolated):
    """A USD run and a USDT run must be reproducibly distinct."""

    def test_the_same_asset_yields_different_market_symbols(self):
        usd = self.load("--exchange", "kraken", "--quote", "USD", "diagnose")
        self.assertEqual(usd.market_for("SOL"), "SOL/USD")
        for k in KEYS:
            os.environ.pop(k, None)
        usdt = self.load("--exchange", "kraken", "--quote", "USDT", "diagnose")
        self.assertEqual(usdt.market_for("SOL"), "SOL/USDT")
        self.assertNotEqual(usd.market_for("SOL"), usdt.market_for("SOL"))

    def test_market_age_is_keyed_per_market_not_per_asset(self):
        """SOL/USD and SOL/USDT are different markets with different listings."""
        repo, _ = temp_repo()
        repo.record_market_age("SOL/USD", now_ms() - 900 * 86_400_000,
                               "coarse_ohlcv", now_ms())
        repo.record_market_age("SOL/USDT", now_ms() - 30 * 86_400_000,
                               "coarse_ohlcv", now_ms())
        self.assertNotEqual(repo.get_market_age("SOL/USD")["first_ms"],
                            repo.get_market_age("SOL/USDT")["first_ms"])

    def test_the_venue_label_distinguishes_the_two_runs(self):
        usd = self.load("--exchange", "kraken", "--quote", "USD", "diagnose")
        self.assertEqual(usd.exchange_label(), "kraken/USD")
        for k in KEYS:
            os.environ.pop(k, None)
        usdt = self.load("--exchange", "kraken", "--quote", "USDT", "diagnose")
        self.assertEqual(usdt.exchange_label(), "kraken/USDT")

    def test_both_configurations_validate(self):
        for quote in ("USD", "USDT", "EUR"):
            for k in KEYS:
                os.environ.pop(k, None)
            cfg = self.load("--exchange", "kraken", "--quote", quote, "diagnose")
            cfg.telegram.enabled = False
            self.assertEqual(cfg.validate(), [], f"quote={quote}")


class TestNoThresholdMoved(unittest.TestCase):
    def test_every_gate_is_at_its_documented_value(self):
        cfg = load_config(CONFIG, NO_ENV)
        self.assertEqual(cfg.universe.min_dollar_volume_24h, 5_000_000.0)
        self.assertEqual(cfg.universe.min_atr_pct, 0.8)
        self.assertEqual(cfg.universe.max_atr_pct, 25.0)
        self.assertEqual(cfg.universe.min_candles_1h, 400)
        self.assertEqual(cfg.universe.min_market_age_days, 45)
        self.assertEqual(cfg.universe.max_spread_bps, 15.0)
        self.assertEqual(cfg.risk.risk_per_trade_pct, 0.5)
        self.assertEqual(cfg.risk.max_position_pct, 15.0)
        self.assertEqual(cfg.strategy.min_score, 55.0)
        self.assertEqual(cfg.strategy.min_adx, 20.0)


if __name__ == "__main__":
    unittest.main()
