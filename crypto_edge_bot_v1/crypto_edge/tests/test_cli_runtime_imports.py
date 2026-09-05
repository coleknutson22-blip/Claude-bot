"""Every CLI command must be able to RUN, not merely parse.

THE DEFECT THIS FILE EXISTS FOR
-------------------------------
`cmd_scan` called `time.time()` and `crypto_edge/cli.py` never imported `time`.
The command crashed with `NameError: name 'time' is not defined` the first time
a real operator ran it against a live venue -- after the network work was
already done.

Nothing caught it. The suite exercised the strategy, the ranking and the scan
pipeline directly, and the only check on the command itself ran `scan --help`,
which builds an argument parser and returns without executing a single line of
the function body. A missing module-level import is invisible until the line
that uses it actually runs.

So there are two tests here, deliberately overlapping:

  * one EXECUTES the whole command against a fixture feed, which would have
    caught this exact bug;
  * one statically resolves every global name in every module of the package,
    which catches the same CLASS of bug in commands no test happens to drive --
    an operator-only path like `export` or `verify-restart` can hide one just as
    easily, and the next one will not necessarily be in `scan`.
"""
import builtins
import io
import os
import symtable
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import helpers  # noqa: F401  -- silences the engine's log handlers
from crypto_edge import cli

PKG = Path(cli.__file__).resolve().parent


def module_files():
    return sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in str(p))


def undefined_globals(path: Path) -> list[str]:
    """Global names a module uses but never binds, and that are not builtins.

    Uses Python's own scope analysis rather than a hand-rolled AST walk:
    `symtable` already knows what is local, free, or resolved globally in every
    nested scope, including comprehensions and closures.
    """
    src = path.read_text(encoding="utf-8")
    top = symtable.symtable(src, str(path), "exec")
    bound = {s.get_name() for s in top.get_symbols()
             if s.is_assigned() or s.is_imported() or s.is_namespace()}
    # Module dunders exist at runtime but are not in `builtins`.
    builtin_names = set(dir(builtins)) | {
        "__file__", "__name__", "__doc__", "__package__", "__spec__",
        "__loader__", "__builtins__", "__debug__", "__path__"}

    missing: set[str] = set()

    def walk(table):
        for sym in table.get_symbols():
            if sym.is_global() and not sym.is_assigned():
                name = sym.get_name()
                if name not in bound and name not in builtin_names:
                    missing.add(name)
        for child in table.get_children():
            walk(child)

    walk(top)
    return sorted(missing)


class TestEveryModuleResolvesItsGlobals(unittest.TestCase):
    """A missing import is a crash waiting for the right code path."""

    def test_the_cli_module_has_no_undefined_names(self):
        missing = undefined_globals(PKG / "cli.py")
        self.assertEqual(missing, [],
                         f"cli.py uses names it never imports: {missing}")

    def test_no_module_in_the_package_has_undefined_names(self):
        offenders = {}
        for path in module_files():
            missing = undefined_globals(path)
            if missing:
                offenders[str(path.relative_to(PKG))] = missing
        self.assertEqual(offenders, {}, f"undefined names: {offenders}")

    def test_the_checker_actually_detects_a_missing_import(self):
        """A test that cannot fail proves nothing. Prove this one can."""
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.py"
            bad.write_text("def go():\n    return time.time()\n")
            self.assertEqual(undefined_globals(bad), ["time"])

    def test_the_checker_does_not_flag_legitimate_code(self):
        with tempfile.TemporaryDirectory() as d:
            ok = Path(d) / "ok.py"
            ok.write_text(
                "import time\n"
                "X = 1\n"
                "def go(a, b=2):\n"
                "    c = [i for i in range(a) if i > b]\n"
                "    with open('f') as fh:\n"
                "        pass\n"
                "    try:\n"
                "        pass\n"
                "    except ValueError as e:\n"
                "        print(e)\n"
                "    def inner():\n"
                "        return c, X, time\n"
                "    return inner(), len(c), fh\n")
            self.assertEqual(undefined_globals(ok), [])


class TestScanCommandExecutes(unittest.TestCase):
    """Drive `cmd_scan` end to end. --help would not have caught the NameError."""

    def setUp(self):
        from fixtures_fast import frames

        self.dir = Path(tempfile.mkdtemp())
        (self.dir / "data").mkdir()
        (self.dir / "config").mkdir()
        self.symbols = [f"S{i:02d}/USD" for i in range(6)] + ["BTC/USD"]
        # 4h included so the BTC regime RESOLVES: without it every candidate
        # short-circuits on "BTC regime unavailable" and the entry-printing
        # branch of the command never executes -- which is exactly the kind of
        # unrun line that hid the missing import in the first place.
        self.frames = {
            s: frames((i - 3) * 0.00025, seed=400 + i, symbol=s, n_5m=8000,
                      include_4h=True)
            for i, s in enumerate(self.symbols)}

        cfg_src = (PKG.parent / "config" / "config.toml").read_text(
            encoding="utf-8-sig")
        cfg_src = cfg_src.replace('broad_source = "coingecko"',
                                  'broad_source = "static"')
        self.cfg_path = self.dir / "config" / "config.toml"
        self.cfg_path.write_text(cfg_src)
        self._cwd = os.getcwd()
        os.chdir(self.dir)
        self.addCleanup(lambda: os.chdir(self._cwd))

    def make_args(self):
        return cli.build_parser().parse_args(
            ["--config", str(self.cfg_path), "--env", str(self.dir / "no-env"),
             "scan"])

    def fake_feed(self):
        from crypto_edge.models import MarketMeta

        outer = self

        class Feed:
            name, quote = "fixture", "USD"

            def load_markets(self):
                return {s: MarketMeta(s, s.split("/")[0], "USD", True,
                                      4, 4, 0.0001, 10.0)
                        for s in outer.symbols}

            def fetch_tickers(self):
                # 2 bps: inside the 15 bps universe limit. A wide fixture
                # spread would be filtered out before the scan ever ran.
                return {s: {"quoteVolume": 5e7, "bid": 99.99, "ask": 100.01,
                            "last": 100.0} for s in outer.symbols}

            def fetch_ohlcv(self, symbol, timeframe, limit):
                fr = outer.frames.get(symbol)
                if fr is None or timeframe not in fr:
                    from crypto_edge.data.feed import DataUnavailable
                    raise DataUnavailable(f"no {timeframe} for {symbol}")
                return fr[timeframe]

            def fetch_quote(self, symbol):
                return None

            def server_time_ms(self):
                return None

        return Feed()

    def run_scan(self):
        from unittest import mock

        from crypto_edge.config import load_config
        from crypto_edge.storage import db
        from crypto_edge.storage.repo import Repo

        args = self.make_args()
        cfg = load_config(args.config, args.env)
        cfg.telegram.enabled = False
        # The fixtures are USD pairs, so the regime reference has to be
        # BTC/USD. Left as BTC/USDT it simply is not in the data and every
        # candidate short-circuits on "BTC regime unavailable".
        cfg.exchange.quote = "USD"
        cfg.resolve_symbols()
        cfg.universe.broad_source = "static"
        cfg.universe.broad_static_assets = [s.split("/")[0]
                                            for s in self.symbols]
        cfg.universe.broad_min_assets = 1      # a 7-asset fixture, not a market
        cfg.universe.broad_limit = 50
        cfg.universe.min_market_age_days = 0
        cfg.universe.min_candles_1h = 60
        cfg.strategy.warmup_bars = 60
        cfg.aggressive.shortlist_size = 5

        conn = db.connect(cfg.engine.db_path)
        db.init_db(conn)
        repo = Repo(conn)
        feed = self.fake_feed()
        notifier = mock.MagicMock()

        with mock.patch.object(cli, "_bootstrap",
                               return_value=(cfg, repo, feed, notifier)):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli.cmd_scan(args)
        return rc, buf.getvalue(), repo

    def test_the_command_runs_to_completion(self):
        rc, out, _ = self.run_scan()
        self.assertEqual(rc, 0, out)
        self.assertIn("OPPORTUNITY SCAN", out)

    def test_it_reports_the_timing_line_that_needed_the_missing_import(self):
        _, out, _ = self.run_scan()
        self.assertIn("scan took", out,
                      "the line whose time.time() call was the crash")
        self.assertIn("deep fetches", out)

    def test_it_prints_a_row_per_shortlisted_symbol(self):
        _, out, _ = self.run_scan()
        self.assertIn("SETUP", out)
        rows = [ln for ln in out.splitlines()
                if any(s.split("/")[0] in ln for s in self.symbols)
                and "SYMBOL" not in ln]
        self.assertGreater(len(rows), 0, out)

    def test_the_entry_branch_of_the_printer_is_exercised(self):
        """Coverage guard: a line that never runs cannot fail a test.

        The missing import survived because nothing executed the line that
        used it. If this fixture ever stops producing a tradable setup, the
        ENTRY formatting path goes unrun again and the same class of bug can
        hide there -- so the absence of entries is itself a failure.
        """
        _, out, _ = self.run_scan()
        self.assertIn("ENTRY", out,
                      "the fixture must produce at least one tradable setup")
        self.assertNotIn("0 tradable setup(s)", out)

    def test_both_the_veto_and_the_entry_paths_are_reachable(self):
        _, out, _ = self.run_scan()
        self.assertIn("vetoed", out, "a regime veto row must render too")

    def test_it_says_plainly_that_nothing_was_traded(self):
        _, out, _ = self.run_scan()
        self.assertIn("nothing was traded", out)

    def test_it_records_observations_for_research(self):
        _, _, repo = self.run_scan()
        obs = repo.get_observations()
        self.assertGreater(len(obs), 0, "every candidate must be journalled")
        for row in obs:
            self.assertEqual(row["strategy"], "aggressive_momentum_v2")
            self.assertIn(row["side"], ("long", "short", "none"))

    def test_it_opens_no_position_and_spends_no_cash(self):
        _, _, repo = self.run_scan()
        self.assertEqual(repo.get_positions(), [],
                         "scan is signal generation only")


class TestEveryCommandIsReachable(unittest.TestCase):
    """The parser and the functions must not drift apart."""

    def test_every_subcommand_has_a_callable_handler(self):
        parser = cli.build_parser()
        actions = [a for a in parser._actions
                   if hasattr(a, "choices") and isinstance(a.choices, dict)]
        self.assertTrue(actions, "no subparsers found")
        for sub in actions:
            for name, p in sub.choices.items():
                fn = p.get_default("func")
                self.assertTrue(callable(fn), f"{name} has no handler")

    def test_scan_is_registered(self):
        parser = cli.build_parser()
        subs = [a for a in parser._actions
                if hasattr(a, "choices") and isinstance(a.choices, dict)][0]
        self.assertIn("scan", subs.choices)
        self.assertIs(subs.choices["scan"].get_default("func"), cli.cmd_scan)


if __name__ == "__main__":
    unittest.main()
