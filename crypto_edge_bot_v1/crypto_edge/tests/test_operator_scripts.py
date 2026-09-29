"""The two discovery scripts an operator runs before any analysis.

WHAT THESE ARE DEFENDING
------------------------
These scripts exist because a report run against the wrong file is worse than
no report: it looks like a result. Three properties have to hold or they make
that failure more likely rather than less.

  1. THEY MUST NOT WRITE. `db.connect` + `init_db` CREATE and MIGRATE a
     database. A discovery tool that did either would modify the very ledger
     the operator is trying to identify, before they had decided anything.
     Both open every candidate with `immutable=1`.

  2. THEY MUST RUN WITHOUT THE BOT'S DEPENDENCIES. An operator whose numpy
     install is broken still needs to find their ledger. `find_ledger` is
     stdlib-only, so it cannot import the package it is looking for.

  3. THEY MUST RESOLVE PATHS, NOT ECHO THEM. `db_path` and `--config` are both
     relative by default, so the answer to "which file" depends on the working
     directory, which is exactly what the operator is unsure about.
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

import helpers  # noqa: F401  -- silences the engine's log handlers
from crypto_edge.models import ClosedTrade
from crypto_edge.storage import db
from crypto_edge.storage.repo import Repo
from helpers import temp_repo

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
B = "aggressive_momentum_v2"


def run(script, *args, cwd=None, env=None):
    e = dict(os.environ)
    e.setdefault("PYTHONPATH", str(REPO_ROOT))
    e.update(env or {})
    return subprocess.run([sys.executable, str(SCRIPTS / script), *args],
                          cwd=cwd, env=e, capture_output=True, text=True)


def seed(path, n=3, strategy=B):
    conn = db.connect(path)
    db.init_db(conn)
    repo = Repo(conn)
    for i in range(n):
        repo.add_trade(ClosedTrade(
            id=f"{strategy}-{i}", position_id=f"p{i}", symbol="X/USD",
            strategy=strategy, strategy_version="v2", side="long", qty=1.0,
            entry_ref_price=100.0, entry_fill_price=100.0,
            entry_ms=1_700_000_000_000, exit_ref_price=101.0,
            exit_fill_price=101.0, exit_ms=1_700_000_000_000 + i * 3_600_000,
            exit_reason="target", initial_stop=98.0, final_stop=98.0,
            gross_pnl=1.0, fees=0.15, slippage_cost=0.12, net_pnl=0.73,
            financing=0.0, return_pct=0.0, account_return_pct=0.0, mfe=1.0,
            mae=0.0, duration_s=60.0, equity_after=10_000.0, journal={}))
    conn.commit()
    conn.close()


class TestFindLedger(unittest.TestCase):
    def setUp(self):
        self.repo, self.path = temp_repo()
        self.root = Path(self.path).parent

    def test_it_finds_a_ledger_and_counts_its_trades(self):
        seed(self.path, n=5)
        r = run("find_ledger.py", str(self.root))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(str(Path(self.path).resolve()), r.stdout)
        self.assertIn(B, r.stdout)
        self.assertIn("5 trades", r.stdout)

    def test_it_reports_each_strategy_separately(self):
        seed(self.path, n=5, strategy=B)
        seed(self.path, n=2, strategy="trend_breakout")
        out = run("find_ledger.py", str(self.root)).stdout
        self.assertIn("5 trades", out)
        self.assertIn("2 trades", out)
        self.assertIn("7 closed trade(s) in total", out)

    def test_it_does_not_create_or_modify_anything(self):
        # THE property. A discovery tool that migrated the ledger would change
        # the thing being identified before the operator chose it.
        seed(self.path, n=3)
        before = Path(self.path).read_bytes()
        run("find_ledger.py", str(self.root))
        self.assertEqual(Path(self.path).read_bytes(), before)

    def test_it_creates_no_file_in_an_empty_tree(self):
        empty = self.root / "nothing"
        empty.mkdir()
        run("find_ledger.py", str(empty))
        self.assertEqual(list(empty.iterdir()), [])

    def test_it_exits_non_zero_when_nothing_is_found(self):
        empty = self.root / "bare"
        empty.mkdir()
        r = run("find_ledger.py", str(empty))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("No Crypto Edge ledger found", r.stdout)

    def test_an_unrelated_database_is_skipped_without_aborting_the_scan(self):
        import sqlite3
        other = self.root / "unrelated.db"
        c = sqlite3.connect(other)
        c.execute("CREATE TABLE things(x INTEGER)")
        c.commit()
        c.close()
        seed(self.path, n=3)
        r = run("find_ledger.py", str(self.root))
        self.assertNotIn("unrelated.db", r.stdout)
        # And the real one alongside it is still reported: an unreadable
        # candidate must not take the whole scan down with it.
        self.assertIn(str(Path(self.path).resolve()), r.stdout)
        self.assertIn("3 trades", r.stdout)

    def test_open_readonly_returns_a_connection_that_cannot_write(self):
        # The property both scripts depend on, asserted directly rather than
        # only through their output: these tools must be incapable of
        # modifying a ledger an operator is still deciding about.
        import importlib.util
        import sqlite3
        spec = importlib.util.spec_from_file_location(
            "find_ledger", SCRIPTS / "find_ledger.py")
        mod = importlib.util.module_from_spec(spec)
        # Importable without scanning anything: the script guards its main
        # body, so this cannot touch the filesystem or `sys.argv`.
        spec.loader.exec_module(mod)
        seed(self.path, n=1)
        con, mode = mod.open_readonly(self.path)
        with self.assertRaises(sqlite3.OperationalError):
            con.execute("INSERT INTO meta(key, value) VALUES('x','y')")
            con.commit()
        con.close()

    def test_it_sees_rows_still_in_a_live_write_ahead_log(self):
        # THE false negative these scripts exist to prevent. `immutable=1`
        # skips the -wal, so a bot that is running right now would be reported
        # as having zero trades -- which reads exactly like a wrong path.
        seed(self.path, n=4)
        live = db.connect(self.path)          # holds the WAL open, as the bot does
        try:
            out = run("find_ledger.py", str(self.root)).stdout
            self.assertIn("4 trades", out)
            self.assertNotIn("LIVE write-ahead log was NOT read", out)
        finally:
            live.close()

    def test_it_needs_none_of_the_bot_dependencies(self):
        # Stdlib only: an operator with a broken numpy still has to be able to
        # find their ledger. Asserted by running with the package unreachable.
        seed(self.path, n=1)
        r = run("find_ledger.py", str(self.root), env={"PYTHONPATH": ""})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(B, r.stdout)


class TestShowConfig(unittest.TestCase):
    def setUp(self):
        self.repo, self.path = temp_repo()
        self.home = Path(self.path).parent
        (self.home / "config").mkdir(exist_ok=True)

    def write_config(self, body):
        (self.home / "config" / "config.toml").write_text(body)

    def test_it_resolves_a_relative_db_path_against_the_working_directory(self):
        self.write_config('[engine]\ndb_path = "data/crypto_edge.db"\n')
        out = run("show_config.py", cwd=self.home).stdout
        self.assertIn(str(self.home / "data" / "crypto_edge.db"), out)

    def test_it_warns_when_the_configured_database_is_absent(self):
        self.write_config('[engine]\ndb_path = "data/missing.db"\n')
        out = run("show_config.py", cwd=self.home).stdout
        self.assertIn("DOES NOT EXIST", out)
        self.assertIn("CREATE an empty one", out)
        self.assertFalse((self.home / "data" / "missing.db").exists())

    def test_it_counts_the_trades_in_a_database_that_is_present(self):
        (self.home / "data").mkdir(exist_ok=True)
        seed(str(self.home / "data" / "crypto_edge.db"), n=4)
        self.write_config('[engine]\ndb_path = "data/crypto_edge.db"\n')
        out = run("show_config.py", cwd=self.home).stdout
        self.assertIn(B, out)
        self.assertIn("4", out)

    def test_it_reports_the_effective_fee_not_the_raw_field(self):
        self.write_config('[execution]\ntaker_fee_bps = 7.5\n'
                          'fee_tier = "tier3"\n')
        out = run("show_config.py", cwd=self.home).stdout
        self.assertIn("EFFECTIVE taker fee : 38.0 bps per side", out)
        self.assertIn("tier3", out)

    def test_it_flags_a_custom_fee_that_matches_no_kraken_tier(self):
        self.write_config('[execution]\ntaker_fee_bps = 7.5\n')
        out = run("show_config.py", cwd=self.home).stdout
        self.assertIn("no Kraken tier is being simulated", out)
        self.assertIn("matches NO tier", out)

    def test_a_real_tier_is_not_flagged_as_matching_nothing(self):
        self.write_config('[execution]\nfee_tier = "tier1"\n')
        out = run("show_config.py", cwd=self.home).stdout
        self.assertNotIn("matches NO tier", out)
        self.assertIn("80.0 bps per side", out)

    def test_it_shows_where_the_venue_came_from(self):
        self.write_config('[exchange]\nname = "binance"\nquote = "USDT"\n')
        plain = run("show_config.py", cwd=self.home).stdout
        self.assertIn("binance/USDT", plain)
        self.assertIn("config file", plain)
        over = run("show_config.py", cwd=self.home,
                   env={"CRYPTO_EDGE_EXCHANGE": "kraken",
                        "CRYPTO_EDGE_QUOTE": "USD"}).stdout
        self.assertIn("kraken/USD", over)
        self.assertIn("override", over)

    def test_it_does_not_open_or_create_the_database(self):
        (self.home / "data").mkdir(exist_ok=True)
        target = self.home / "data" / "crypto_edge.db"
        seed(str(target), n=2)
        before = target.read_bytes()
        self.write_config('[engine]\ndb_path = "data/crypto_edge.db"\n')
        run("show_config.py", cwd=self.home)
        self.assertEqual(target.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
