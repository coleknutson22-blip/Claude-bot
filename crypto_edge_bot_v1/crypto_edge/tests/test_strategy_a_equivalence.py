"""Strategy A must be BEHAVIOURALLY IDENTICAL after the multi-strategy refactor.

WHY THIS FILE GATES THE STAGE
-----------------------------
Stage 1 changes how money is counted. The old model was spot-long-only --
`equity = cash + SUM(qty * mark)`, cash debited by notional on entry and
credited by notional on exit -- and it cannot express a short. The new model
reserves collateral per position:

    equity = cash + SUM(margin_held + unrealized)

For a 1x long these are the same number, because `margin_held` is the entry
notional and `unrealized` is `(mark - entry) * qty`:

    margin_held + unrealized  ==  qty*entry + (mark - entry)*qty  ==  qty*mark

That identity is the entire safety argument for touching a live paper account
with real history, so it is asserted here rather than reasoned about in a
comment. If any test in this file fails, the refactor is wrong and Strategy A's
recorded results have moved -- which is not allowed.

It also proves the v3 -> v4 migration preserves the account row exactly, by
diffing against `account_pre_v4`, the pre-migration table the migration keeps
on purpose.
"""
import sqlite3
import tempfile
import unittest
from pathlib import Path

import helpers  # noqa: F401  -- silences the engine's log handlers
from crypto_edge.execution.paper_broker import PaperBroker, realise_pnl
from crypto_edge.models import Fill, MarketMeta, Position
from crypto_edge.portfolio.account import PaperAccount
from crypto_edge.storage import db
from crypto_edge.storage.repo import Repo
from crypto_edge.timeutils import now_ms, utc_date
from helpers import STRATEGY, temp_repo

META = MarketMeta("SOL/USDT", "SOL", "USDT", True, 4, 4, 0.0001, 10.0)


def long_position(qty=10.0, entry=100.0, **kw):
    base = dict(
        id="p1", symbol="SOL/USDT", strategy=STRATEGY, strategy_version="1.0.0",
        side="long", qty=qty, entry_ref_price=entry, entry_fill_price=entry,
        entry_ms=now_ms(), entry_fee=1.0, entry_slippage=0.5,
        initial_stop=entry * 0.95, current_stop=entry * 0.95,
        highest_price=entry, lowest_price=entry, risk_amount=50.0,
        candle_id="SOL/USDT|1h|1", journal={})
    base.update(kw)
    return Position(**base)


class TestTheCollateralIdentity(unittest.TestCase):
    """cash + SUM(margin + unrealized) == cash + SUM(qty * mark), for longs."""

    def test_a_1x_long_contributes_exactly_its_mark_value(self):
        p = long_position()
        for mark in (50.0, 99.9, 100.0, 100.1, 250.0, 1e6):
            self.assertAlmostEqual(p.collateral_value(mark), p.qty * mark, places=9,
                                   msg=f"mark={mark}")

    def test_the_identity_holds_for_awkward_quantities(self):
        for qty, entry in ((0.00031234, 61234.5678), (1e-8, 1.0),
                           (123456.789, 0.00004321), (7.0, 3.0)):
            p = long_position(qty=qty, entry=entry)
            self.assertAlmostEqual(p.collateral_value(entry * 1.37),
                                   qty * entry * 1.37, places=6)

    def test_margin_defaults_to_the_entry_notional(self):
        """A position built without margin must not contribute zero capital."""
        p = long_position(qty=3.0, entry=250.0)
        self.assertAlmostEqual(p.margin_held, 750.0)

    def test_unrealized_is_unchanged_for_a_long(self):
        p = long_position()
        self.assertAlmostEqual(p.unrealized(110.0), 100.0)
        self.assertAlmostEqual(p.unrealized(90.0), -100.0)

    def test_direction_of_a_long_is_positive(self):
        self.assertEqual(long_position().direction, 1)
        self.assertFalse(long_position().is_short)


class TestRoundTripArithmeticIsUnchanged(unittest.TestCase):
    """The cash a closed long returns must be what it always was."""

    def setUp(self):
        self.repo, self.path = temp_repo()
        self.broker = PaperBroker(taker_fee_bps=7.5, slippage_bps=6.0,
                                  stop_slippage_bps=15.0)
        self.acct = PaperAccount(self.repo, self.broker, 10_000.0, STRATEGY)

    def _open(self, qty=10.0, ref=100.0):
        fill = self.broker.buy("SOL/USDT", qty, ref, meta=META)
        return self.acct.open_position(
            symbol="SOL/USDT", strategy=STRATEGY, strategy_version="1.0.0",
            qty=qty, ref_price=ref, fill=fill, initial_stop=ref * 0.95,
            risk_amount=50.0, candle_id="SOL/USDT|1h|1", signal_score=70.0,
            journal={}), fill

    def test_entry_debits_notional_plus_fee_exactly_as_before(self):
        pos, fill = self._open()
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(self.acct.cash(), 10_000.0 - fill.notional - fill.fee,
                               places=9)

    def test_exit_credits_notional_minus_fee_exactly_as_before(self):
        """The new expression is margin + move - fee; it must equal the old one."""
        pos, _ = self._open()
        exit_fill = self.broker.sell("SOL/USDT", pos.qty, 110.0, meta=META)
        cash_before = self.acct.cash()
        self.acct.close_position(pos, exit_fill, "target")
        old_formula = cash_before + exit_fill.notional - exit_fill.fee
        self.assertAlmostEqual(self.acct.cash(), old_formula, places=9)

    def test_round_trip_cash_change_equals_net_pnl(self):
        start = self.acct.cash()
        pos, _ = self._open()
        exit_fill = self.broker.sell("SOL/USDT", pos.qty, 112.0, meta=META)
        trade = self.acct.close_position(pos, exit_fill, "target")
        self.assertAlmostEqual(self.acct.cash() - start, trade.net_pnl, places=6)

    def test_a_losing_round_trip_also_reconciles(self):
        start = self.acct.cash()
        pos, _ = self._open()
        exit_fill = self.broker.sell("SOL/USDT", pos.qty, 88.0, meta=META)
        trade = self.acct.close_position(pos, exit_fill, "stop")
        self.assertLess(trade.net_pnl, 0)
        self.assertAlmostEqual(self.acct.cash() - start, trade.net_pnl, places=6)

    def test_equity_matches_the_old_cash_plus_mark_value_formula(self):
        self._open()
        marks = {"SOL/USDT": 107.5}
        old = self.acct.cash() + sum(
            p.qty * marks[p.symbol] for p in self.acct.positions())
        self.assertAlmostEqual(self.acct.equity(marks), old, places=9)

    def test_realise_pnl_is_unchanged_for_a_long(self):
        """The direction argument defaults to +1 and must change nothing."""
        args = (100.0, 100.5, 110.0, 109.4, 10.0, 0.75, 0.82)
        self.assertEqual(realise_pnl(*args), realise_pnl(*args, direction=1))

    def test_a_long_trade_records_its_side(self):
        pos, _ = self._open()
        trade = self.acct.close_position(
            pos, self.broker.sell("SOL/USDT", pos.qty, 105.0, meta=META), "target")
        self.assertEqual(trade.side, "long")


V3_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    starting_equity REAL NOT NULL, cash REAL NOT NULL, peak_equity REAL NOT NULL,
    daily_start_equity REAL NOT NULL, daily_date TEXT NOT NULL,
    realized_pnl REAL NOT NULL DEFAULT 0, total_fees REAL NOT NULL DEFAULT 0,
    total_slippage REAL NOT NULL DEFAULT 0, halted INTEGER NOT NULL DEFAULT 0,
    halt_reason TEXT NOT NULL DEFAULT '', halt_ms INTEGER NOT NULL DEFAULT 0,
    created_ms INTEGER NOT NULL, updated_ms INTEGER NOT NULL);
CREATE TABLE positions (
    id TEXT PRIMARY KEY, symbol TEXT NOT NULL, strategy TEXT NOT NULL,
    strategy_version TEXT NOT NULL, side TEXT NOT NULL, qty REAL NOT NULL,
    entry_ref_price REAL NOT NULL, entry_fill_price REAL NOT NULL,
    entry_ms INTEGER NOT NULL, entry_fee REAL NOT NULL, entry_slippage REAL NOT NULL,
    initial_stop REAL NOT NULL, current_stop REAL NOT NULL,
    highest_price REAL NOT NULL, lowest_price REAL NOT NULL,
    risk_amount REAL NOT NULL, candle_id TEXT NOT NULL UNIQUE,
    signal_score REAL NOT NULL DEFAULT 0, mfe REAL NOT NULL DEFAULT 0,
    mae REAL NOT NULL DEFAULT 0, journal TEXT NOT NULL DEFAULT '{}');
CREATE TABLE trades (
    id TEXT PRIMARY KEY, position_id TEXT NOT NULL, symbol TEXT NOT NULL,
    strategy TEXT NOT NULL, strategy_version TEXT NOT NULL, qty REAL NOT NULL,
    entry_ref_price REAL NOT NULL, entry_fill_price REAL NOT NULL,
    entry_ms INTEGER NOT NULL, exit_ref_price REAL NOT NULL,
    exit_fill_price REAL NOT NULL, exit_ms INTEGER NOT NULL,
    exit_reason TEXT NOT NULL, initial_stop REAL NOT NULL, final_stop REAL NOT NULL,
    gross_pnl REAL NOT NULL, fees REAL NOT NULL, slippage_cost REAL NOT NULL,
    net_pnl REAL NOT NULL, return_pct REAL NOT NULL, account_return_pct REAL NOT NULL,
    mfe REAL NOT NULL, mae REAL NOT NULL, duration_s REAL NOT NULL,
    equity_after REAL NOT NULL, journal TEXT NOT NULL DEFAULT '{}');
CREATE TABLE processed_candles (
    candle_id TEXT PRIMARY KEY, processed_ms INTEGER NOT NULL);
"""


def build_v3_database(path: str) -> dict:
    """A pre-refactor database, exactly as a running Strategy A left it."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(V3_SCHEMA)
    ts = now_ms()
    acct = dict(starting_equity=10_000.0, cash=8_734.51, peak_equity=10_412.88,
                daily_start_equity=10_180.02, daily_date=utc_date(ts),
                realized_pnl=412.88, total_fees=31.44, total_slippage=17.09,
                halted=0, halt_reason="", halt_ms=0, created_ms=ts - 999,
                updated_ms=ts)
    conn.execute(
        """INSERT INTO account(id, starting_equity, cash, peak_equity,
               daily_start_equity, daily_date, realized_pnl, total_fees,
               total_slippage, halted, halt_reason, halt_ms, created_ms, updated_ms)
           VALUES(1,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        tuple(acct[k] for k in (
            "starting_equity", "cash", "peak_equity", "daily_start_equity",
            "daily_date", "realized_pnl", "total_fees", "total_slippage",
            "halted", "halt_reason", "halt_ms", "created_ms", "updated_ms")))
    conn.execute(
        """INSERT INTO positions VALUES(
            'pos_old','SOL/USDT','trend_breakout','1.0.0','long',12.5,
            101.0,101.4,?,0.95,5.0,96.3,98.1,109.0,100.0,62.5,
            'SOL/USDT|1h|7777',71.5,90.0,-14.0,'{}')""", (ts - 86_400_000,))
    conn.execute(
        """INSERT INTO trades VALUES(
            'trd_old','pos_gone','ETH/USDT','trend_breakout','1.0.0',0.5,
            2000.0,2002.0,?,2100.0,2098.0,?, 'target',1900.0,2050.0,
            50.0,3.0,2.0,45.0,4.5,0.45,60.0,-10.0,7200.0,10412.88,'{}')""",
        (ts - 200_000, ts - 100_000))
    conn.execute("INSERT INTO processed_candles VALUES('SOL/USDT|1h|7777', ?)", (ts,))
    conn.execute("INSERT INTO meta VALUES('schema_version','3')")
    conn.commit()
    conn.close()
    return acct


class TestTheV3MigrationPreservesEverything(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.path = str(self.dir / "v3.db")
        self.before = build_v3_database(self.path)
        self.conn = db.connect(self.path)
        db.init_db(self.conn)
        self.repo = Repo(self.conn)

    def test_the_schema_version_advances(self):
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()
        self.assertEqual(int(row["value"]), db.SCHEMA_VERSION)
        self.assertEqual(db.SCHEMA_VERSION, 4)

    def test_the_account_becomes_a_sub_account_owned_by_strategy_a(self):
        acct = self.repo.get_account("trend_breakout")
        for field, want in self.before.items():
            if field == "updated_ms":
                continue          # the migration stamps its own write time
            self.assertEqual(acct[field], want, field)

    def test_the_original_account_row_is_kept_for_audit(self):
        """Nothing is destroyed: the pre-migration table is still readable."""
        old = self.conn.execute("SELECT * FROM account_pre_v4 WHERE id=1").fetchone()
        new = self.repo.get_account("trend_breakout")
        for field in ("starting_equity", "cash", "peak_equity", "realized_pnl",
                      "total_fees", "total_slippage", "daily_start_equity"):
            self.assertAlmostEqual(old[field], new[field], places=9, msg=field)

    def test_no_second_sub_account_is_invented(self):
        self.assertEqual([a["strategy"] for a in self.repo.all_accounts()],
                         ["trend_breakout"])

    def test_the_open_position_survives_with_its_collateral_filled_in(self):
        pos = self.repo.get_position("trend_breakout", "SOL/USDT")
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos.qty, 12.5)
        self.assertAlmostEqual(pos.entry_fill_price, 101.4)
        self.assertAlmostEqual(pos.margin_held, 12.5 * 101.4, places=9,
                               msg="a pre-v4 long is fully collateralised")

    def test_equity_after_migration_equals_the_old_formula(self):
        """The number an operator would see must not move."""
        marks = {"SOL/USDT": 105.25}
        acct = PaperAccount(self.repo, PaperBroker(7.5, 6.0, 15.0), 10_000.0,
                            "trend_breakout")
        old = float(self.before["cash"]) + sum(
            p.qty * marks[p.symbol] for p in self.repo.get_positions("trend_breakout"))
        self.assertAlmostEqual(acct.equity(marks), old, places=9)

    def test_the_closed_trade_is_unchanged_and_reads_as_a_long(self):
        trades = self.repo.get_trades("trend_breakout")
        self.assertEqual(len(trades), 1)
        self.assertAlmostEqual(trades[0]["net_pnl"], 45.0)
        self.assertEqual(trades[0]["side"], "long",
                         "pre-v4 trades were all longs and must read as such")

    def test_the_processed_candle_is_still_claimed_by_strategy_a(self):
        self.assertTrue(self.repo.is_candle_processed(
            "SOL/USDT|1h|7777", "trend_breakout"))

    def test_that_claim_does_not_leak_to_another_strategy(self):
        self.assertFalse(self.repo.is_candle_processed(
            "SOL/USDT|1h|7777", "aggressive_momentum_v2"))

    def test_a_shared_pre_v4_ledger_refuses_to_migrate_rather_than_guess(self):
        """Two strategies on one old account cannot be split after the fact."""
        path = str(self.dir / "shared.db")
        build_v3_database(path)
        c = sqlite3.connect(path)
        c.execute("""INSERT INTO trades VALUES(
            'trd_b','pos_b','BTC/USDT','some_other_strategy','9',1.0,
            1.0,1.0,1,1.0,1.0,2,'x',1.0,1.0,0,0,0,0,0,0,0,0,1,1,'{}')""")
        c.commit(); c.close()
        conn = db.connect(path)
        with self.assertRaises(RuntimeError) as ctx:
            db.init_db(conn)
        self.assertIn("cannot migrate", str(ctx.exception))

    def test_migrating_twice_is_a_no_op(self):
        self.conn.close()
        conn2 = db.connect(self.path)
        db.init_db(conn2)                      # already at v4
        repo2 = Repo(conn2)
        self.assertEqual(len(repo2.all_accounts()), 1)
        self.assertAlmostEqual(repo2.get_account("trend_breakout")["cash"],
                               self.before["cash"])


class TestAFreshDatabaseNeedsNoMigration(unittest.TestCase):
    def test_a_new_database_is_created_at_v4_directly(self):
        repo, path = temp_repo()
        row = repo.conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()
        self.assertEqual(int(row["value"]), 4)

    def test_a_new_database_has_no_legacy_account_table(self):
        repo, path = temp_repo()
        names = {r[0] for r in repo.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        self.assertIn("sub_accounts", names)
        self.assertNotIn("account", names,
                         "the single-account table must not be recreated")


if __name__ == "__main__":
    unittest.main()
