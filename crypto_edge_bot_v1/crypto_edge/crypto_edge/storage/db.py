"""SQLite schema and connection management.

Design notes:
  * WAL journalling + FULL synchronous: we accept slower writes in exchange for
    surviving an ungraceful kill without a torn account state.
  * Every external-information table carries BOTH `event_time` (when the thing
    happened) and `observed_at` (when this system could first have known about
    it). Backtests must filter on `observed_at`, which is what makes news and
    derivatives look-ahead structurally impossible rather than merely discouraged.
  * `processed_candles` is the durable duplicate-entry guard: a (symbol,
    timeframe, candle) tuple can only ever be acted on once, across restarts.
  * `telegram_outbox` is a real outbox, not a "have we seen this key" set. A
    row is written PENDING *before* the send and only flipped to SENT once the
    transport confirms delivery, so a failed send stays recoverable across a
    restart while a delivered one stays deduplicated forever.
  * `broad_universe_cache` is append-only. Every fetch of the broad
    (market-cap) asset ranking is kept with its source, its provider timestamp
    and a content hash, so a universe decision made on any past day can be
    reproduced exactly and an outage can fall back to the last valid snapshot.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from ..timeutils import now_ms

SCHEMA_VERSION = 6

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One independent paper ledger PER STRATEGY. Two strategies sharing a cash
-- balance cannot be compared: each one's position sizes would depend on what
-- the other happened to be holding at the time.
CREATE TABLE IF NOT EXISTS sub_accounts (
    strategy TEXT PRIMARY KEY,
    starting_equity REAL NOT NULL,
    cash REAL NOT NULL,
    peak_equity REAL NOT NULL,
    daily_start_equity REAL NOT NULL,
    daily_date TEXT NOT NULL,
    realized_pnl REAL NOT NULL DEFAULT 0,
    total_fees REAL NOT NULL DEFAULT 0,
    total_slippage REAL NOT NULL DEFAULT 0,
    total_financing REAL NOT NULL DEFAULT 0,
    halted INTEGER NOT NULL DEFAULT 0,
    halt_reason TEXT NOT NULL DEFAULT '',
    halt_ms INTEGER NOT NULL DEFAULT 0,
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    entry_ref_price REAL NOT NULL,
    entry_fill_price REAL NOT NULL,
    entry_ms INTEGER NOT NULL,
    entry_fee REAL NOT NULL,
    entry_slippage REAL NOT NULL,
    initial_stop REAL NOT NULL,
    current_stop REAL NOT NULL,
    highest_price REAL NOT NULL,
    lowest_price REAL NOT NULL,
    risk_amount REAL NOT NULL,
    candle_id TEXT NOT NULL,
    signal_score REAL NOT NULL DEFAULT 0,
    mfe REAL NOT NULL DEFAULT 0,
    mae REAL NOT NULL DEFAULT 0,
    margin_held REAL NOT NULL DEFAULT 0,
    journal TEXT NOT NULL DEFAULT '{}',
    -- SCOPED BY STRATEGY. A bare UNIQUE(candle_id) let whichever strategy
    -- reached a candle first lock every other strategy out of it.
    UNIQUE (candle_id, strategy)
);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE UNIQUE INDEX IF NOT EXISTS idx_positions_open
    ON positions(strategy, symbol);

CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    position_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    side TEXT NOT NULL DEFAULT 'long',
    qty REAL NOT NULL,
    entry_ref_price REAL NOT NULL,
    entry_fill_price REAL NOT NULL,
    entry_ms INTEGER NOT NULL,
    exit_ref_price REAL NOT NULL,
    exit_fill_price REAL NOT NULL,
    exit_ms INTEGER NOT NULL,
    exit_reason TEXT NOT NULL,
    initial_stop REAL NOT NULL,
    final_stop REAL NOT NULL,
    gross_pnl REAL NOT NULL,
    fees REAL NOT NULL,
    slippage_cost REAL NOT NULL,
    net_pnl REAL NOT NULL,
    financing REAL NOT NULL DEFAULT 0,
    return_pct REAL NOT NULL,
    account_return_pct REAL NOT NULL,
    mfe REAL NOT NULL,
    mae REAL NOT NULL,
    duration_s REAL NOT NULL,
    equity_after REAL NOT NULL,
    journal TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_trades_exit ON trades(exit_ms);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_version ON trades(strategy, strategy_version);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    ts_ms INTEGER PRIMARY KEY,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    unrealized REAL NOT NULL,
    open_positions INTEGER NOT NULL,
    drawdown_pct REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_stats (
    date TEXT PRIMARY KEY,
    start_equity REAL NOT NULL,
    end_equity REAL NOT NULL,
    realized_pnl REAL NOT NULL DEFAULT 0,
    trades INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    fees REAL NOT NULL DEFAULT 0
);

-- Claiming a candle is per STRATEGY. With a bare candle_id primary key the
-- first strategy to claim a bar silently prevented every other strategy from
-- ever evaluating it -- no error, just a symbol that never traded.
CREATE TABLE IF NOT EXISTS processed_candles (
    candle_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    processed_ms INTEGER NOT NULL,
    PRIMARY KEY (candle_id, strategy)
);

CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    ts_ms INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    candle_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    decision TEXT NOT NULL,
    reject_reason TEXT NOT NULL DEFAULT '',
    -- A real column, not a JSON key: "are shorts stronger than longs?" is a
    -- GROUP BY, and it is one of the four questions this database exists for.
    side TEXT NOT NULL DEFAULT 'long',
    score REAL,
    rank INTEGER,
    price REAL,
    features TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_obs_ts ON observations(ts_ms);
CREATE INDEX IF NOT EXISTS idx_obs_decision ON observations(decision);
CREATE UNIQUE INDEX IF NOT EXISTS idx_obs_unique ON observations(candle_id, strategy);

CREATE TABLE IF NOT EXISTS counterfactuals (
    observation_id TEXT NOT NULL,
    horizon_h INTEGER NOT NULL,
    entry_ref REAL NOT NULL,
    price_at REAL,
    return_pct REAL,
    evaluated_ms INTEGER,
    hypothetical INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (observation_id, horizon_h)
);

CREATE TABLE IF NOT EXISTS universe_snapshots (
    ts_ms INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    included INTEGER NOT NULL,
    reject_reason TEXT NOT NULL DEFAULT '',
    dollar_volume REAL,
    spread_bps REAL,
    rank INTEGER,
    PRIMARY KEY (ts_ms, symbol)
);

-- ---- external information: event_time vs observed_at ---------------------
CREATE TABLE IF NOT EXISTS news_events (
    id TEXT PRIMARY KEY,
    event_time INTEGER NOT NULL,
    observed_at INTEGER NOT NULL,
    source TEXT NOT NULL,
    source_tier INTEGER NOT NULL,
    headline TEXT NOT NULL,
    url TEXT NOT NULL DEFAULT '',
    assets TEXT NOT NULL DEFAULT '[]',
    category TEXT NOT NULL DEFAULT 'other',
    sentiment TEXT NOT NULL DEFAULT 'neutral',
    severity INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0,
    caused_restriction INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_news_observed ON news_events(observed_at);

CREATE TABLE IF NOT EXISTS token_events (
    id TEXT PRIMARY KEY,
    event_time INTEGER NOT NULL,
    observed_at INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    kind TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    severity INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_token_events ON token_events(symbol, event_time);

CREATE TABLE IF NOT EXISTS derivatives (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    event_time INTEGER NOT NULL,
    observed_at INTEGER NOT NULL,
    funding_rate REAL,
    funding_change REAL,
    open_interest REAL,
    oi_change_pct REAL,
    liq_long REAL,
    liq_short REAL,
    basis REAL,
    source TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_deriv ON derivatives(symbol, observed_at);

CREATE TABLE IF NOT EXISTS no_trade_list (
    symbol TEXT PRIMARY KEY,
    reason TEXT NOT NULL,
    added_ms INTEGER NOT NULL,
    expires_ms INTEGER NOT NULL
);

-- Durable notification outbox. status: PENDING -> SENT (or -> FAILED once
-- the attempt budget is exhausted). `text` is stored so a message that was
-- never delivered can be retried after a restart.
CREATE TABLE IF NOT EXISTS telegram_outbox (
    dedupe_key TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'PENDING',
    kind TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL DEFAULT '',
    created_ms INTEGER NOT NULL DEFAULT 0,
    claimed_ms INTEGER NOT NULL DEFAULT 0,
    sent_ms INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_outbox_status ON telegram_outbox(status, claimed_ms);

-- Append-only history of the broad (market-cap) universe snapshots.
CREATE TABLE IF NOT EXISTS broad_universe_cache (
    fetched_ms INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    as_of_ms INTEGER NOT NULL,
    n_assets INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_broad_universe ON broad_universe_cache(fetched_ms DESC);

-- Market age, established independently of indicator history and remembered
-- with its provenance. `first_ms` only ever moves EARLIER (see
-- Repo.record_market_age): a venue trimming its history must not make a market
-- appear to grow younger.
CREATE TABLE IF NOT EXISTS market_age (
    symbol TEXT PRIMARY KEY,
    first_ms INTEGER NOT NULL,
    source TEXT NOT NULL,
    observed_ms INTEGER NOT NULL,
    detail TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS strategy_versions (
    strategy TEXT NOT NULL,
    version TEXT NOT NULL,
    activated_ms INTEGER NOT NULL,
    config_json TEXT NOT NULL,
    PRIMARY KEY (strategy, version, activated_ms)
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    p = Path(db_path)
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """v1 had a bare `telegram_outbox(dedupe_key, sent_ms, kind)` whose rows
    meant "this key was claimed", not "this message was delivered". We cannot
    retroactively tell the two apart, so every pre-existing row is migrated as
    SENT: suppressing a possibly-delivered old notification is safe, re-sending
    a batch of stale alerts on upgrade is not."""
    cols = _columns(conn, "telegram_outbox")
    if not cols or "status" in cols:
        return
    conn.executescript("""
        ALTER TABLE telegram_outbox ADD COLUMN status TEXT NOT NULL DEFAULT 'SENT';
        ALTER TABLE telegram_outbox ADD COLUMN text TEXT NOT NULL DEFAULT '';
        ALTER TABLE telegram_outbox ADD COLUMN created_ms INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE telegram_outbox ADD COLUMN claimed_ms INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE telegram_outbox ADD COLUMN attempts INTEGER NOT NULL DEFAULT 1;
        ALTER TABLE telegram_outbox ADD COLUMN last_error TEXT NOT NULL DEFAULT '';
        UPDATE telegram_outbox SET status='SENT', created_ms=sent_ms
         WHERE status IS NULL OR status='';
    """)


def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    """v3 only ADDS the market_age table, which the schema script creates with
    CREATE TABLE IF NOT EXISTS. Nothing to rewrite; this exists so the version
    step is explicit and auditable rather than implied."""
    return


def _legacy_strategy_name(conn: sqlite3.Connection) -> str:
    """Which strategy owns the pre-v4 single-account history.

    Derived from the data rather than from configuration, because the config
    can be edited between runs and the migration has to be correct for the rows
    that actually exist. Positions and trades carry a strategy column already;
    if they agree, that is the answer.
    """
    names: list[str] = []
    for table in ("positions", "trades"):
        if not _columns(conn, table):
            continue
        names += [r[0] for r in conn.execute(
            f"SELECT DISTINCT strategy FROM {table}").fetchall() if r[0]]
    distinct = sorted(set(names))
    if len(distinct) == 1:
        return distinct[0]
    if _columns(conn, "strategy_versions"):
        row = conn.execute(
            "SELECT strategy FROM strategy_versions ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if row and row[0]:
            return str(row[0])
    if distinct:
        # More than one strategy already traded on a single shared ledger. That
        # ledger cannot be split after the fact, so refuse rather than guess.
        raise RuntimeError(
            "cannot migrate to schema v4: the pre-v4 account was shared by "
            f"strategies {distinct}. A shared ledger cannot be attributed "
            "retroactively; restore a backup or start a fresh database.")
    return "trend_breakout"


def _migrate_v3_to_v4(conn: sqlite3.Connection) -> None:
    """Single account -> per-strategy sub-accounts, and strategy-scoped claims.

    Nothing is discarded. The old `account` row is copied into `sub_accounts`
    under the strategy that actually traded it, and the original table is kept
    as `account_pre_v4` so the migration can be audited against the source.
    """
    owner = _legacy_strategy_name(conn)
    ts = now_ms()

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sub_accounts (
            strategy TEXT PRIMARY KEY,
            starting_equity REAL NOT NULL,
            cash REAL NOT NULL,
            peak_equity REAL NOT NULL,
            daily_start_equity REAL NOT NULL,
            daily_date TEXT NOT NULL,
            realized_pnl REAL NOT NULL DEFAULT 0,
            total_fees REAL NOT NULL DEFAULT 0,
            total_slippage REAL NOT NULL DEFAULT 0,
            total_financing REAL NOT NULL DEFAULT 0,
            halted INTEGER NOT NULL DEFAULT 0,
            halt_reason TEXT NOT NULL DEFAULT '',
            halt_ms INTEGER NOT NULL DEFAULT 0,
            created_ms INTEGER NOT NULL,
            updated_ms INTEGER NOT NULL
        );
    """)

    if _columns(conn, "account"):
        row = conn.execute("SELECT * FROM account WHERE id=1").fetchone()
        if row is not None:
            d = dict(row)
            conn.execute(
                """INSERT OR IGNORE INTO sub_accounts(
                       strategy, starting_equity, cash, peak_equity,
                       daily_start_equity, daily_date, realized_pnl, total_fees,
                       total_slippage, total_financing, halted, halt_reason,
                       halt_ms, created_ms, updated_ms)
                   VALUES(?,?,?,?,?,?,?,?,?,0,?,?,?,?,?)""",
                (owner, d["starting_equity"], d["cash"], d["peak_equity"],
                 d["daily_start_equity"], d["daily_date"], d["realized_pnl"],
                 d["total_fees"], d["total_slippage"], d["halted"],
                 d["halt_reason"], d["halt_ms"], d["created_ms"], ts))
        # Kept, not dropped: the source of truth for verifying this migration.
        conn.execute("ALTER TABLE account RENAME TO account_pre_v4")

    # --- processed_candles: (candle_id) -> (candle_id, strategy) -------------
    pc_cols = _columns(conn, "processed_candles")
    if pc_cols and "strategy" not in pc_cols:
        conn.executescript(f"""
            ALTER TABLE processed_candles RENAME TO processed_candles_pre_v4;
            CREATE TABLE processed_candles (
                candle_id TEXT NOT NULL,
                strategy TEXT NOT NULL,
                processed_ms INTEGER NOT NULL,
                PRIMARY KEY (candle_id, strategy)
            );
            INSERT INTO processed_candles(candle_id, strategy, processed_ms)
                SELECT candle_id, '{owner}', processed_ms
                  FROM processed_candles_pre_v4;
            DROP TABLE processed_candles_pre_v4;
        """)

    # --- positions: drop UNIQUE(candle_id), add margin_held -----------------
    pos_cols = _columns(conn, "positions")
    if pos_cols and "margin_held" not in pos_cols:
        conn.executescript("""
            ALTER TABLE positions RENAME TO positions_pre_v4;
            CREATE TABLE positions (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                strategy_version TEXT NOT NULL,
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                entry_ref_price REAL NOT NULL,
                entry_fill_price REAL NOT NULL,
                entry_ms INTEGER NOT NULL,
                entry_fee REAL NOT NULL,
                entry_slippage REAL NOT NULL,
                initial_stop REAL NOT NULL,
                current_stop REAL NOT NULL,
                highest_price REAL NOT NULL,
                lowest_price REAL NOT NULL,
                risk_amount REAL NOT NULL,
                candle_id TEXT NOT NULL,
                signal_score REAL NOT NULL DEFAULT 0,
                mfe REAL NOT NULL DEFAULT 0,
                mae REAL NOT NULL DEFAULT 0,
                margin_held REAL NOT NULL DEFAULT 0,
                journal TEXT NOT NULL DEFAULT '{}',
                UNIQUE (candle_id, strategy)
            );
            INSERT INTO positions(
                id, symbol, strategy, strategy_version, side, qty,
                entry_ref_price, entry_fill_price, entry_ms, entry_fee,
                entry_slippage, initial_stop, current_stop, highest_price,
                lowest_price, risk_amount, candle_id, signal_score, mfe, mae,
                margin_held, journal)
            SELECT
                id, symbol, strategy, strategy_version, side, qty,
                entry_ref_price, entry_fill_price, entry_ms, entry_fee,
                entry_slippage, initial_stop, current_stop, highest_price,
                lowest_price, risk_amount, candle_id, signal_score, mfe, mae,
                -- a pre-v4 position is a 1x long, whose collateral IS its
                -- entry notional; that identity is what keeps equity unchanged
                qty * entry_fill_price, journal
            FROM positions_pre_v4;
            DROP TABLE positions_pre_v4;
        """)

    # --- trades: record which side the trade was -----------------------------
    trade_cols = _columns(conn, "trades")
    if trade_cols and "side" not in trade_cols:
        conn.execute(
            "ALTER TABLE trades ADD COLUMN side TEXT NOT NULL DEFAULT 'long'")


def _migrate_v4_to_v5(conn: sqlite3.Connection) -> None:
    """Record which SIDE each observation was. Pre-v5 rows were all longs."""
    cols = _columns(conn, "observations")
    if cols and "side" not in cols:
        conn.execute(
            "ALTER TABLE observations ADD COLUMN side TEXT NOT NULL DEFAULT 'long'")


def _migrate_v5_to_v6(conn: sqlite3.Connection) -> None:
    """Record simulated short financing per trade. Longs pay none, and every
    pre-v6 trade was a long, so backfilling zero is exact rather than assumed."""
    cols = _columns(conn, "trades")
    if cols and "financing" not in cols:
        conn.execute(
            "ALTER TABLE trades ADD COLUMN financing REAL NOT NULL DEFAULT 0")


MIGRATIONS = {1: _migrate_v1_to_v2, 2: _migrate_v2_to_v3, 3: _migrate_v3_to_v4,
              4: _migrate_v4_to_v5, 5: _migrate_v5_to_v6}


def init_db(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT value FROM meta WHERE key='schema_version'").fetchone() \
        if _columns(conn, "meta") else None
    version = int(row["value"]) if row else None

    if version is not None and version != SCHEMA_VERSION:
        while version in MIGRATIONS:
            MIGRATIONS[version](conn)
            version += 1
        if version != SCHEMA_VERSION:
            raise RuntimeError(
                f"database schema version {row['value']} != expected "
                f"{SCHEMA_VERSION} and no migration path exists")

    conn.executescript(SCHEMA)
    if version is None:
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),))
    elif version != int(row["value"]):
        conn.execute("UPDATE meta SET value=? WHERE key='schema_version'",
                     (str(SCHEMA_VERSION),))


def integrity_ok(conn: sqlite3.Connection) -> tuple[bool, str]:
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        val = row[0] if row else "unknown"
        return (val == "ok"), str(val)
    except sqlite3.Error as e:
        return False, str(e)
