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

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    starting_equity REAL NOT NULL,
    cash REAL NOT NULL,
    peak_equity REAL NOT NULL,
    daily_start_equity REAL NOT NULL,
    daily_date TEXT NOT NULL,
    realized_pnl REAL NOT NULL DEFAULT 0,
    total_fees REAL NOT NULL DEFAULT 0,
    total_slippage REAL NOT NULL DEFAULT 0,
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
    candle_id TEXT NOT NULL UNIQUE,
    signal_score REAL NOT NULL DEFAULT 0,
    mfe REAL NOT NULL DEFAULT 0,
    mae REAL NOT NULL DEFAULT 0,
    journal TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);

CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    position_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
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

CREATE TABLE IF NOT EXISTS processed_candles (
    candle_id TEXT PRIMARY KEY,
    processed_ms INTEGER NOT NULL
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


MIGRATIONS = {1: _migrate_v1_to_v2}


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
