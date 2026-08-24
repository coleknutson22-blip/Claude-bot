"""Repository layer: all SQLite reads/writes live here.

The account row and the positions/trades tables are mutated together inside
explicit transactions, so a crash mid-trade can never leave cash and positions
disagreeing.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from typing import Any, Iterable

from ..models import ClosedTrade, Position
from ..timeutils import now_ms, utc_date


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


class Repo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # ------------------------------------------------------------------ tx
    @contextmanager
    def tx(self):
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    # -------------------------------------------------------------- account
    def ensure_account(self, starting_equity: float) -> dict:
        row = self.conn.execute("SELECT * FROM account WHERE id=1").fetchone()
        if row:
            return dict(row)
        ts = now_ms()
        self.conn.execute(
            """INSERT INTO account(id, starting_equity, cash, peak_equity,
                   daily_start_equity, daily_date, created_ms, updated_ms)
               VALUES(1,?,?,?,?,?,?,?)""",
            (starting_equity, starting_equity, starting_equity, starting_equity,
             utc_date(ts), ts, ts))
        return dict(self.conn.execute("SELECT * FROM account WHERE id=1").fetchone())

    def get_account(self) -> dict:
        row = self.conn.execute("SELECT * FROM account WHERE id=1").fetchone()
        if not row:
            raise RuntimeError("account row missing -- state is not trustworthy")
        return dict(row)

    def update_account(self, **fields) -> None:
        if not fields:
            return
        fields["updated_ms"] = now_ms()
        sets = ", ".join(f"{k}=?" for k in fields)
        self.conn.execute(f"UPDATE account SET {sets} WHERE id=1", tuple(fields.values()))

    def set_halt(self, halted: bool, reason: str = "") -> None:
        self.update_account(halted=1 if halted else 0, halt_reason=reason,
                            halt_ms=now_ms() if halted else 0)

    # ------------------------------------------------------------ positions
    def add_position(self, pos: Position) -> None:
        d = pos.to_row()
        cols = ", ".join(d.keys())
        marks = ", ".join("?" for _ in d)
        self.conn.execute(f"INSERT INTO positions({cols}) VALUES({marks})",
                          tuple(d.values()))

    def update_position(self, pos_id: str, **fields) -> None:
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        self.conn.execute(f"UPDATE positions SET {sets} WHERE id=?",
                          tuple(fields.values()) + (pos_id,))

    def remove_position(self, pos_id: str) -> None:
        self.conn.execute("DELETE FROM positions WHERE id=?", (pos_id,))

    def get_positions(self) -> list[Position]:
        rows = self.conn.execute("SELECT * FROM positions ORDER BY entry_ms").fetchall()
        return [self._row_to_position(r) for r in rows]

    def get_position_by_symbol(self, symbol: str) -> Position | None:
        r = self.conn.execute("SELECT * FROM positions WHERE symbol=?", (symbol,)).fetchone()
        return self._row_to_position(r) if r else None

    @staticmethod
    def _row_to_position(r: sqlite3.Row) -> Position:
        d = dict(r)
        d["journal"] = json.loads(d.get("journal") or "{}")
        return Position(**d)

    # --------------------------------------------------------------- trades
    def add_trade(self, t: ClosedTrade) -> None:
        d = dict(t.__dict__)
        d["journal"] = json.dumps(d["journal"], default=str)
        cols = ", ".join(d.keys())
        marks = ", ".join("?" for _ in d)
        self.conn.execute(f"INSERT INTO trades({cols}) VALUES({marks})", tuple(d.values()))

    def get_trades(self, strategy: str | None = None,
                   version: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM trades", []
        conds = []
        if strategy:
            conds.append("strategy=?"); args.append(strategy)
        if version:
            conds.append("strategy_version=?"); args.append(version)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY exit_ms"
        out = []
        for r in self.conn.execute(q, tuple(args)).fetchall():
            d = dict(r)
            d["journal"] = json.loads(d.get("journal") or "{}")
            out.append(d)
        return out

    # ------------------------------------------------- duplicate protection
    def is_candle_processed(self, cid: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM processed_candles WHERE candle_id=?", (cid,)).fetchone() is not None

    def mark_candle_processed(self, cid: str) -> bool:
        """Returns True if this call claimed the candle, False if already taken.
        Atomic, so two code paths cannot both act on one candle."""
        try:
            self.conn.execute(
                "INSERT INTO processed_candles(candle_id, processed_ms) VALUES(?,?)",
                (cid, now_ms()))
            return True
        except sqlite3.IntegrityError:
            return False

    def prune_processed_candles(self, older_than_ms: int) -> None:
        self.conn.execute("DELETE FROM processed_candles WHERE processed_ms < ?",
                          (older_than_ms,))

    # ------------------------------------------------------------ snapshots
    def add_equity_snapshot(self, ts: int, equity: float, cash: float,
                            unrealized: float, n_open: int, dd_pct: float) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO equity_snapshots
               (ts_ms, equity, cash, unrealized, open_positions, drawdown_pct)
               VALUES(?,?,?,?,?,?)""", (ts, equity, cash, unrealized, n_open, dd_pct))

    def get_equity_curve(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM equity_snapshots ORDER BY ts_ms")]

    def upsert_daily(self, date: str, start_equity: float, end_equity: float,
                     **inc) -> None:
        self.conn.execute(
            """INSERT INTO daily_stats(date, start_equity, end_equity)
               VALUES(?,?,?)
               ON CONFLICT(date) DO UPDATE SET end_equity=excluded.end_equity""",
            (date, start_equity, end_equity))
        if inc:
            sets = ", ".join(f"{k}={k}+?" for k in inc)
            self.conn.execute(f"UPDATE daily_stats SET {sets} WHERE date=?",
                              tuple(inc.values()) + (date,))

    def get_daily(self, date: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM daily_stats WHERE date=?", (date,)).fetchone()
        return dict(r) if r else None

    def get_all_daily(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM daily_stats ORDER BY date")]

    # ---------------------------------------------------------- observations
    def add_observation(self, obs: dict) -> str | None:
        obs = dict(obs)
        obs.setdefault("id", new_id("obs"))
        obs["features"] = json.dumps(obs.get("features", {}), default=str)
        cols = ", ".join(obs.keys())
        marks = ", ".join("?" for _ in obs)
        try:
            self.conn.execute(f"INSERT INTO observations({cols}) VALUES({marks})",
                              tuple(obs.values()))
        except sqlite3.IntegrityError:
            return None      # same candle already recorded for this strategy
        return obs["id"]

    def get_observations(self, decision: str | None = None,
                         since_ms: int | None = None) -> list[dict]:
        q, args, conds = "SELECT * FROM observations", [], []
        if decision:
            conds.append("decision=?"); args.append(decision)
        if since_ms is not None:
            conds.append("ts_ms >= ?"); args.append(since_ms)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY ts_ms"
        out = []
        for r in self.conn.execute(q, tuple(args)):
            d = dict(r)
            d["features"] = json.loads(d.get("features") or "{}")
            out.append(d)
        return out

    def add_counterfactual(self, obs_id: str, horizon_h: int, entry_ref: float,
                           price_at: float | None, return_pct: float | None,
                           evaluated_ms: int | None) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO counterfactuals
               (observation_id, horizon_h, entry_ref, price_at, return_pct,
                evaluated_ms, hypothetical)
               VALUES(?,?,?,?,?,?,1)""",
            (obs_id, horizon_h, entry_ref, price_at, return_pct, evaluated_ms))

    def pending_counterfactuals(self, horizons: Iterable[int], now: int) -> list[dict]:
        """Observations whose horizon has elapsed but which have no result yet."""
        out = []
        for h in horizons:
            cutoff = now - h * 3_600_000
            rows = self.conn.execute(
                """SELECT o.id, o.symbol, o.ts_ms, o.price FROM observations o
                   LEFT JOIN counterfactuals c
                     ON c.observation_id = o.id AND c.horizon_h = ?
                   WHERE o.decision != 'ENTERED' AND o.price IS NOT NULL
                     AND o.ts_ms <= ? AND c.observation_id IS NULL
                   LIMIT 500""", (h, cutoff)).fetchall()
            for r in rows:
                d = dict(r); d["horizon_h"] = h
                out.append(d)
        return out

    # -------------------------------------------------------------- universe
    def add_universe_snapshot(self, ts: int, rows: list[dict]) -> None:
        self.conn.executemany(
            """INSERT OR REPLACE INTO universe_snapshots
               (ts_ms, symbol, included, reject_reason, dollar_volume, spread_bps, rank)
               VALUES(?,?,?,?,?,?,?)""",
            [(ts, r["symbol"], 1 if r.get("included") else 0, r.get("reject_reason", ""),
              r.get("dollar_volume"), r.get("spread_bps"), r.get("rank")) for r in rows])

    # ------------------------------------------------------------ intel data
    def add_news_event(self, ev: dict) -> None:
        ev = dict(ev)
        ev.setdefault("id", new_id("news"))
        ev["assets"] = json.dumps(ev.get("assets", []))
        cols = ", ".join(ev.keys()); marks = ", ".join("?" for _ in ev)
        self.conn.execute(
            f"INSERT OR REPLACE INTO news_events({cols}) VALUES({marks})", tuple(ev.values()))

    def news_visible_at(self, as_of_ms: int, since_ms: int | None = None) -> list[dict]:
        """Only events this system could actually have known about by `as_of_ms`.
        This is the guard against news look-ahead in backtests."""
        q = "SELECT * FROM news_events WHERE observed_at <= ?"
        args: list[Any] = [as_of_ms]
        if since_ms is not None:
            q += " AND observed_at >= ?"; args.append(since_ms)
        out = []
        for r in self.conn.execute(q + " ORDER BY observed_at", tuple(args)):
            d = dict(r); d["assets"] = json.loads(d.get("assets") or "[]")
            out.append(d)
        return out

    def add_token_event(self, ev: dict) -> None:
        ev = dict(ev); ev.setdefault("id", new_id("tok"))
        cols = ", ".join(ev.keys()); marks = ", ".join("?" for _ in ev)
        self.conn.execute(
            f"INSERT OR REPLACE INTO token_events({cols}) VALUES({marks})", tuple(ev.values()))

    def token_events_visible(self, symbol: str, as_of_ms: int,
                             window_ms: int) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            """SELECT * FROM token_events
               WHERE symbol=? AND observed_at <= ?
                 AND event_time BETWEEN ? AND ?""",
            (symbol, as_of_ms, as_of_ms, as_of_ms + window_ms))]

    def add_derivatives(self, rec: dict) -> None:
        rec = dict(rec); rec.setdefault("id", new_id("drv"))
        cols = ", ".join(rec.keys()); marks = ", ".join("?" for _ in rec)
        self.conn.execute(
            f"INSERT OR REPLACE INTO derivatives({cols}) VALUES({marks})", tuple(rec.values()))

    def latest_derivatives(self, symbol: str, as_of_ms: int) -> dict | None:
        r = self.conn.execute(
            """SELECT * FROM derivatives WHERE symbol=? AND observed_at <= ?
               ORDER BY observed_at DESC LIMIT 1""", (symbol, as_of_ms)).fetchone()
        return dict(r) if r else None

    # --------------------------------------------------------- no-trade list
    def block_symbol(self, symbol: str, reason: str, expires_ms: int) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO no_trade_list(symbol, reason, added_ms, expires_ms)
               VALUES(?,?,?,?)""", (symbol, reason, now_ms(), expires_ms))

    def blocked_symbols(self, as_of_ms: int) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT symbol, reason FROM no_trade_list WHERE expires_ms > ?", (as_of_ms,))
        return {r["symbol"]: r["reason"] for r in rows}

    # ------------------------------------------------------------- telegram
    # Durable outbox. The state machine is:
    #
    #     (no row) --claim--> PENDING --deliver--> SENT     [terminal]
    #                            |  ^                 
    #                            |  +--release--+     (failed attempt, retryable)
    #                            +--exhausted--> FAILED    [terminal, audited]
    #
    # Nothing is ever marked SENT before the transport confirms delivery, which
    # is what makes a failed send recoverable instead of silently swallowed.
    CLAIMED, ALREADY_SENT, IN_FLIGHT, GAVE_UP = (
        "CLAIMED", "ALREADY_SENT", "IN_FLIGHT", "GAVE_UP")

    def telegram_already_sent(self, key: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM telegram_outbox WHERE dedupe_key=? AND status='SENT'",
            (key,)).fetchone() is not None

    def telegram_status(self, key: str) -> dict | None:
        r = self.conn.execute(
            "SELECT * FROM telegram_outbox WHERE dedupe_key=?", (key,)).fetchone()
        return dict(r) if r else None

    def claim_telegram(self, key: str, kind: str = "", text: str = "",
                       lease_ms: int = 120_000, max_attempts: int = 12) -> str:
        """Reserve the right to deliver `key`, atomically.

        Returns CLAIMED (go ahead and send), ALREADY_SENT (delivered before --
        suppress), IN_FLIGHT (another attempt holds an unexpired lease --
        suppress, this is the anti-duplicate-spam guard) or GAVE_UP (attempt
        budget exhausted).
        """
        now = now_ms()
        try:
            self.conn.execute(
                """INSERT INTO telegram_outbox(dedupe_key, status, kind, text,
                       created_ms, claimed_ms, sent_ms, attempts, last_error)
                   VALUES(?, 'PENDING', ?, ?, ?, ?, 0, 1, '')""",
                (key, kind, text, now, now))
            return self.CLAIMED
        except sqlite3.IntegrityError:
            pass

        row = self.conn.execute(
            "SELECT status, claimed_ms, attempts FROM telegram_outbox WHERE dedupe_key=?",
            (key,)).fetchone()
        if row is None:                       # deleted between the two statements
            return self.IN_FLIGHT
        if row["status"] == "SENT":
            return self.ALREADY_SENT
        if row["status"] == "FAILED":
            return self.GAVE_UP
        if now - int(row["claimed_ms"]) < lease_ms:
            # someone else is mid-delivery; sending now would duplicate it
            return self.IN_FLIGHT
        if int(row["attempts"]) >= max_attempts:
            self.conn.execute(
                "UPDATE telegram_outbox SET status='FAILED' WHERE dedupe_key=?", (key,))
            return self.GAVE_UP
        # compare-and-swap on claimed_ms: only one racer can take the lease
        cur = self.conn.execute(
            """UPDATE telegram_outbox SET claimed_ms=?, attempts=attempts+1
               WHERE dedupe_key=? AND status='PENDING' AND claimed_ms=?""",
            (now, key, int(row["claimed_ms"])))
        return self.CLAIMED if cur.rowcount == 1 else self.IN_FLIGHT

    def mark_telegram_delivered(self, key: str) -> None:
        """Flip to SENT. Only ever called after the transport said OK."""
        self.conn.execute(
            """UPDATE telegram_outbox SET status='SENT', sent_ms=?, claimed_ms=0,
                   last_error='' WHERE dedupe_key=?""", (now_ms(), key))

    def release_telegram_claim(self, key: str, error: str = "",
                               max_attempts: int = 12) -> None:
        """Delivery failed. Drop the lease so the message stays retryable --
        unless the attempt budget is spent, in which case park it as FAILED so
        it stops consuming cycles but remains visible for audit."""
        row = self.conn.execute(
            "SELECT attempts FROM telegram_outbox WHERE dedupe_key=?", (key,)).fetchone()
        if row is None:
            return
        status = "FAILED" if int(row["attempts"]) >= max_attempts else "PENDING"
        self.conn.execute(
            "UPDATE telegram_outbox SET status=?, claimed_ms=0, last_error=? "
            "WHERE dedupe_key=? AND status='PENDING'",
            (status, error[:200], key))

    def pending_telegram(self, limit: int = 50, ready_before_ms: int | None = None
                         ) -> list[dict]:
        """Undelivered messages, oldest first. `ready_before_ms` filters out
        rows whose lease has not yet expired."""
        q = "SELECT * FROM telegram_outbox WHERE status='PENDING'"
        args: list[Any] = []
        if ready_before_ms is not None:
            q += " AND claimed_ms <= ?"
            args.append(ready_before_ms)
        q += " ORDER BY created_ms LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.conn.execute(q, tuple(args))]

    def telegram_outbox_counts(self) -> dict[str, int]:
        return {r["status"]: int(r["n"]) for r in self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM telegram_outbox GROUP BY status")}

    def prune_telegram_outbox(self, older_than_ms: int) -> None:
        """Only delivered rows are prunable. PENDING and FAILED are kept."""
        self.conn.execute(
            "DELETE FROM telegram_outbox WHERE status='SENT' AND sent_ms < ?",
            (older_than_ms,))

    # ------------------------------------------------------- broad universe
    def save_broad_universe(self, source: str, fetched_ms: int, as_of_ms: int,
                            assets: list[dict], content_hash: str,
                            ambiguous: dict | None = None,
                            scanned: int | None = None) -> None:
        """Append-only: every fetch is retained with its provenance.

        The payload is an object rather than a bare list so the ticker-ambiguity
        map travels with the ranking -- a cached universe used during a provider
        outage must be exactly as careful about asset identity as a live one.
        """
        payload = {"assets": assets, "ambiguous": ambiguous or {},
                   "scanned": scanned if scanned is not None else len(assets)}
        self.conn.execute(
            """INSERT OR REPLACE INTO broad_universe_cache
               (fetched_ms, source, as_of_ms, n_assets, content_hash, payload)
               VALUES(?,?,?,?,?,?)""",
            (fetched_ms, source, as_of_ms, len(assets), content_hash,
             json.dumps(payload, separators=(",", ":"))))

    def latest_broad_universe(self, min_assets: int = 1) -> dict | None:
        """Most recent cached snapshot that still meets the size floor."""
        r = self.conn.execute(
            """SELECT * FROM broad_universe_cache WHERE n_assets >= ?
               ORDER BY fetched_ms DESC LIMIT 1""", (min_assets,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        payload = json.loads(d.pop("payload"))
        if isinstance(payload, list):        # snapshot written before ambiguity
            d["assets"], d["ambiguous"], d["scanned"] = payload, {}, len(payload)
        else:
            d["assets"] = payload.get("assets", [])
            d["ambiguous"] = payload.get("ambiguous", {})
            d["scanned"] = payload.get("scanned", len(d["assets"]))
        return d

    def prune_broad_universe(self, keep: int = 60) -> None:
        self.conn.execute(
            """DELETE FROM broad_universe_cache WHERE fetched_ms NOT IN
               (SELECT fetched_ms FROM broad_universe_cache
                ORDER BY fetched_ms DESC LIMIT ?)""", (keep,))

    # ------------------------------------------------------------- versions
    def record_strategy_version(self, strategy: str, version: str, cfg: dict) -> None:
        last = self.conn.execute(
            """SELECT config_json FROM strategy_versions
               WHERE strategy=? AND version=? ORDER BY activated_ms DESC LIMIT 1""",
            (strategy, version)).fetchone()
        blob = json.dumps(cfg, sort_keys=True, default=str)
        if last and last["config_json"] == blob:
            return
        self.conn.execute(
            """INSERT INTO strategy_versions(strategy, version, activated_ms, config_json)
               VALUES(?,?,?,?)""", (strategy, version, now_ms(), blob))
