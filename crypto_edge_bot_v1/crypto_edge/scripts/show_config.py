#!/usr/bin/env python3
"""Print exactly what configuration is IN FORCE, resolved the way the bot does.

Reads nothing by eye: it calls the bot's own `load_config`, so environment
overrides, .env values and CLI-equivalent flags all land the same way they do
in a real run. READ ONLY -- it never opens or creates the database.

Usage (from the directory you normally start the bot in):
    python3 show_config.py [--config PATH] [--env PATH]
"""
import argparse
import os
import sqlite3
import sys

def open_readonly(path):
    """A connection that CANNOT write, and can still see a live WAL.

    Two read-only modes exist and they are not interchangeable:

      mode=ro      SQLITE_OPEN_READONLY. Cannot write, and DOES read the -wal,
                   so a bot that is running right now is reported accurately.
                   Needs the -shm file to be openable.
      immutable=1  Promises the file cannot change, which makes SQLite skip
                   the -wal entirely. Works when mode=ro cannot open the
                   sidecars -- but on a LIVE database it silently omits every
                   row still sitting in the write-ahead log, which on this
                   schema is where the most recent trades are.

    So mode=ro is tried first and immutable only as a fallback, and the caller
    is told which one it got. Reporting "0 trades" because the WAL was skipped
    is exactly the false negative these scripts exist to prevent.
    """
    import sqlite3
    uri = f"file:{os.path.abspath(path)}"
    for suffix, mode in ((f"{uri}?mode=ro", "mode=ro"),
                         (f"{uri}?immutable=1", "immutable")):
        try:
            con = sqlite3.connect(suffix, uri=True)
            con.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchall()
            return con, mode
        except sqlite3.Error:
            continue
    raise sqlite3.Error(f"cannot open {path} read-only")


ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config/config.toml")
ap.add_argument("--env", default=".env")
a = ap.parse_args()

try:
    from crypto_edge.config import (FEE_TIER_CUSTOM, KRAKEN_SPOT_TAKER_BPS,
                                    load_config)
except ImportError as e:
    sys.exit(f"cannot import crypto_edge ({e}).\n"
             f"Run this from the repo root, or: PYTHONPATH=/path/to/crypto_edge "
             f"python3 show_config.py")

print(f"working directory   : {os.getcwd()}")
print(f"--config resolves to: {os.path.abspath(a.config)}"
      f"   [{'EXISTS' if os.path.exists(a.config) else 'MISSING -> DEFAULTS'}]")
print(f"--env resolves to   : {os.path.abspath(a.env)}"
      f"   [{'EXISTS' if os.path.exists(a.env) else 'missing'}]")
for k in ("CRYPTO_EDGE_EXCHANGE", "CRYPTO_EDGE_QUOTE"):
    v = os.environ.get(k)
    if v:
        print(f"env override        : {k}={v}")

cfg = load_config(a.config, a.env)
x = cfg.execution
print()
print(f"exchange            : {cfg.exchange.name}")
print(f"quote currency      : {cfg.exchange.quote}")
print(f"venue label         : {cfg.exchange_label()}")
print(f"venue set by        : {cfg.exchange_source}")
print()
print(f"execution.fee_tier  : {x.fee_tier!r}")
print(f"taker_fee_bps       : {x.taker_fee_bps}")
print(f"EFFECTIVE taker fee : {x.effective_taker_bps()} bps per side"
      f"   <-- what fills are actually charged")
print(f"fee label in reports: {x.fee_label()}")
print(f"slippage_bps        : {x.slippage_bps}")
print(f"stop_slippage_bps   : {x.stop_slippage_bps}")
if x.fee_tier == FEE_TIER_CUSTOM:
    eff = x.effective_taker_bps()
    near = min(KRAKEN_SPOT_TAKER_BPS.items(), key=lambda kv: abs(kv[1] - eff))
    print(f"  NOTE: fee_tier is 'custom', so no Kraken tier is being simulated.")
    print(f"        Kraken spot tiers on file: "
          + ", ".join(f"{k}={v:g}" for k, v in KRAKEN_SPOT_TAKER_BPS.items()))
    if abs(near[1] - eff) > 1e-9:
        print(f"        {eff:g} bps matches NO tier "
              f"(cheapest on file is {near[0]}={near[1]:g}).")
print()
db = cfg.engine.db_path
absdb = os.path.abspath(db)
print(f"engine.db_path      : {db}")
print(f"  resolves to       : {absdb}")
if not os.path.exists(absdb):
    print(f"  !! THAT FILE DOES NOT EXIST. Running a bot command from here")
    print(f"     would CREATE an empty one. Find the real ledger first.")
else:
    con, mode = open_readonly(absdb)
    rows = list(con.execute("SELECT strategy, COUNT(*) FROM trades"
                            " GROUP BY strategy"))
    con.close()
    for s, n in rows:
        print(f"  closed trades     : {s:<26} {n}")
    if not rows:
        print(f"  closed trades     : NONE in this file")
    if mode == "immutable":
        print(f"  !! read as immutable -- a LIVE write-ahead log was NOT read.")
        print(f"     Stop the bot and re-run for an accurate count.")
