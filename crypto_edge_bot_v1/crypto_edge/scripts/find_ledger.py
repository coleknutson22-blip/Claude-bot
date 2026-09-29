#!/usr/bin/env python3
"""Find every Crypto Edge ledger on this machine and count what is in it.

Stdlib only -- no numpy, no ccxt -- so it runs even where the bot's
dependencies are not installed. READ ONLY: every database is opened through
`open_readonly` below, which cannot create, migrate or modify anything.
"""
import os
import signal
import sqlite3
import sys

# Piping this into `head` is the normal way to read it, and a raw
# BrokenPipeError traceback on an otherwise successful scan reads like a
# failure. Restore the default so the process just ends.
try:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
except (AttributeError, ValueError):      # not POSIX, or not the main thread
    pass

SKIP = {".git", "node_modules", "__pycache__", ".venv", "venv",
        "Library", ".cache", ".Trash", "site-packages"}


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


def candidates(roots):
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
            dirnames[:] = [d for d in dirnames if d not in SKIP
                           and not d.startswith(".")]
            for fn in filenames:
                if fn.endswith((".db", ".sqlite", ".sqlite3")):
                    yield os.path.join(dirpath, fn)


def inspect(path):
    """Row counts per strategy. Immutable URI: never writes, never migrates."""
    try:
        con, mode = open_readonly(path)
        con.row_factory = sqlite3.Row
        names = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "trades" not in names:
            return None
        out = {"path": os.path.abspath(path), "read_mode": mode,
               "bytes": os.path.getsize(path), "strategies": {}}
        for r in con.execute(
                "SELECT strategy, COUNT(*) n, MIN(exit_ms) lo, MAX(exit_ms) hi"
                " FROM trades GROUP BY strategy"):
            out["strategies"][r["strategy"]] = {
                "trades": r["n"], "first_exit_ms": r["lo"],
                "last_exit_ms": r["hi"]}
        if "observations" in names:
            for r in con.execute("SELECT strategy, COUNT(*) n FROM observations"
                                 " GROUP BY strategy"):
                out["strategies"].setdefault(r["strategy"], {"trades": 0})
                out["strategies"][r["strategy"]]["observations"] = r["n"]
        con.close()
        return out
    except (sqlite3.Error, OSError):
        return None


def ms(v):
    """Epoch milliseconds as a UTC wall-clock string."""
    import datetime as dt
    if not v:
        return "-"
    return dt.datetime.fromtimestamp(
        v / 1000, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")


def main(roots):
    found = []
    for path in candidates(roots):
        info = inspect(path)
        if info and info["strategies"]:
            found.append(info)

    if not found:
        print("No Crypto Edge ledger found under:", ", ".join(roots))
        print("Widen the search, e.g.:  python3 find_ledger.py /")
        return 1

    found.sort(key=lambda f: -sum(s.get("trades", 0)
                                  for s in f["strategies"].values()))
    print(f"{len(found)} ledger(s) with a trades table, most trades first\n")
    for f in found:
        total = sum(s.get("trades", 0) for s in f["strategies"].values())
        print(f"{f['path']}")
        warn = ("   [read as immutable: a LIVE write-ahead log was NOT read, "
                "so stop the bot and re-run]"
                if f["read_mode"] == "immutable" else "")
        print(f"    {f['bytes']:,} bytes, {total} closed trade(s) in total{warn}")
        for name, st in sorted(f["strategies"].items()):
            print(f"    {name:<26} {st.get('trades', 0):>4} trades  "
                  f"{st.get('observations', 0):>6} obs   "
                  f"{ms(st.get('first_exit_ms'))} -> {ms(st.get('last_exit_ms'))}")
        print()
    return 0


# Guarded so the module can be imported (by the tests, and by anything else
# that wants `open_readonly`) without kicking off a filesystem scan.
if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or [os.path.expanduser("~")]))
