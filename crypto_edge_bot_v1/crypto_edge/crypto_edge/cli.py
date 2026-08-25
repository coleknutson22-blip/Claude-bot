"""Operator commands. You should not need to read any Python to run this bot.

    python -m crypto_edge.cli selfcheck
    python -m crypto_edge.cli start
    python -m crypto_edge.cli status
    python -m crypto_edge.cli positions
    python -m crypto_edge.cli performance
    python -m crypto_edge.cli export --out trades.csv
    python -m crypto_edge.cli research
    python -m crypto_edge.cli resume
    python -m crypto_edge.cli test
    python -m crypto_edge.cli verify-live --cycle
    python -m crypto_edge.cli verify-restart
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
from pathlib import Path

from .config import Config, load_config
from .logging_setup import log_event, setup_logging
from .notify.telegram import TelegramNotifier
from .performance import PerformanceCalculator
from .storage import db
from .storage.repo import Repo
from .timeutils import fmt_duration, iso, now_ms


def _bootstrap(args, need_feed: bool = True):
    cfg = load_config(args.config, args.env)
    setup_logging(cfg.engine.log_dir, args.log_level)
    conn = db.connect(cfg.engine.db_path)
    db.init_db(conn)
    repo = Repo(conn)
    notifier = TelegramNotifier(cfg.telegram_token, cfg.telegram_chat_id, repo,
                                enabled=cfg.telegram.enabled,
                                timeout_s=cfg.telegram.timeout_s,
                                max_retries=cfg.telegram.max_retries,
                                error_cooldown_s=cfg.telegram.error_cooldown_s,
                                outbox_lease_s=cfg.telegram.outbox_lease_s,
                                outbox_max_attempts=cfg.telegram.outbox_max_attempts,
                                outbox_flush_limit=cfg.telegram.outbox_flush_limit)
    feed = None
    if need_feed:
        from .data.ccxt_feed import CCXTFeed
        feed = CCXTFeed(cfg.exchange.name, cfg.exchange.quote,
                        cfg.exchange.rate_limit_ms,
                        quote_ts_fallback=cfg.execution.quote_ts_fallback)
    return cfg, repo, feed, notifier


def _warn_if_venue_changed(cfg: Config, repo: Repo) -> None:
    """Say so, loudly, if this database was last used with a different venue."""
    try:
        changed, previous = repo.record_exchange(cfg.exchange_label())
    except Exception:
        return
    if changed:
        print(f"  !! THIS DATABASE WAS LAST USED WITH {previous}, NOW "
              f"{cfg.exchange_label()}")
        print(f"  !! Its positions and history came from {previous}.")


def _broad_service(cfg: Config, repo: Repo):
    """The broad (market-cap) asset universe service, built from config."""
    from .data.broad_universe import BroadUniverseService
    from .engine import _build_broad_provider
    return BroadUniverseService(
        repo, _build_broad_provider(cfg), limit=cfg.universe.broad_limit,
        min_assets=cfg.universe.broad_min_assets,
        refresh_hours=cfg.universe.broad_refresh_hours,
        max_cache_age_hours=cfg.universe.broad_max_cache_age_hours,
        collision_scan_limit=cfg.universe.broad_collision_scan_limit,
        symbol_overrides=cfg.universe.broad_symbol_overrides)


# ------------------------------------------------------------------ commands
def cmd_selfcheck(args) -> int:
    from .selfcheck import run_selfcheck
    offline = args.offline
    cfg, repo, feed, notifier = _bootstrap(args, need_feed=not offline)
    broad = _broad_service(cfg, repo) if not offline else None
    rep = run_selfcheck(cfg, repo, feed, notifier, check_network=not offline,
                        broad_service=broad)
    print(rep.render())
    return 0 if rep.passed else 1


def cmd_start(args) -> int:
    from .engine import TradingEngine
    from .selfcheck import run_selfcheck
    cfg, repo, feed, notifier = _bootstrap(args)

    rep = run_selfcheck(cfg, repo, feed, notifier, check_network=True,
                        broad_service=_broad_service(cfg, repo))
    print(rep.render())
    if not rep.passed:
        print("\nRefusing to start. Fix the FAIL items above.", file=sys.stderr)
        return 1

    engine = TradingEngine(cfg, repo, feed, notifier)
    engine.refresh_universe(force=True)
    engine.announce_start()

    def _graceful(signum, frame):
        print("\nStopping after current cycle...")
        log_event("app", "INFO", "shutdown signal received", signal=signum)
        engine.stop()

    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)

    print(f"\nPAPER TRADING ACTIVE — polling every {cfg.engine.poll_seconds}s. "
          f"Ctrl-C to stop safely.")
    engine.run(max_cycles=args.max_cycles)
    print("Stopped. State persisted; restart resumes exactly where it left off.")
    return 0


def cmd_status(args) -> int:
    cfg, repo, _, _ = _bootstrap(args, need_feed=False)
    # A database that has never run the engine has no account row yet; reporting
    # a clean starting balance is far more useful to an operator than a
    # traceback about missing state.
    repo.ensure_account(cfg.execution.starting_equity)
    perf = PerformanceCalculator(repo)
    rep = perf.report().as_dict()
    a, r, t = rep["account"], rep["risk"], rep["trading"]
    acct = repo.get_account()
    print("=" * 62)
    print(f"  CRYPTO EDGE — {cfg.safety.mode} MODE ({cfg.exchange_label()})")
    print(f"  Exchange from: {cfg.exchange_source}")
    _warn_if_venue_changed(cfg, repo)
    print("=" * 62)
    print(f"  Equity            ${a['current_equity']:>14,.2f}")
    print(f"  Cash              ${a['cash']:>14,.2f}")
    print(f"  Deployed          ${a['capital_deployed']:>14,.2f}")
    print(f"  Total P&L         ${a['total_pnl']:>14,.2f}  ({a['account_return_pct']:+.2f}%)")
    print(f"  Realized          ${a['realized_pnl']:>14,.2f}")
    print(f"  Unrealized        ${a['unrealized_pnl']:>14,.2f}")
    print(f"  Fees              ${a['total_fees']:>14,.2f}")
    print(f"  Slippage          ${a['estimated_slippage_cost']:>14,.2f}")
    print("-" * 62)
    print(f"  Open positions    {r['open_positions']:>15}")
    print(f"  Exposure          {r['current_exposure_pct']:>14.1f}%")
    print(f"  Drawdown          {r['current_drawdown_pct']:>14.2f}%  (max {r['max_drawdown_pct']:.2f}%)")
    print(f"  Closed trades     {t['closed_trades']:>15}")
    if t["closed_trades"]:
        print(f"  Win rate          {t['win_rate_pct']:>14.1f}%")
    if int(acct["halted"]):
        print("-" * 62)
        print(f"  ⚠ HALTED: {acct['halt_reason']}")
    print("=" * 62)
    if rep["open_positions"]:
        print("\nOPEN POSITIONS")
        for p in rep["open_positions"]:
            print(f"  {p['symbol']:<14} entry ${p['entry']:<12,.6g} now ${p['current']:<12,.6g} "
                  f"P&L ${p['unrealized']:>+9,.2f} ({p['unrealized_pct']:+.2f}%) "
                  f"stop ${p['stop']:,.6g}  held {p['held']}")
    print(f"\n  Sample: {rep['sample']['closed_trades']} trades / "
          f"{rep['sample']['trading_days']} days — {rep['sample']['note']}")
    return 0


def cmd_positions(args) -> int:
    cfg, repo, _, _ = _bootstrap(args, need_feed=False)
    print(f"# exchange: {cfg.exchange_label()} (from {cfg.exchange_source})")
    positions = repo.get_positions()
    if not positions:
        print("No open positions.")
        return 0
    for p in positions:
        print(f"{p.symbol:<14} qty={p.qty:<14,.8g} entry=${p.entry_fill_price:<12,.6g} "
              f"stop=${p.current_stop:<12,.6g} risk=${p.risk_amount:,.2f} "
              f"score={p.signal_score:.1f} opened={iso(p.entry_ms)}")
    return 0


def cmd_performance(args) -> int:
    _, repo, _, _ = _bootstrap(args, need_feed=False)
    perf = PerformanceCalculator(repo)
    rep = perf.report().as_dict()
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
        return 0
    for section in ("account", "trading", "risk", "advanced"):
        print(f"\n{section.upper()}")
        for k, v in rep[section].items():
            print(f"  {k:<34} {v}")
    print(f"\nSAMPLE\n  {rep['sample']}")
    if args.categories:
        print("\nBY CATEGORY (buckets below sample threshold are flagged)")
        for name, rows in perf.by_category().items():
            shown = [r for r in rows if r["sufficient_sample"]]
            print(f"\n  {name}: {len(rows)} bucket(s), {len(shown)} with sufficient sample")
            for r in rows[:10]:
                flag = "" if r["sufficient_sample"] else "  [SAMPLE TOO SMALL]"
                print(f"    {r['bucket']:<24} n={r['n']:<4} net=${r['net_pnl']:>+9,.2f} "
                      f"win={r['win_rate_pct']:.0f}%{flag}")
    return 0


def cmd_export(args) -> int:
    _, repo, _, _ = _bootstrap(args, need_feed=False)
    trades = repo.get_trades()
    out = Path(args.out)
    if not trades:
        print("No closed trades to export.")
        return 0
    fields = [k for k in trades[0] if k != "journal"]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields + ["journal"])
        w.writeheader()
        for t in trades:
            row = {k: t[k] for k in fields}
            row["journal"] = json.dumps(t["journal"], default=str)
            w.writerow(row)
    print(f"Exported {len(trades)} trades to {out}")
    return 0


def cmd_research(args) -> int:
    from .research.counterfactual import CounterfactualTracker
    cfg, repo, _, _ = _bootstrap(args, need_feed=False)
    obs = repo.get_observations()
    counts: dict[str, int] = {}
    for o in obs:
        counts[o["decision"]] = counts.get(o["decision"], 0) + 1
    print(f"Observations recorded: {len(obs)}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22} {v}")
    reasons: dict[str, int] = {}
    for o in obs:
        if o["reject_reason"]:
            key = o["reject_reason"].split("(")[0].strip()[:60]
            reasons[key] = reasons.get(key, 0) + 1
    if reasons:
        print("\nTop rejection reasons")
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {v:>6}  {k}")
    tracker = CounterfactualTracker(repo, cfg.engine.counterfactual_horizons_h)
    rows = tracker.filter_report()
    if rows:
        print("\nHYPOTHETICAL / NOT EXECUTED — rejected-signal outcomes")
        for r in rows[:20]:
            flag = "" if r["sufficient_sample"] else "  [SAMPLE TOO SMALL]"
            print(f"  {r['reason'][:46]:<46} {r['h']:>4}h  n={r['n']:<4} "
                  f"avg={r['avg_ret']:+.2f}%{flag}")
    return 0


def cmd_resume(args) -> int:
    """Clear a halt after you have reviewed why it tripped."""
    _, repo, _, _ = _bootstrap(args, need_feed=False)
    acct = repo.get_account()
    if not int(acct["halted"]):
        print("Bot is not halted.")
        return 0
    print(f"Current halt: {acct['halt_reason']}")
    if not args.yes:
        print("Re-run with --yes to clear it.")
        return 1
    repo.set_halt(False, "")
    print("Halt cleared. Restart the trader to resume entries.")
    return 0


def cmd_verify_live(args) -> int:
    """Exercise everything that cannot be verified offline. See verify_live.py."""
    from .execution.paper_broker import PaperBroker
    from .verify_live import (VerifyReport, verify_cycle, verify_exchange,
                              verify_pending_first, verify_quotes,
                              verify_telegram, verify_universe)

    cfg, repo, feed, notifier = _bootstrap(args, need_feed=True)
    rep = VerifyReport()
    print("=" * 72)
    print("CRYPTO EDGE -- LIVE NETWORK VERIFICATION")
    print("=" * 72)
    print("Read-only with respect to trading. No orders exist in this codebase.")

    ex = verify_exchange(cfg, feed, rep)

    broker = PaperBroker(cfg.execution.taker_fee_bps, cfg.execution.slippage_bps,
                         cfg.execution.stop_slippage_bps,
                         cfg.execution.use_book_spread,
                         cfg.execution.max_spread_bps_entry)
    verify_quotes(cfg, feed, broker, args.quote_samples, args.quote_interval, rep)

    if not args.skip_universe:
        verify_universe(cfg, repo, _broad_service(cfg, repo),
                        ex.get("markets", {}), ex.get("tickers", {}), rep)

    if not args.skip_telegram:
        verify_telegram(cfg, repo, notifier, rep)
        verify_pending_first(notifier, repo, rep)

    if args.cycle:
        verify_cycle(cfg, repo, feed, notifier, rep)

    print(rep.render_summary())
    return 0 if rep.passed else 1


def cmd_verify_restart(args) -> int:
    """Prove that stopping and restarting loses nothing.

    Reads the persisted state twice through two independent connections, which
    is what a real restart does, and diffs what matters.
    """
    import sqlite3

    cfg, repo, _, _ = _bootstrap(args, need_feed=False)
    acct = repo.get_account()
    positions = repo.get_positions()
    broad = repo.latest_broad_universe()
    outbox = repo.telegram_outbox_counts()
    candles = repo.conn.execute(
        "SELECT COUNT(*) AS n FROM processed_candles").fetchone()["n"]
    obs = repo.conn.execute("SELECT COUNT(*) AS n FROM observations").fetchone()["n"]
    repo.conn.close()

    conn2 = db.connect(cfg.engine.db_path)
    db.init_db(conn2)
    repo2 = Repo(conn2)
    acct2 = repo2.get_account()
    positions2 = repo2.get_positions()
    broad2 = repo2.latest_broad_universe()
    outbox2 = repo2.telegram_outbox_counts()
    candles2 = repo2.conn.execute(
        "SELECT COUNT(*) AS n FROM processed_candles").fetchone()["n"]
    obs2 = repo2.conn.execute("SELECT COUNT(*) AS n FROM observations").fetchone()["n"]

    checks = [
        ("account row restored", acct2["starting_equity"] == acct["starting_equity"]
         and abs(float(acct2["cash"]) - float(acct["cash"])) < 1e-9,
         f"cash ${float(acct2['cash']):,.2f} start ${float(acct2['starting_equity']):,.2f}"),
        ("peak equity restored",
         abs(float(acct2["peak_equity"]) - float(acct["peak_equity"])) < 1e-9,
         f"${float(acct2['peak_equity']):,.2f}"),
        ("halt state restored", int(acct2["halted"]) == int(acct["halted"]),
         f"halted={bool(int(acct2['halted']))} {acct2['halt_reason']}"),
        ("daily anchor restored", acct2["daily_date"] == acct["daily_date"],
         f"{acct2['daily_date']} start ${float(acct2['daily_start_equity']):,.2f}"),
        ("open positions restored", len(positions2) == len(positions),
         f"{len(positions2)} position(s)"
         + ("" if positions2 else " (none open -- restore path exercised by tests)")),
        ("universe cache survives", (broad2 is not None) == (broad is not None)
         and (broad is None or broad2["content_hash"] == broad["content_hash"]),
         f"{broad2['n_assets']} assets, hash {broad2['content_hash'][:12]}, "
         f"source {broad2['source']}" if broad2 else "no cache present"),
        ("telegram outbox survives", outbox2 == outbox, f"{outbox2 or 'empty'}"),
        ("processed candles survive", candles2 == candles,
         f"{candles2} candle(s) claimed -- these prevent duplicate entries"),
        ("observations survive", obs2 == obs, f"{obs2} journal rows"),
    ]
    print("=" * 72)
    print("RESTART VERIFICATION -- state re-read through a fresh connection")
    print("=" * 72)
    ok_all = True
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:32} {detail}")
        ok_all &= ok
    for p in positions2:
        print(f"         {p.symbol} qty={p.qty:g} stop=${p.current_stop:,.6g} "
              f"risk=${p.risk_amount:,.2f}")
    print("=" * 72)
    print("RESTART RESULT: " + ("ALL STATE RESTORED" if ok_all else "MISMATCH ABOVE"))
    return 0 if ok_all else 1


def cmd_test(args) -> int:
    import subprocess
    root = Path(__file__).resolve().parent.parent
    return subprocess.call([sys.executable, "-m", "unittest", "discover",
                            "-s", "tests", "-v"], cwd=root)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="crypto_edge", description="Crypto Edge paper trader")
    p.add_argument("--config", default="config/config.toml")
    p.add_argument("--env", default=".env")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--exchange", default=None,
                   help="override the exchange for this run (e.g. kraken, "
                        "coinbase, kucoin, okx) without editing any file")
    p.add_argument("--quote", default=None,
                   help="override the quote currency (e.g. USD instead of USDT)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selfcheck", help="run startup checks and exit")
    s.add_argument("--offline", action="store_true",
                   help="skip all network checks")
    s.set_defaults(func=cmd_selfcheck)

    s = sub.add_parser("start", help="start the paper trader")
    s.add_argument("--max-cycles", type=int, default=None)
    s.set_defaults(func=cmd_start)

    sub.add_parser("status", help="account snapshot").set_defaults(func=cmd_status)
    sub.add_parser("positions", help="list open positions").set_defaults(func=cmd_positions)

    s = sub.add_parser("performance", help="full performance report")
    s.add_argument("--json", action="store_true")
    s.add_argument("--categories", action="store_true")
    s.set_defaults(func=cmd_performance)

    s = sub.add_parser("export", help="export closed trades to CSV")
    s.add_argument("--out", default="trades.csv")
    s.set_defaults(func=cmd_export)

    sub.add_parser("research", help="research database summary").set_defaults(func=cmd_research)

    s = sub.add_parser("resume", help="clear a circuit-breaker halt")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(func=cmd_resume)

    sub.add_parser("test", help="run the automated test suite").set_defaults(func=cmd_test)

    s = sub.add_parser("verify-live",
                       help="exercise exchange, universe and Telegram against "
                            "the real network")
    s.add_argument("--quote-samples", type=int, default=10,
                   help="ticker samples for quote-age calibration (default 10)")
    s.add_argument("--quote-interval", type=float, default=2.0,
                   help="seconds between quote samples (default 2)")
    s.add_argument("--cycle", action="store_true",
                   help="also run one complete engine cycle on live data")
    s.add_argument("--skip-telegram", action="store_true")
    s.add_argument("--skip-universe", action="store_true")
    s.set_defaults(func=cmd_verify_live)

    sub.add_parser("verify-restart",
                   help="prove persisted state survives a restart"
                   ).set_defaults(func=cmd_verify_restart)
    return p


def _force_utf8_output() -> None:
    """Make stdout/stderr survive non-ASCII on Windows.

    Python uses the Unicode console API when writing to a real terminal, but
    falls back to the system code page (cp1252) the moment output is redirected
    to a file or piped. A single emoji in a status line then raises
    UnicodeEncodeError and takes the process down -- which would happen to
    anyone capturing output to send to support. Reconfiguring with a
    replacement error handler makes that impossible.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass          # very old Python or an unusual stream; not fatal


def main(argv=None) -> int:
    _force_utf8_output()
    args = build_parser().parse_args(argv)
    if getattr(args, "exchange", None):
        os.environ["CRYPTO_EDGE_EXCHANGE"] = args.exchange
    if getattr(args, "quote", None):
        os.environ["CRYPTO_EDGE_QUOTE"] = args.quote
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
