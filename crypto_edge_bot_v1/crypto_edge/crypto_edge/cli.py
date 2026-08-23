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
"""
from __future__ import annotations

import argparse
import csv
import json
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
                                error_cooldown_s=cfg.telegram.error_cooldown_s)
    feed = None
    if need_feed:
        from .data.ccxt_feed import CCXTFeed
        feed = CCXTFeed(cfg.exchange.name, cfg.exchange.quote,
                        cfg.exchange.rate_limit_ms)
    return cfg, repo, feed, notifier


# ------------------------------------------------------------------ commands
def cmd_selfcheck(args) -> int:
    from .selfcheck import run_selfcheck
    offline = args.offline
    cfg, repo, feed, notifier = _bootstrap(args, need_feed=not offline)
    rep = run_selfcheck(cfg, repo, feed, notifier, check_network=not offline)
    print(rep.render())
    return 0 if rep.passed else 1


def cmd_start(args) -> int:
    from .engine import TradingEngine
    from .selfcheck import run_selfcheck
    cfg, repo, feed, notifier = _bootstrap(args)

    rep = run_selfcheck(cfg, repo, feed, notifier, check_network=True)
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
    perf = PerformanceCalculator(repo)
    rep = perf.report().as_dict()
    a, r, t = rep["account"], rep["risk"], rep["trading"]
    acct = repo.get_account()
    print("=" * 62)
    print(f"  CRYPTO EDGE — {cfg.safety.mode} MODE ({cfg.exchange.name})")
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
    _, repo, _, _ = _bootstrap(args, need_feed=False)
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
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
