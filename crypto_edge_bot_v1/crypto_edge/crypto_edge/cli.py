"""Operator commands. You should not need to read any Python to run this bot.

    python -m crypto_edge.cli selfcheck
    python -m crypto_edge.cli start
    python -m crypto_edge.cli status
    python -m crypto_edge.cli positions
    python -m crypto_edge.cli performance
    python -m crypto_edge.cli export --out trades.csv
    python -m crypto_edge.cli performance --aggressive
    python -m crypto_edge.cli research
    python -m crypto_edge.cli research --aggressive
    python -m crypto_edge.cli research --aggressive --excursions
    python -m crypto_edge.cli research --aggressive --policy-sim
    python -m crypto_edge.cli resume
    python -m crypto_edge.cli test
    python -m crypto_edge.cli verify-live --cycle
    python -m crypto_edge.cli verify-restart
    python -m crypto_edge.cli diagnose
    python -m crypto_edge.cli preflight

Which strategies TRADE is chosen at the command line, not in a config file:

    python -m crypto_edge.cli --strategies b start      # Strategy B only

`--strategies` gates NEW ENTRIES only. A strategy that is off keeps managing
what it already holds, right through to its exits.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
from pathlib import Path

from .config import Config, load_config
from .data.feed import DataUnavailable
from .logging_setup import log_event, setup_logging
from .notify.telegram import TelegramNotifier
from .performance import PerformanceCalculator
from .storage import db
from .storage.repo import Repo
from .timeutils import fmt_duration, iso, now_ms


def _bootstrap(args, need_feed: bool = True):
    cfg = load_config(args.config, args.env)
    if getattr(args, "strategies", None):
        cfg.apply_runtime_mode(args.strategies)
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
                        quote_ts_fallback=cfg.execution.quote_ts_fallback,
                        page_limit=cfg.exchange.ohlcv_limit,
                        close_buffer_ms=cfg.safety.candle_close_buffer_s * 1000,
                        cache_bars=cfg.exchange.ohlcv_cache_bars)
    return cfg, repo, feed, notifier


def _strategy_arg(args, cfg: Config) -> str:
    """Which ledger a read-only command is reporting on.

    Sub-accounts are per strategy, so every report has to name one. The
    configured strategy is the default, which keeps every existing invocation
    meaning exactly what it meant before.
    """
    return getattr(args, "strategy", None) or cfg.strategy.name


def _print_runtime_mode(cfg: Config, repo: Repo) -> None:
    """Say plainly which strategies trade this run, and what the others do.

    "Disabled" is the word most likely to be misread here, so the line spells
    out the consequence instead: a strategy that is off opens nothing, and
    still manages what it holds. An operator should never have to infer from a
    flag name whether their open stops are still being watched.
    """
    print("=" * 62)
    print(f"  RUNTIME MODE: {cfg.runtime_mode()}   ({cfg.exchange_label()})")
    for name, on in ((cfg.strategy.name, cfg.strategy.enabled),
                     (cfg.aggressive.name, cfg.aggressive.enabled)):
        try:
            held = len(repo.get_positions(name))
        except Exception:
            held = 0
        if on:
            state = "ENTRIES ON"
        elif held:
            state = f"entries OFF — still managing {held} open position(s)"
        else:
            state = "entries OFF — holds nothing"
        equity = cfg.starting_equity_for(name)
        print(f"    {name:<24} {state:<46} (${equity:,.0f} sub-account)")
    print("=" * 62)


def _warn_if_venue_changed(cfg: Config, repo: Repo) -> None:
    """Say so, loudly, if this database was last used with a different venue."""
    try:
        changed, previous = repo.record_exchange(cfg.exchange_label())
    except Exception:
        return
    if changed:
        prev_venue, _, prev_quote = previous.partition("/")
        what = "QUOTE CURRENCY" if prev_venue == cfg.exchange.name else "EXCHANGE"
        print(f"  !! {what} CHANGED: last used with {previous}, now "
              f"{cfg.exchange_label()}")
        print(f"  !! Its positions and history came from {previous}.")
        if prev_quote and prev_quote != cfg.quote_currency:
            print(f"  !! Equity and fills below are denominated in {prev_quote}, "
                  f"NOT {cfg.quote_currency}.")


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

    mode = cfg.runtime_mode()
    if mode == "none":
        print("No strategy is enabled — nothing would trade. Use --strategies "
              "a|b|both.", file=sys.stderr)
        return 1
    _print_runtime_mode(cfg, repo)

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

    print(f"\nPAPER TRADING ACTIVE — polling every {cfg.engine.poll_seconds}s "
          f"(minimum {cfg.engine.min_pause_seconds:.0f}s between cycles; a cycle "
          f"slower than the interval slows the cadence, it never overlaps). "
          f"Ctrl-C to stop safely.")
    engine.run(max_cycles=args.max_cycles)
    print("Stopped. State persisted; restart resumes exactly where it left off.")
    return 0


def cmd_status(args) -> int:
    cfg, repo, _, _ = _bootstrap(args, need_feed=False)
    # A database that has never run the engine has no account row yet; reporting
    # a clean starting balance is far more useful to an operator than a
    # traceback about missing state.
    strategy = _strategy_arg(args, cfg)
    repo.ensure_account(strategy, cfg.starting_equity_for(strategy))
    perf = PerformanceCalculator(repo, strategy)
    rep = perf.report().as_dict()
    a, r, t = rep["account"], rep["risk"], rep["trading"]
    acct = repo.get_account(strategy)
    print("=" * 62)
    print(f"  CRYPTO EDGE — {cfg.safety.mode} MODE ({cfg.exchange_label()})")
    print(f"  Strategy: {strategy}")
    print(f"  Exchange from: {cfg.exchange_source}")
    _warn_if_venue_changed(cfg, repo)
    print("=" * 62)
    print(f"  Equity            ${a['current_equity']:>14,.2f}")
    print(f"  Cash              ${a['cash']:>14,.2f}")
    print(f"  Deployed          ${a['capital_deployed']:>14,.2f}")
    print(f"  Total P&L         ${a['total_pnl']:>14,.2f}  ({a['account_return_pct']:+.2f}%)")
    print(f"  Realized          ${a['realized_pnl']:>14,.2f}")
    print(f"  Unrealized        ${a['unrealized_pnl']:>14,.2f}")
    print(f"  Fees              ${a['total_fees']:>14,.2f}"
          f"   [{cfg.execution.fee_label()}]")
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
    positions = repo.get_positions(_strategy_arg(args, cfg))
    if not positions:
        print("No open positions.")
        return 0
    for p in positions:
        print(f"{p.symbol:<14} qty={p.qty:<14,.8g} entry=${p.entry_fill_price:<12,.6g} "
              f"stop=${p.current_stop:<12,.6g} risk=${p.risk_amount:,.2f} "
              f"score={p.signal_score:.1f} opened={iso(p.entry_ms)}")
    return 0


def cmd_performance(args) -> int:
    cfg, repo, _, _ = _bootstrap(args, need_feed=False)
    strategy = cfg.aggressive.name if getattr(args, "aggressive", False) \
        else _strategy_arg(args, cfg)
    perf = PerformanceCalculator(repo, strategy)
    rep = perf.report().as_dict()
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
        return 0
    print(f"\nSTRATEGY  {strategy}   ({cfg.exchange_label()})")
    for section in ("account", "trading", "risk", "advanced"):
        print(f"\n{section.upper()}")
        for k, v in rep[section].items():
            print(f"  {k:<34} {v}")
    _print_sides(rep["sides"])
    print(f"\nSAMPLE\n  {rep['sample']}")
    if args.categories or getattr(args, "aggressive", False):
        cats = perf.by_category()
        # The aggressive view leads with the three slices the forward test is
        # actually asking about, then everything else.
        order = (["conf_bucket", "ladder_slot", "binding_constraint", "side",
                  "exit_reason"] if getattr(args, "aggressive", False) else [])
        names = order + [k for k in cats if k not in order]
        print("\nBY CATEGORY (buckets below sample threshold are flagged)")
        for name in names:
            rows = cats.get(name) or []
            if not rows:
                continue
            shown = [r for r in rows if r["sufficient_sample"]]
            print(f"\n  {name}: {len(rows)} bucket(s), "
                  f"{len(shown)} with sufficient sample")
            for r in rows[:10]:
                flag = "" if r["sufficient_sample"] else "  [SAMPLE TOO SMALL]"
                print(f"    {r['bucket']:<24} n={r['n']:<4} "
                      f"net=${r['net_pnl']:>+9,.2f} "
                      f"win={r['win_rate_pct']:.0f}%{flag}")
    return 0


def _print_sides(sides: dict) -> None:
    """Long and short side by side. Blank rows are still printed.

    A side with zero trades is a RESULT during a forward test -- "the strategy
    took no shorts in three weeks" is exactly the kind of thing that goes
    unnoticed when the empty half is simply omitted.
    """
    if not sides:
        return
    print("\nBY SIDE")
    cols = ["trades", "win_rate_pct", "average_winner", "average_loser",
            "expectancy_per_trade", "profit_factor", "gross_pnl", "net_pnl",
            "fees", "slippage", "financing"]
    print(f"  {'':<22}{'LONG':>14}{'SHORT':>14}")
    for c in cols:
        lo, sh = sides.get("long", {}).get(c, 0), sides.get("short", {}).get(c, 0)
        fmt_one = (lambda v: f"{v:>14,.0f}") if c == "trades" else \
                  (lambda v: f"{v:>14,.2f}" if v != float("inf") else f"{'inf':>14}")
        print(f"  {c:<22}{fmt_one(lo)}{fmt_one(sh)}")


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
    for flag, fn in (("exit_quality", _research_exit_quality),
                     ("fee_scenarios", _research_fee_scenarios),
                     ("atr_economics", _research_atr_economics)):
        if getattr(args, flag, False):
            return fn(cfg, repo,
                      cfg.aggressive.name if getattr(args, "aggressive", False)
                      else _strategy_arg(args, cfg), args)
    if getattr(args, "short_funnel", False):
        return _research_short_funnel(
            cfg, repo, cfg.aggressive.name if getattr(args, "aggressive", False)
            else _strategy_arg(args, cfg), args)
    if getattr(args, "policy_sim", False):
        return _research_policy_sim(
            cfg, repo, cfg.aggressive.name if getattr(args, "aggressive", False)
            else _strategy_arg(args, cfg), args)
    if getattr(args, "excursions", False):
        return _research_excursions(
            cfg, repo, cfg.aggressive.name if getattr(args, "aggressive", False)
            else _strategy_arg(args, cfg), args)
    if getattr(args, "aggressive", False) or getattr(args, "strategy", None):
        return _research_forward_test(
            cfg, repo, cfg.aggressive.name if getattr(args, "aggressive", False)
            else _strategy_arg(args, cfg), args)
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


def _research_forward_test(cfg, repo, strategy: str, args) -> int:
    """The forward-test view of one strategy's journal."""
    from .research.forward_test import ForwardTestReport
    r = ForwardTestReport(repo, strategy, min_sample=args.min_sample)
    if getattr(args, "json", False):
        print(json.dumps({
            "strategy": strategy, "decisions": r.decisions(),
            "by_side": r.by_side(),
            "rejections": [vars(x) for x in r.rejection_counts()],
            "score_buckets": [vars(x) for x in r.score_buckets()],
            "gates": r.gate_sensitivity(),
            "confidence_buckets": [vars(x) for x in r.confidence_buckets()],
        }, indent=2, default=str))
        return 0

    def flag(n):
        return "" if r.sufficient(n) else "  [SAMPLE TOO SMALL]"

    print("=" * 74)
    print(f"  RESEARCH — {strategy}   ({cfg.exchange_label()})")
    print(f"  {len(r.obs)} observations, {len(r.trades)} closed trades")
    print("=" * 74)

    print("\nDECISIONS")
    for k, v in sorted(r.decisions().items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22} {v:>6}")

    print("\nLONG vs SHORT")
    print(f"  {'':<12}{'evaluated':>11}{'entered':>9}{'rejected':>10}"
          f"{'closed':>8}{'net P&L':>12}{'financing':>11}")
    for side, d in r.by_side().items():
        print(f"  {side:<12}{d['evaluated']:>11}{d['entered']:>9}"
              f"{d['rejected']:>10}{d['closed_trades']:>8}"
              f"{d['net_pnl']:>+12,.2f}{d['financing']:>11,.2f}")

    print("\nFILTER REJECTION COUNTS")
    print("  (avg move is HYPOTHETICAL / NOT EXECUTED, signed by signal side)")
    for row in r.rejection_counts()[:20]:
        seen = row.extra["with_outcome"]
        tail = (f"  avg move {row.avg:+.2f}% over {seen}{flag(seen)}"
                if seen else "  (no outcomes yet)")
        print(f"  {row.n:>6}  {row.bucket[:44]:<44}{tail}")

    print("\nIS EACH GATE EARNING ITS PLACE?")
    print("  A gate is doing its job when what it rejected went on to move")
    print("  BADLY in the signal's direction. Positive average = worth a look.")
    for g in r.gate_sensitivity():
        print(f"\n  {g['gate']}  ({g['config_key']})")
        print(f"    {g['question']}")
        med = g["measured_median"]
        rng = (f"measured {g['measured_min']:.3g} .. {g['measured_max']:.3g} "
               f"(median {med:.3g})" if med is not None else "no values recorded")
        print(f"    rejected {g['rejected']}, outcomes {g['with_outcome']}; {rng}")
        if g["with_outcome"]:
            print(f"    avg move {g['avg_return_pct']:+.2f}%, "
                  f"{g['win_rate_pct']:.0f}% moved the right way"
                  f"{flag(g['with_outcome'])}")

    print("\nSETUP SCORE BUCKETS  (does the score predict anything?)")
    for row in r.score_buckets():
        seen = row.extra["with_outcome"]
        print(f"  score {row.bucket:<10} n={row.n:<5} entered={row.extra['entered']:<4} "
              f"avg move {row.avg:+.2f}% over {seen}{flag(seen)}")

    print("\nCONFIDENCE BUCKETS  (realised, from closed trades)")
    rows = r.confidence_buckets()
    if not rows:
        print("  (no closed Strategy B trades yet)")
    for row in rows:
        print(f"  {row.bucket:<10} n={row.n:<5} win={row.win_rate:>5.1f}%  "
              f"net={row.net:>+10,.2f}  avg={row.avg:>+9,.2f}  "
              f"avg size ${row.extra['avg_notional']:>9,.0f}{flag(row.n)}")
    print("\n  The confidence buckets are a HYPOTHESIS, not a calibration:")
    print("  nothing yet shows an 85 wins more often than a 65. These rows are")
    print("  the evidence that will eventually confirm or kill that.")
    return 0


def _research_excursions(cfg, repo, strategy: str, args) -> int:
    """The take-profit question, answered from recorded forward paths."""
    from .research.forward_test import ExcursionReport
    r = ExcursionReport(repo, strategy, cfg=cfg.aggressive,
                        min_sample=args.min_sample)
    counts = r.status_counts()
    done = r.complete()

    if getattr(args, "json", False):
        print(json.dumps({
            "strategy": strategy, "status_counts": counts,
            "crossover_atr_pct": r.crossover_atr,
            "excursions": r.excursion_stats(),
            "hit_rates": r.hit_rates(),
            "stalled_2pct_to_2r": r.stalled_between_2pct_and_2r(),
            "two_r_cheaper": r.two_r_cheaper_than_2pct(),
            "ambiguity": r.ambiguity_share(),
            "breakdowns": {k: {b: len(v) for b, v in g.items()}
                           for k, g in r.breakdowns().items()},
        }, indent=2, default=str))
        return 0

    def flag(n):
        return "" if n >= args.min_sample else "  [SAMPLE TOO SMALL]"

    print("=" * 76)
    print(f"  FORWARD EXCURSIONS — {strategy}   ({cfg.exchange_label()})")
    print(f"  paths: {counts.get('open', 0)} open, {counts.get('complete', 0)} complete")
    print(f"  2R and a fixed +2% coincide at atr_pct = {r.crossover_atr:.4f}%")
    print("=" * 76)
    if not done:
        print("\n  No COMPLETE paths yet. A path completes when its stop is")
        print("  touched or its 24h horizon elapses, so this stays empty until")
        print("  the forward test has been running. Nothing below can be")
        print("  computed from open paths without biasing every rate downward.")
        return 0

    e = r.excursion_stats()
    print(f"\nMFE / MAE over {e['n']} complete path(s){flag(e['n'])}")
    print(f"  MFE   mean {e['mfe_mean']:+.2f}%  median {e['mfe_median']:+.2f}%  "
          f"max {e['mfe_max']:+.2f}%")
    print(f"  MAE   mean {e['mae_mean']:+.2f}%  median {e['mae_median']:+.2f}%  "
          f"min {e['mae_min']:+.2f}%")
    print(f"  median minutes to MFE {e['minutes_to_mfe_median']}, "
          f"to MAE {e['minutes_to_mae_median']}")
    print(f"  stopped out: {e['stopped']} ({e['stopped_pct']:.1f}%), "
          f"median minutes to stop {e['minutes_to_stop_median']}")

    amb = r.ambiguity_share()
    print(f"\nINTRABAR AMBIGUITY  {amb['any_ambiguous']}/{amb['paths']} paths "
          f"({amb['share_pct']:.1f}%) had a stop and a target in the SAME 5m bar")
    print("  Those are excluded from every rate below, on both sides. A large")
    print("  share here means the answer is finer data, not a bolder assumption.")

    print("\nHIT RATES — reached BEFORE the stop, unambiguously")
    print(f"  {'target':<10} {'decided':>8} {'hit':>6} {'rate':>8} {'ambig':>7}")
    for row in r.hit_rates():
        print(f"  {row['target']:<10} {row['decided']:>8} {row['hit']:>6} "
              f"{row['hit_rate_pct']:>7.1f}% {row['ambiguous']:>7}"
              f"{flag(row['decided'])}")

    q1 = r.stalled_between_2pct_and_2r()
    print(f"\nQ1  {q1['question']}   ({q1['atr_filter']})")
    print(f"  {q1['stalled']}/{q1['decided']} decided = {q1['stalled_pct']:.1f}% "
          f"({q1['ambiguous']} ambiguous, {q1['reached_2r']} did reach 2R)"
          f"{flag(q1['decided'])}")
    print("  This is the money-on-the-table case: a fixed +2% would have banked")
    print("  a move the 2R target gave back.")

    q2 = r.two_r_cheaper_than_2pct()
    print(f"\nQ2  {q2['question']}   ({q2['atr_filter']})")
    print(f"  {q2['two_r_only']}/{q2['decided']} decided = {q2['two_r_only_pct']:.1f}% "
          f"({q2['ambiguous']} ambiguous){flag(q2['decided'])}")
    print("  Below the crossover the CURRENT rule is the less demanding one.")

    print("\nBREAKDOWNS  (hit rate for +2.0% and for 2R, before the stop)")
    for name, groups in r.breakdowns().items():
        if not groups:
            continue
        print(f"\n  {name}")
        for bucket, paths in sorted(groups.items()):
            rows = {x['target']: x for x in r.hit_rates(paths)}
            p2, r2 = rows['pct_2.0'], rows['r_2.0']
            print(f"    {bucket:<14} n={len(paths):<5} "
                  f"+2.0%: {p2['hit_rate_pct']:>5.1f}% ({p2['decided']} decided)   "
                  f"2R: {r2['hit_rate_pct']:>5.1f}% ({r2['decided']} decided)"
                  f"{flag(p2['decided'])}")
    return 0


def _research_exit_quality(cfg, repo, strategy: str, args) -> int:
    """Exit behaviour on one accounting basis at a time.

    Gross and net are computed from the same per-trade rows and printed side by
    side. They are never combined: a trade can be a gross win and a net loss,
    and mixing a win count from one basis with a total from the other is how a
    working exit system gets certified from numbers that do not describe the
    same trades.
    """
    from .research import economics as ec

    q = ec.exit_quality(repo, strategy)
    if getattr(args, "json", False):
        print(json.dumps({"strategy": strategy, **q}, indent=2, default=str))
        return 0

    def num(v, w=10, p=2, sign=True):
        if v is None:
            return " " * (w - 2) + "--"
        return f"{v:{'+' if sign else ''}{w}.{p}f}"

    print("=" * 78)
    print(f"  EXIT QUALITY — {strategy}   ({cfg.exchange_label()})")
    print(f"  {q['closed_trades']} closed trade(s); fees simulated at "
          f"{cfg.execution.fee_label()}")
    print("=" * 78)
    if not q["closed_trades"]:
        print("\n  No closed trades. Nothing to measure.")
        return 0

    print("\nTWO BASES, NEVER COMBINED")
    print("  GROSS = (exit_ref - entry_ref) x qty x direction, BEFORE fees and")
    print("  slippage. NET = after both. A win count from one and a total from")
    print("  the other do not describe the same trades.")
    print(f"  {'':<22} {'GROSS':>14} {'NET':>14}")
    g, n = q["gross"], q["net"]
    for label, key, pct in (("trades", "n", False), ("wins", "wins", False),
                            ("losses", "losses", False),
                            ("win rate", "win_rate_pct", True),
                            ("total P&L", "total", False),
                            ("average winner", "avg_win", False),
                            ("average loser", "avg_loss", False),
                            ("win/loss ratio", "win_loss_ratio", False),
                            ("expectancy/trade", "expectancy", False),
                            ("profit factor", "profit_factor", False)):
        def cell(d):
            v = d[key]
            if v is None:
                return f"{'--':>14}"
            if isinstance(v, int) and not pct:
                return f"{v:>14}"
            return f"{v:>13.1f}%" if pct else f"{v:>14.3f}"
        print(f"  {label:<22} {cell(g)} {cell(n)}")

    print(f"\n  COST-FLIPPED: {q['cost_flipped']} trade(s) had gross P&L > 0 and")
    print(f"  net P&L <= 0 — the market paid and the costs took it back. Those")
    print(f"  carried {q['cost_flipped_gross']:+.2f} of gross P&L while counting")
    print("  as losses on the net basis. This figure IS the size of the error a")
    print("  gross/net mix-up produces.")
    if q["impossible_rows"]:
        print(f"  !! {q['impossible_rows']} row(s) have net > gross, which the")
        print("     cost model cannot produce. The ledger is inconsistent —")
        print("     stop here and investigate before reading anything above.")

    print("\nR-MULTIPLES — P&L over the risk each trade was OPENED with")
    print("  |entry_fill - initial_stop| x qty. The initial stop, not the final")
    print("  one: dividing by a ratcheted stop measures the trail and makes")
    print("  every trailed winner a multiple of a risk nobody took.")
    print(f"  measurable on {q['r_measurable']}/{q['closed_trades']} trades"
          f" ({q['r_unmeasurable']} had no usable initial stop)")
    rg, rn = q["r_gross"], q["r_net"]
    print(f"  {'':<22} {'GROSS R':>14} {'NET R':>14}")
    for label, key in (("average winner", "avg_win"),
                       ("average loser", "avg_loss"),
                       ("win/loss ratio", "win_loss_ratio"),
                       ("expectancy (R)", "expectancy"),
                       ("largest winner", "largest_win"),
                       ("largest loser", "largest_loss")):
        def rcell(d):
            v = d[key]
            return f"{'--':>14}" if v is None else f"{v:>14.3f}"
        print(f"  {label:<22} {rcell(rg)} {rcell(rn)}")

    print("\nBY EXIT REASON  (gross and net kept apart)")
    print(f"  {'reason':<18} {'n':>4} {'gross W':>8} {'net W':>7} "
          f"{'gross $':>11} {'net $':>11} {'avg R gross':>12}")
    for b in q["by_exit_reason"]:
        ar = ("--" if b["avg_r_gross"] is None
              else f"{b['avg_r_gross']:+.2f}")
        print(f"  {b['reason']:<18} {b['n']:>4} {b['gross_wins']:>8} "
              f"{b['net_wins']:>7} {b['gross']:>+11.2f} {b['net']:>+11.2f} "
              f"{ar:>12}")

    print(f"\n  {q['warning']}.")
    print("\n  This reads the ledger; it does not model an exit. For exit")
    print("  QUESTIONS use `--policy-sim`, which replays the real engine over")
    print("  the stored v8 tape. This report exists to stop an aggregate")
    print("  shortcut being taken, not to become one.")
    return 0


def _research_fee_scenarios(cfg, repo, strategy: str, args) -> int:
    """The same closed trades priced at every fee tier, fills unchanged."""
    from .research import economics as ec

    out = ec.fee_scenarios(repo, strategy, cfg.execution.effective_taker_bps())
    if getattr(args, "json", False):
        print(json.dumps(out, indent=2, default=str))
        return 0

    print("=" * 78)
    print(f"  FEE SCENARIOS — {strategy}   ({cfg.exchange_label()})")
    print(f"  simulating at: {cfg.execution.fee_label()}")
    print(f"  {out['closed_trades']} closed trade(s); entry turnover "
          f"{out['entry_turnover']:,.2f}; two-leg notional "
          f"{out['two_leg_notional']:,.2f}")
    print("=" * 78)
    if not out["closed_trades"]:
        print("\n  No closed trades. Nothing to re-price.")
        return 0
    print(f"\n  HELD FIXED: gross P&L {out['gross_pnl']:+.2f} "
          f"({out['gross_bps']:+.1f} bps of turnover), slippage "
          f"{-out['slippage']:+.2f}, financing {-out['financing']:+.2f}.")
    print(f"  {out['note']}.")

    print(f"\n  {'tier':<8} {'bps/side':>8} {'total fees':>11} {'net P&L':>11} "
          f"{'exp/trade':>10} {'PF':>6} {'BE gross':>10} {'vs actual':>10}")
    for r in out["rows"]:
        pf = ("--" if r["profit_factor"] is None
              else ("inf" if r["profit_factor"] == float("inf")
                    else f"{r['profit_factor']:.2f}"))
        mult = ("no edge" if r["gross_shortfall_multiple"] is None
                else f"{r['gross_shortfall_multiple']:,.1f}x")
        print(f"  {r['tier']:<8} {r['bps_per_side']:>8.1f} "
              f"{-r['total_fees']:>+11.2f} {r['net_pnl']:>+11.2f} "
              f"{r['expectancy_per_trade']:>+10.2f} {pf:>6} "
              f"{r['breakeven_gross_bps']:>9.1f}b {mult:>10}")
    print("\n  BE gross = the gross edge, in bps of entry turnover, needed for")
    print("  net zero at that tier with slippage unchanged. `vs actual` is how")
    print(f"  many times the realised gross edge ({out['gross_bps']:+.1f} bps)"
          " that is;")
    print("  `no edge` means gross P&L is not positive, so NO fee tier can")
    print("  rescue it -- the sign is the problem, not the size.")
    viable = [r["tier"] for r in out["rows"] if r["viable"]]
    print(f"\n  PROFITABLE AT: {', '.join(viable) if viable else 'NO TIER'}")
    return 0


def _research_atr_economics(cfg, repo, strategy: str, args) -> int:
    """What each ATR band can pay for. Arithmetic — needs no trades."""
    from .research import economics as ec

    atrs = tuple(float(x) for x in str(args.atr_pcts).split(",") if x.strip())
    out = ec.atr_economics(cfg.aggressive, cfg.execution, atrs)
    if getattr(args, "json", False):
        print(json.dumps(out, indent=2, default=str))
        return 0

    print("=" * 78)
    print(f"  ATR ECONOMICS — {strategy}")
    print(f"  stop = {out['stop_atr_mult']}x ATR; target = {out['target_r']}R"
          f" = {out['target_r'] * out['stop_atr_mult']}x ATR")
    print(f"  configured ATR floor: {out['min_atr_pct_configured']:g}%")
    print(f"  break-even win rate at ZERO cost: "
          f"{out['zero_cost_breakeven_win_rate_pct']:.1f}%")
    print("=" * 78)
    print(f"\n  {out['caveat']}.")

    for row in out["rows"]:
        print(f"\n  ATR {row['atr_pct']:g}%   stop {row['stop_pct']:.2f}%   "
              f"2R target {row['target_pct']:.2f}%")
        print(f"    {'tier':<8} {'fee r/t':>8} {'slip win':>9} {'slip loss':>10} "
              f"{'net target':>11} {'cost/target':>12} {'BE win rate':>12}")
        for t in row["tiers"]:
            be = ("UNTRADEABLE" if t["structurally_untradeable"]
                  else f"{t['breakeven_win_rate_pct']:.1f}%")
            print(f"    {t['tier']:<8} {t['fee_round_trip_pct']:>7.2f}% "
                  f"{t['slippage_win_pct']:>8.2f}% {t['slippage_loss_pct']:>9.2f}% "
                  f"{t['net_target_pct']:>10.2f}% "
                  f"{t['cost_share_of_target_pct']:>11.1f}% {be:>12}")

    print("\n  MINIMUM TRADABLE ATR% BY TIER")
    print("  Below this the 2R target does not cover a WINNING round trip, so")
    print("  no win rate rescues it. A setup under this line is structurally")
    print("  untradeable, not merely marginal.")
    floor = out["min_atr_pct_configured"]
    for name, a in out["min_tradable_atr_pct"].items():
        flag = "  <-- ABOVE the configured ATR floor" if a > floor else ""
        print(f"    {name:<8} {a:>6.3f}%{flag}")
    return 0


def _research_short_funnel(cfg, repo, strategy: str, args) -> int:
    """Why no shorts, and whether the score floor is in the right place.

    Everything printed here is recomputed from stored features, never parsed
    out of a rejection label -- see `research/short_funnel` for why the labels
    cannot answer it.
    """
    from .research import short_funnel as sf

    floors = tuple(sorted({float(x) for x in str(args.score_floors).split(",")
                           if x.strip()})) or (60.0, 70.0, 75.0, 80.0)
    f = sf.ShortFunnel(repo, strategy, cfg,
                       horizon_h=getattr(args, "horizon", None))
    directional = tuple(g for g in sf.ALL_GATES
                        if g not in (sf.GATE_SCORE, sf.GATE_CONFIDENCE))
    funnel = f.funnel()
    overlap, overlap_dir = f.overlap(), f.overlap(directional)
    ablations, floors_out = f.ablations(), f.score_floors(floors)
    costs, buckets = f.cost_split(), f.score_bucket_costs()
    contest = f.side_contest()

    if getattr(args, "json", False):
        print(json.dumps({
            "strategy": strategy, "cost_bps": f.cost_bps,
            "cost_bps_measured": f.measured_cost,
            "horizon_h": f.horizon_h, "observations": len(f.candidates),
            "funnel": funnel, "overlap": overlap,
            "overlap_directional": overlap_dir, "side_contest": contest,
            "ablations": ablations, "score_floors": floors_out,
            "cost_split": costs, "score_buckets": buckets,
        }, indent=2, default=str))
        return 0

    def pc(v, w=7):
        return f"{v:+{w}.2f}%" if isinstance(v, (int, float)) else " " * (w - 1) + "--"

    src = "measured from the ledger" if f.measured_cost else "DEFAULT, no trades yet"
    print("=" * 78)
    print(f"  SHORT FUNNEL — {strategy}   ({cfg.exchange_label()})")
    print(f"  {len(f.candidates)} observation(s); round-trip cost drag "
          f"{f.cost_bps:.1f} bps ({src})")
    hz = (f"horizon {f.horizon_h}h only" if f.horizon_h is not None
          else "all recorded horizons, averaged per observation")
    print(f"  counterfactuals: {hz}; signed TOWARD THE SHORT")
    print("=" * 78)
    print("\nHYPOTHETICAL / NOT EXECUTED. A counterfactual is a raw price move")
    print("with no stop, no target and no costs. `after cost` subtracts the")
    print("measured drag only -- it does NOT model a stop, so a positive")
    print("figure is an upper bound on what the signal could have paid.")

    print("\n1. SEQUENTIAL FUNNEL  (live order; `rejected` is out of `reached`)")
    print(f"  {'stage':<24} {'reached':>8} {'rejected':>9} {'survivor cf':>12} "
          f"{'rejected cf':>12}")
    for r in funnel:
        rs = r["rejected_stats"]
        print(f"  {r['stage']:<24} {r['reached']:>8} {r['rejected']:>9} "
              f"{pc(r['mean_cf_pct'], 11):>12} {pc(rs['mean_cf_pct'], 11):>12}")
        if r["stage"] != "observations evaluated":
            extra = (f"   [{r['unavailable']} of those had no recorded value]"
                     if r.get("unavailable") else "")
            print(f"      rule: {r['gate']}{extra}")
    print(f"\n  side contest: {contest['short_clear']} observation(s) cleared all"
          f" three short blockers;")
    print(f"  {contest['short_and_long_both_clear']} would ALSO have cleared the"
          " long structure gates.")
    print(f"  {contest['note']}.")

    print("\n2. OVERLAPPING SHORT REJECTIONS  (each gate judged INDEPENDENTLY)")
    print("  The recorded rejection labels are MUTUALLY EXCLUSIVE by")
    print("  construction -- `choose_side` joins every blocker into one")
    print("  string -- so a single-gate count there UNDERSTATES that gate's")
    print("  reach. These totals are recomputed per gate, ignoring order.")
    print("  `total` is what a gate would stop on its own. `only this` is what")
    print("  NOTHING else would have stopped -- the only figure that says what")
    print("  relaxing it ALONE would admit.")

    def show_overlap(o, title):
        print(f"\n  {title}")
        print(f"  {'gate':<24} {'total':>7} {'only this':>10}   also blocked by")
        any_row = False
        for g in o["gates"]:
            if not o["totals"][g]:
                continue
            any_row = True
            co = [f"{h}:{n}" for h, n in o["matrix"][g].items()
                  if h != g and n]
            print(f"  {g:<24} {o['totals'][g]:>7} "
                  f"{o['only_this_gate'][g]:>10}   "
                  f"{', '.join(co) if co else '(nothing)'}")
        if not any_row:
            print("  (no gate rejected anything)")
        print(f"  gates blocking each observation: {o['n_blockers_histogram']}")

    show_overlap(overlap_dir, "DIRECTIONAL GATES ONLY (score/confidence excluded)")
    show_overlap(overlap, "EVERY GATE, including the score and confidence floors")

    print("\n3. SHORT-GATE ABLATION  (offline; nothing is applied)")
    print("  Each variant DROPS the gate rather than nudging it, so")
    print("  `extra` is a CEILING on what relaxing it could admit.")
    print("  `extra dir` counts what clears the DIRECTIONAL gates once the")
    print("  named one is dropped; `extra all` also requires the score and")
    print("  confidence floors. Where the two differ, the structural gate was")
    print("  never the binding constraint -- the score floor was.")
    print(f"  {'variant':<38} {'extra dir':>9} {'extra all':>9} {'cf':>9} "
          f"{'after cost':>11} {'replayable':>11}")
    for a in ablations:
        e = a["extra_directional_stats"]
        print(f"  {a['variant']:<38} {a['extra_directional']:>9} "
              f"{a['extra_admitted']:>9} "
              f"{pc(e['mean_cf_pct'], 8):>9} "
              f"{pc(e['mean_after_cost_pct'], 10):>11} "
              f"{e['replayable']:>4}/{e['n']:<6}")
    unreplayable = sum(1 for a in ablations
                       if a["extra_directional"]
                       and a["extra_directional_stats"]["replayable"]
                       < a["extra_directional_stats"]["n"])
    if unreplayable:
        print(f"\n  WARNING: {unreplayable} variant(s) admit candidates whose"
              " forward path was")
        print("  NOT taped. For those, stop and take-profit P&L cannot be")
        print("  reproduced from this data AT ALL -- only the raw forward move")
        print("  exists, and a 1.8-ATR stop sits well inside most of these")
        print("  moves. Nothing in this table says those shorts would have")
        print("  survived to collect the figure beside them.")

    print("\n4. SCORE THRESHOLD COMPARISON")
    print(f"  REALISED (closed trades only; {floors_out['total_closed_trades']}"
          f" total, {floors_out['trades_without_recorded_score']} without a"
          " recorded score)")
    print(f"  {'floor':>6} {'trades':>7} {'win%':>6} {'gross':>10} {'fees':>9} "
          f"{'slip':>9} {'net':>10} {'net bps':>9}")
    for r in floors_out["realised"]:
        nb = (f"{r['net_bps']:+9.1f}" if r["net_bps"] is not None else "       --")
        print(f"  {r['floor']:>6.0f} {r['closed_trades']:>7} "
              f"{r['win_rate_pct']:>5.0f}% {r['gross_pnl']:>+10.2f} "
              f"{-r['fees']:>+9.2f} {-r['slippage']:>+9.2f} "
              f"{r['net_pnl']:>+10.2f} {nb}")
    print(f"\n  HYPOTHETICAL (observations at or above the floor, short-side"
          " score)")
    print(f"  recomputed short score available on"
          f" {floors_out['short_score_computable']} observation(s),"
          f" missing on {floors_out['short_score_missing']}")
    print(f"  distribution: {floors_out['short_score_histogram']}")
    print(f"  {'floor':>6} {'obs':>7} {'with outcome':>13} {'cf':>9} "
          f"{'after cost':>11} {'share +':>9}")
    for r in floors_out["hypothetical"]:
        sp = (f"{r['share_positive']:>8.1f}%" if r["share_positive"] is not None
              else "       --")
        print(f"  {r['floor']:>6.0f} {r['observations']:>7} "
              f"{r['with_outcome']:>13} {pc(r['mean_cf_pct'], 8):>9} "
              f"{pc(r['mean_after_cost_pct'], 10):>11} {sp}")
    print(f"\n  {floors_out['warning']}.")
    for c in floors_out["caveats"]:
        print(f"  - {c}")

    print("\n5. CONFIDENCE CALIBRATION  (realised, per unit of notional risked)")
    print("  Dollar P&L confounds quality with size: the ladder gives a higher")
    print("  score more notional. bps of turnover separates them, and adding")
    print("  the cost back recovers the GROSS edge -- which is what says")
    print("  whether a bucket has no edge or an edge smaller than its costs.")
    print(f"  {'bucket':<10} {'n':>4} {'win%':>6} {'avg notional':>13} "
          f"{'net $':>10} {'net bps':>9} {'cost bps':>9} {'gross bps':>10}")
    for b in buckets:
        def bp(v):
            return f"{v:+9.1f}" if v is not None else "       --"
        print(f"  {b['bucket']:<10} {b['n']:>4} {b['win_rate_pct']:>5.0f}% "
              f"{b['avg_notional']:>13,.0f} {b['net_pnl']:>+10.2f} "
              f"{bp(b['net_bps'])} {bp(b['cost_bps'])} {bp(b['gross_bps']):>10}")

    print("\n6. COST-DRAG BREAKDOWN  (rebuilt from the four stored prices)")
    if not costs["n_trades"]:
        print("  No closed trades yet -- nothing to decompose.")
        return 0
    b = costs.get("bps", {})
    print(f"  turnover {costs['turnover']:>14,.2f}  over {costs['n_trades']}"
          f" trade(s), {costs['stop_exits']} stop exit(s),"
          f" {costs['gapped_exits']} GAPPED")
    for label, key in (("entry fee", "entry_fee"), ("exit fee", "exit_fee"),
                       ("entry slippage", "entry_slippage"),
                       ("exit slippage", "exit_slippage"),
                       ("  of which modelled stop bps",
                        "stop_slippage_modelled"),
                       ("  of which GAP through the stop", "gap_component"),
                       ("  of which non-stop exits", "exit_slippage_non_stop"),
                       ("financing", "financing")):
        print(f"  {label:<34} {-costs[key]:>+12.2f} "
              f"{-b.get(key, 0.0):>+9.1f} bps")
    print(f"  {'gross P&L':<34} {costs['gross_pnl']:>+12.2f} "
          f"{b.get('gross_pnl', 0.0):>+9.1f} bps")
    print(f"  {'net P&L':<34} {costs['net_pnl']:>+12.2f} "
          f"{b.get('net_pnl', 0.0):>+9.1f} bps")
    print("\n  reconstruction residuals (must be ~0, else read nothing above):")
    print(f"    fees {costs['fee_residual']:+.6f}   "
          f"slippage {costs['slippage_residual']:+.6f}   "
          f"gap split {costs['gap_residual']:+.6f}")
    print("\n  The GAP line is not a modelling assumption -- it is the distance")
    print("  a 5m bar opened beyond the stop. Lowering `slippage_bps` cannot")
    print("  remove it, and doing so would only hide the part that is.")
    return 0


def _research_policy_sim(cfg, repo, strategy: str, args) -> int:
    """CONTROL vs a fixed +2% target, replayed from stored tapes."""
    from .research import policy_report as rep
    from .research import policy_sim as sim

    cmp_ = rep.PolicyComparison(repo, strategy, cfg,
                                fixed_tp_pct=args.fixed_tp)
    recon = cmp_.reconcile()
    pairs = cmp_.pairs()
    summary = cmp_.summarise(pairs)

    if getattr(args, "json", False):
        print(json.dumps({
            "strategy": strategy,
            "reconciliation": {
                "compared": recon.compared,
                "exit_reason_match_pct": recon.reason_match_pct,
                "exact_bar": recon.exact_bar,
                "within_one_bar": recon.within_one_bar,
                "fill_ok": recon.fill_ok, "net_ok": recon.net_ok,
                "trustworthy": recon.trustworthy,
                "unreplayable": recon.unreplayable,
                "by_exit_reason": recon.by_exit_reason,
                "discrepancies": [vars(d) for d in recon.discrepancies[:100]],
            },
            "summary": summary,
            "pairs": [p.as_dict() for p in pairs if p.ok][:500],
        }, indent=2, default=str))
        return 0

    print("=" * 78)
    print(f"  EXIT-POLICY REPLAY — {strategy}   ({cfg.exchange_label()})")
    print(f"  CONTROL = 2R close-based target   "
          f"TP2 = fixed +{args.fixed_tp:.2f}% close-based")
    print("=" * 78)

    # ---- reconciliation FIRST. Nothing below it is worth reading until
    #      the simulator has shown it can reproduce what already happened.
    print("\n[1] RECONCILIATION — CONTROL replayed against the real ledger")
    if not recon.compared:
        print("  No closed Strategy B trade could be replayed yet.")
        for why, n in sorted(recon.unreplayable.items()):
            print(f"    {n:>5}  {why}")
        print("  Until this reconciles, every TP2 number below is UNVALIDATED.")
    else:
        print(f"  compared            {recon.compared}")
        print(f"  exit reason matched {recon.exact_reason} "
              f"({recon.reason_match_pct:.1f}%)")
        print(f"  same bar            {recon.exact_bar}   "
              f"within one bar: {recon.within_one_bar}")
        print(f"  fill within tol     {recon.fill_ok}")
        print(f"  net P&L within tol  {recon.net_ok}")
        for why, n in sorted(recon.unreplayable.items()):
            print(f"  unreplayable        {n:>5}  {why}")
        if recon.by_exit_reason:
            print("  by exit reason:")
            for k, v in sorted(recon.by_exit_reason.items()):
                print(f"    {k:<26} n={v['n']:<4} reason {v['reason_match']}"
                      f"  fill {v['fill_ok']}")
        if recon.discrepancies:
            print(f"\n  {len(recon.discrepancies)} DISCREPANCY(IES):")
            for d in recon.discrepancies[:12]:
                print(f"    {d.trade_id} {d.field}: live={d.live} "
                      f"sim={d.simulated}  ({d.note})")
        verdict = ("TRUSTWORTHY" if recon.trustworthy
                   else "NOT VALIDATED — treat everything below as suspect")
        print(f"\n  VERDICT: {verdict}")

    # ---- coverage -----------------------------------------------------
    print(f"\n[2] COVERAGE")
    print(f"  total paths         {summary['total_paths']}")
    print(f"  replayable pairs    {summary['replayable']}")
    for why, n in sorted(summary["unreplayable"].items()):
        print(f"  excluded            {n:>5}  {why}")

    label = summary["label"]
    print(f"\n[3] PAIRED RESULT — {label}")
    if label == rep.INSUFFICIENT:
        print(f"  Fewer than {rep.MIN_PROVISIONAL} paired complete paths. No")
        print("  winner is declared, and the numbers below are shown only so")
        print("  the pipeline can be seen working.")
    _print_pair_block(summary, indent="  ")

    h = summary["halves"]
    print(f"\n[4] CHRONOLOGICAL HALVES")
    print(f"  first  n={h['first']['n']:<5} mean diff "
          f"{h['first']['mean_diff_pct']:+.4f}%  sign {h['first']['sign']}")
    print(f"  second n={h['second']['n']:<5} mean diff "
          f"{h['second']['mean_diff_pct']:+.4f}%  sign {h['second']['sign']}")
    print(f"  same sign in both halves: {h['same_sign']}")
    print("  An advantage present in only one half is a regime artefact")
    print("  wearing the costume of an edge.")

    print(f"\n[5] CROSSOVER GROUPS  (2R and +{args.fixed_tp:.2f}% coincide at "
          f"atr_pct {cmp_.crossover_atr:.4f}%)")
    for name, block in summary["crossover"].items():
        print(f"\n  {name}")
        _print_pair_block(block, indent="    ")

    for title, key in (("BY SIDE", "by_side"),
                       ("BY CONFIDENCE BUCKET", "by_confidence_bucket"),
                       ("BY SETUP-SCORE BUCKET", "by_score_bucket"),
                       ("BY ATR% BUCKET", "by_atr_bucket"),
                       ("BY BTC REGIME", "by_btc_regime")):
        groups = summary[key]
        if not groups:
            continue
        print(f"\n[6] {title}")
        print(f"  {'bucket':<16}{'n':>5}{'CTRL win':>9}{'TP2 win':>8}"
              f"{'mean diff':>11}{'CTRL exp':>10}{'TP2 exp':>9}")
        for bucket, st_ in groups.items():
            print(f"  {bucket:<16}{st_['n']:>5}{st_['control_wins']:>9}"
                  f"{st_['tp2_wins']:>8}{st_['mean_diff_pct']:>10.4f}%"
                  f"{st_['CONTROL']['expectancy_pct']:>9.4f}%"
                  f"{st_['TP2']['expectancy_pct']:>8.4f}%")
    return 0


def _print_pair_block(s: dict, indent: str = "  ") -> None:
    i = indent
    print(f"{i}pairs {s['n']}   CONTROL wins {s['control_wins']}   "
          f"TP2 wins {s['tp2_wins']}   ties {s['ties']}")
    print(f"{i}mean paired diff (TP2 - CONTROL) {s['mean_diff_pct']:+.4f}%   "
          f"median {s['median_diff_pct']:+.4f}%")
    c, t = s["CONTROL"], s["TP2"]
    print(f"{i}{'':<22}{'CONTROL':>12}{'TP2':>12}")
    for label, key, fmt in (
            ("net expectancy %", "expectancy_pct", "{:>12.4f}"),
            ("gross expectancy %", "gross_expectancy_pct", "{:>12.4f}"),
            ("win rate %", "win_rate_pct", "{:>12.1f}"),
            ("profit factor", "profit_factor", "{:>12.2f}"),
            ("avg winner %", "avg_winner_pct", "{:>12.4f}"),
            ("avg loser %", "avg_loser_pct", "{:>12.4f}"),
            ("median hold (min)", "hold_minutes_median", "{:>12.0f}")):
        cv, tv = c.get(key, 0.0), t.get(key, 0.0)
        cs = "         inf" if cv == float("inf") else fmt.format(cv)
        ts = "         inf" if tv == float("inf") else fmt.format(tv)
        print(f"{i}{label:<22}{cs}{ts}")


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

    broker = PaperBroker(cfg.execution.effective_taker_bps(), cfg.execution.slippage_bps,
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


def cmd_preflight(args) -> int:
    """Everything that must be true before a live forward test starts.

    One command rather than a checklist of four, because a readiness check
    people have to remember to assemble is one they will eventually assemble
    incompletely. Read-only with respect to trading: no orders exist anywhere
    in this codebase, and nothing here can open a position.
    """
    from .execution.paper_broker import PaperBroker
    from .verify_live import (VerifyReport, verify_circuit_breakers,
                              verify_fast_timeframes,
                              verify_fast_timeframes_for, verify_exchange,
                              verify_pending_first, verify_quotes,
                              verify_regime_and_breadth, verify_restart_recovery,
                              verify_schema, verify_strategy_b_contract,
                              verify_telegram, verify_universe)

    cfg, repo, feed, notifier = _bootstrap(args, need_feed=True)
    if not args.strategies:
        # The forward test is Strategy B. Say so unless told otherwise.
        cfg.apply_runtime_mode("b")
    rep = VerifyReport()
    print("=" * 72)
    print(f"FORWARD-TEST PREFLIGHT -- {cfg.exchange_label()}")
    print("=" * 72)
    print("Read-only with respect to trading. No orders exist in this codebase.")
    rep.facts["mode"] = (f"{cfg.runtime_mode()} — entries: "
                         f"{', '.join(cfg.enabled_strategies()) or 'NONE'}")
    _print_runtime_mode(cfg, repo)

    ex = verify_exchange(cfg, feed, rep)
    verify_fast_timeframes(cfg, feed, rep)

    broker = PaperBroker(cfg.execution.effective_taker_bps(), cfg.execution.slippage_bps,
                         cfg.execution.stop_slippage_bps,
                         cfg.execution.use_book_spread,
                         cfg.execution.max_spread_bps_entry)
    verify_quotes(cfg, feed, broker, args.quote_samples, args.quote_interval, rep)

    universe = []
    if not args.skip_universe:
        uni = verify_universe(cfg, repo, _broad_service(cfg, repo),
                              ex.get("markets", {}), ex.get("tickers", {}), rep)
        universe = uni.get("universe", [])
        # The fast frames matter most on the markets B will actually rank, so
        # check one that is NOT BTC.
        alt = next((s for s in universe if s != cfg.strategy.btc_symbol), "")
        verify_fast_timeframes_for(cfg, feed, alt, rep)
        rep.facts["alt"] = (f"OK on {alt}" if alt and
                            not rep.failures_for("alt") else
                            (f"FAILED on {alt}" if alt else "no alt market"))
        verify_regime_and_breadth(cfg, feed, repo, ex.get("markets", {}),
                                  ex.get("tickers", {}), universe, rep)

    if not args.skip_telegram:
        verify_telegram(cfg, repo, notifier, rep)
        verify_pending_first(notifier, repo, rep)

    verify_schema(cfg, repo, rep)
    verify_restart_recovery(cfg, repo, rep)
    verify_strategy_b_contract(cfg, repo, rep)
    # Last, and it can veto the whole verdict: a halted strategy opens nothing.
    verify_circuit_breakers(cfg, repo, rep)

    print(rep.render_preflight(cfg))
    return 0 if rep.passed else 1


def cmd_diagnose(args) -> int:
    """Account for every venue market: where it was dropped and why."""
    from .verify_live import diagnose_universe

    cfg, repo, feed, _ = _bootstrap(args, need_feed=True)
    diagnose_universe(cfg, repo, _broad_service(cfg, repo), feed,
                      limit=args.limit)
    return 0


def cmd_scan(args) -> int:
    """Rank the live universe and show what Strategy B would do. NO TRADING.

    Stage 2 is signal generation: this ranks, deepens the shortlist and prints
    a side and a score per candidate. Nothing is sized and no position is
    opened -- that is Stage 3.
    """
    from .research.journal import ResearchJournal
    from .scan import scan
    from .strategy.base import MarketContext
    from .strategy.regime import btc_regime, market_breadth

    from .data.universe import UniverseBuilder
    from .engine import TradingEngine

    cfg, repo, feed, notifier = _bootstrap(args, need_feed=True)
    a = cfg.aggressive
    engine = TradingEngine(cfg, repo, feed, notifier)
    engine.refresh_universe(force=True)
    symbols = list(engine.status.universe)
    if not symbols:
        print(f"No candidates: {engine.status.broad_universe_reason}")
        return 1
    engine.fetch_data(symbols)

    hourly = {s: engine.series_for(a.rank_timeframe).get(s) for s in symbols}
    hourly = {k: v for k, v in hourly.items() if v is not None}
    btc_1h = engine.series_for(a.rank_timeframe).get(cfg.strategy.btc_symbol)
    label, score = btc_regime(engine.series_for(cfg.strategy.regime_timeframe).get(
        cfg.strategy.btc_symbol), cfg.strategy.regime_ema)
    ctx = MarketContext(
        ts_ms=now_ms(), btc_regime=label, btc_regime_score=score,
        breadth_pct=market_breadth(hourly, cfg.strategy.regime_ema),
        n_candidates=len(hourly), blocked_symbols=repo.blocked_symbols(now_ms()))

    tickers = getattr(engine, "_tickers", {}) or {}
    meta = {}
    for sym in hourly:
        t = tickers.get(sym) or {}
        qv, _how = UniverseBuilder.quote_volume(t)
        meta[sym] = {"dollar_volume": qv,
                     "spread_bps": UniverseBuilder.spread_bps(t)}

    t0 = time.time()
    res = scan(cfg, feed, ctx, rank_series=hourly, btc_1h=btc_1h,
               meta_by_symbol=meta, now_ms=now_ms(),
               buffer_ms=cfg.safety.candle_close_buffer_s * 1000)
    elapsed = time.time() - t0

    print("=" * 78)
    print(f"OPPORTUNITY SCAN -- {cfg.exchange_label()} -- {a.name} v{a.version}")
    print("=" * 78)
    print(f"  BTC regime {label} ({score:.0f})   breadth {ctx.breadth_pct:.0f}%   "
          f"candidates {len(res.ranked)}   shortlist {len(res.shortlist)}")
    print(f"  deep fetches {res.deep_fetches} (bounded by shortlist_size="
          f"{a.shortlist_size})   scan took {elapsed:.1f}s")
    print("-" * 78)
    print(f"  {'#':>2} {'SYMBOL':<14} {'RANK':>5} {'SIDE':<6} {'SETUP':>6}  DETAIL")
    journal = ResearchJournal(repo)
    for sig in res.signals:
        mark = "ENTRY" if sig.passed else ""
        detail = mark or sig.reject_reason[:34]
        print(f"  {sig.features['rank']:>2} {sig.symbol:<14} "
              f"{sig.features['rank_score']:>5.1f} {sig.side:<6} "
              f"{sig.score:>6.1f}  {detail}")
        journal.record(sig, "ENTERED" if sig.passed else "REJECTED_STRATEGY",
                       rank=sig.features["rank"])
    repo.conn.commit()
    print("-" * 78)
    longs = len([s for s in res.entries if s.side == "long"])
    shorts = len([s for s in res.entries if s.side == "short"])
    print(f"  {len(res.entries)} tradable setup(s): {longs} long, {shorts} short "
          f"-- RECORDED ONLY, nothing was traded")
    if res.fetch_failures:
        print(f"  {len(res.fetch_failures)} symbol(s) skipped on data: "
              f"{', '.join(sorted(res.fetch_failures))}")
    print("=" * 78)
    return 0


def cmd_verify_restart(args) -> int:
    """Prove that stopping and restarting loses nothing.

    Reads the persisted state twice through two independent connections, which
    is what a real restart does, and diffs what matters.
    """
    import sqlite3

    cfg, repo, _, _ = _bootstrap(args, need_feed=False)
    strategy = _strategy_arg(args, cfg)
    repo.ensure_account(strategy, cfg.starting_equity_for(strategy))
    acct = repo.get_account(strategy)
    positions = repo.get_positions(strategy)
    broad = repo.latest_broad_universe()
    outbox = repo.telegram_outbox_counts()
    candles = repo.conn.execute(
        "SELECT COUNT(*) AS n FROM processed_candles").fetchone()["n"]
    obs = repo.conn.execute("SELECT COUNT(*) AS n FROM observations").fetchone()["n"]
    repo.conn.close()

    conn2 = db.connect(cfg.engine.db_path)
    db.init_db(conn2)
    repo2 = Repo(conn2)
    acct2 = repo2.get_account(strategy)
    positions2 = repo2.get_positions(strategy)
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
    p.add_argument("--strategy", default=None,
                   help="which strategy ledger to report on; each strategy has "
                        "its own independent paper sub-account")
    p.add_argument("--strategies", default=None, choices=Config.RUNTIME_MODES,
                   help="which strategies may open NEW positions this run: "
                        "'a' = trend_breakout only, 'b' = aggressive_momentum_v2 "
                        "only, 'both'. A disabled strategy still manages the "
                        "positions it already holds through to their exits")
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
    s.add_argument("--aggressive", action="store_true",
                   help="report Strategy B (aggressive_momentum_v2) only, with "
                        "its confidence-bucket, ladder-slot and "
                        "binding-constraint breakdowns")
    s.set_defaults(func=cmd_performance)

    s = sub.add_parser("export", help="export closed trades to CSV")
    s.add_argument("--out", default="trades.csv")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("research", help="research database summary")
    s.add_argument("--aggressive", action="store_true",
                   help="the forward-test view of Strategy B: rejections with "
                        "their counterfactual outcomes, score buckets, "
                        "long vs short, and per-gate sensitivity")
    s.add_argument("--json", action="store_true")
    s.add_argument("--min-sample", type=int, default=20,
                   help="rows below this many outcomes are flagged, never hidden")
    s.add_argument("--policy-sim", action="store_true",
                   help="replay each stored tape under the CURRENT exit rules "
                        "and under a fixed +2%% target, paired on the same "
                        "signals; reconciles against the real ledger first")
    s.add_argument("--fixed-tp", type=float, default=2.0,
                   help="the fixed take-profit percentage TP2 tests "
                        "(default 2.0)")
    s.add_argument("--excursions", action="store_true",
                   help="forward-path view: MFE/MAE, which targets were reached "
                        "before the stop, and the 2R vs fixed +2%% comparison")
    s.add_argument("--short-funnel", action="store_true",
                   help="why no shorts: every short gate recomputed from stored "
                        "features, which gates overlap, what relaxing each "
                        "would admit, the score-floor comparison and the "
                        "cost-drag split")
    s.add_argument("--score-floors", default="60,70,75,80",
                   help="comma-separated setup-score floors to compare "
                        "(default 60,70,75,80)")
    s.add_argument("--horizon", type=int, default=None,
                   help="restrict counterfactual outcomes to one horizon in "
                        "hours; the default averages every recorded horizon")
    s.add_argument("--exit-quality", action="store_true",
                   help="exit behaviour on ONE accounting basis at a time: "
                        "gross and net side by side, R-multiples over the "
                        "risk each trade was opened with, and how many trades "
                        "flip sign between the two")
    s.add_argument("--fee-scenarios", action="store_true",
                   help="re-price every closed trade at each Kraken spot fee "
                        "tier, holding fills and slippage exactly as recorded")
    s.add_argument("--atr-economics", action="store_true",
                   help="what each ATR band can pay for at each fee tier: "
                        "stop, 2R target, round-trip costs and the break-even "
                        "win rate; needs no trades")
    s.add_argument("--atr-pcts", default="0.25,0.40,0.55,0.70,1.00,1.50",
                   help="comma-separated ATR%% values for --atr-economics")
    s.set_defaults(func=cmd_research)

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

    s = sub.add_parser(
        "preflight",
        help="everything that must be true before a live forward test: "
             "5m/15m/1h data, quotes, spreads, universe, BTC regime, breadth, "
             "Telegram, schema migration, restart recovery and the "
             "Strategy B parameter contract")
    s.add_argument("--quote-samples", type=int, default=5)
    s.add_argument("--quote-interval", type=float, default=2.0)
    s.add_argument("--skip-universe", action="store_true")
    s.add_argument("--skip-telegram", action="store_true")
    s.set_defaults(func=cmd_preflight)

    s = sub.add_parser("diagnose",
                       help="explain, per venue market, why it is or is not "
                            "in the tradable universe")
    s.add_argument("--limit", type=int, default=None,
                   help="probe at most N surviving symbols in stage 2 "
                        "(each costs one history fetch)")
    s.set_defaults(func=cmd_diagnose)

    sub.add_parser("scan",
                   help="rank the live universe and show what "
                        "aggressive_momentum_v2 would do -- RECORDS ONLY, "
                        "never trades"
                   ).set_defaults(func=cmd_scan)

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


def _network_error_types() -> tuple:
    """Exception types that mean 'could not reach the venue', not 'bug'.

    Deliberately narrow. A programming error must still surface as a traceback
    -- swallowing those would hide real defects behind a friendly message.
    """
    types: list[type] = [OSError]          # covers socket / TLS handshake failures
    try:
        import ccxt
        types.append(ccxt.NetworkError)
    except Exception:
        pass
    try:
        import requests
        types.append(requests.exceptions.RequestException)
    except Exception:
        pass
    return tuple(types)


def _caused_by_network(exc: BaseException) -> bool:
    """Walk the cause chain: was a data failure really a connectivity failure?

    DataUnavailable is raised both for 'the socket never opened' and for
    'the venue answered, but the answer was not usable'. Telling an operator to
    check their firewall when the venue simply had no candles would send them
    chasing the wrong problem, so the distinction is made from the cause chain
    rather than guessed from the message text.
    """
    net = _network_error_types()
    seen, cur = set(), exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, net):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _report_unreachable(cmd: str, exc: BaseException) -> None:
    """What a non-developer needs: what failed, and what to try."""
    detail = str(exc).strip() or exc.__class__.__name__
    if len(detail) > 300:
        detail = detail[:300] + "..."
    network = _caused_by_network(exc)
    headline = ("COULD NOT REACH THE EXCHANGE" if network
                else "COULD NOT GET USABLE MARKET DATA")
    print()
    print("=" * 72)
    print(f"{headline} -- '{cmd}' did not complete")
    print("=" * 72)
    print(f"  {detail}")
    print()
    print("  This is a data failure, not a result. Nothing above should be")
    print("  read as a finding about the market.")
    print()
    if network:
        print("  Common causes, in the order worth checking:")
        print("    1. No internet connection on this machine.")
        print("    2. A corporate VPN, firewall or proxy blocking the venue.")
        print("    3. The exchange is unreachable from your country.")
        print("    4. The venue is having an outage -- try again shortly.")
    else:
        print("  The venue was reachable but did not return data this run.")
        print("  Re-run the command; if it repeats, the detail line above")
        print("  names the request that failed.")
    print("=" * 72)


def main(argv=None) -> int:
    _force_utf8_output()
    args = build_parser().parse_args(argv)
    if getattr(args, "exchange", None):
        os.environ["CRYPTO_EDGE_EXCHANGE"] = args.exchange
    if getattr(args, "quote", None):
        os.environ["CRYPTO_EDGE_QUOTE"] = args.quote
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n  stopped")
        return 130
    except _network_error_types() as exc:
        _report_unreachable(getattr(args, "cmd", "command"), exc)
        return 2
    except DataUnavailable as exc:
        _report_unreachable(getattr(args, "cmd", "command"), exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
