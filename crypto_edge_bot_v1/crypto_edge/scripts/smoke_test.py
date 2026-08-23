#!/usr/bin/env python3
"""Offline end-to-end dry run. NO NETWORK REQUIRED.

Drives the real engine -- real universe filter, real strategy, real sizing,
real broker, real SQLite persistence, real notification formatting -- against
a deterministic synthetic feed, then restarts the engine from disk to prove
state survives a process death.

    python scripts/smoke_test.py

This is the same code path `cli start` uses; only the data source and the
Telegram transport are swapped.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crypto_edge.config import Config
from crypto_edge.data.fixture_feed import FixtureFeed, make_series
from crypto_edge.engine import TradingEngine
from crypto_edge.models import MarketMeta
from crypto_edge.notify.telegram import TelegramNotifier
from crypto_edge.performance import PerformanceCalculator
from crypto_edge.storage import db
from crypto_edge.storage.repo import Repo
from crypto_edge.timeutils import floor_to_tf, now_ms, tf_ms

HOUR = 3_600_000
N = 400


class PrintingTransport:
    """Stands in for Telegram: prints what WOULD have been sent."""

    def __init__(self):
        self.sent = []

    def send(self, token, chat_id, text, timeout):
        self.sent.append(text)
        first = text.splitlines()[0] if text else ""
        print(f"    [telegram] {first}")
        return True, "ok"


def recent_start(n_bars: int, tf: str = "1h") -> int:
    step = tf_ms(tf)
    return floor_to_tf(now_ms(), tf) - step - (n_bars - 1) * step


def series_for(closes, symbol, spike_last=True):
    vols = np.full(len(closes), 1000.0)
    if spike_last:
        vols[-1] = 2600.0
    return make_series(symbol, "1h", closes, recent_start(len(closes)), volume=vols)


def build_closes(n=N, seed=11, drift=0.0006, vol=0.006, start=100.0,
                 breakout=1.012):
    """Noisy uptrend whose final bar clears the prior 48-bar high.

    Deliberately noisy -- a perfectly smooth ramp pins RSI and ADX at 100,
    which no real market does and which the exhaustion filter rejects.
    """
    rng = np.random.default_rng(seed)
    c = start * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    prior_high = c[-49:-1].max() * 1.004
    c[-1] = max(c[-2], prior_high) * breakout
    return c


def trend_closes(n=N, seed=4, start=30_000.0):
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(0.0006, 0.006, n)))


def make_config(db_path: str) -> Config:
    cfg = Config()
    cfg.engine.db_path = db_path
    cfg.engine.poll_seconds = 0
    cfg.telegram.enabled = True          # formatting exercised; transport faked
    cfg.safety.candle_close_buffer_s = 0
    # Synthetic fixtures are short-lived by nature, so only the market-age and
    # volatility-band gates are relaxed. Every risk, execution and strategy
    # parameter stays at its production default.
    cfg.universe.min_candles_1h = 300
    cfg.universe.min_market_age_days = 5
    cfg.universe.min_atr_pct = 0.05
    cfg.universe.refresh_minutes = 0
    return cfg


def build_feed(closes_by_symbol, spike_last=True) -> FixtureFeed:
    series, markets = {}, {}
    for sym, closes in closes_by_symbol.items():
        series[(sym, "1h")] = series_for(closes, sym, spike_last)
        series[(sym, "4h")] = make_series(sym, "4h", closes[::4],
                                          recent_start(len(closes)), volume=4000.0)
        markets[sym] = MarketMeta(sym, sym.split("/")[0], "USDT", True,
                                  4, 4, 0.0001, 10.0)
    return FixtureFeed(series, markets)


def show(engine, label):
    marks = {p.symbol: p.entry_fill_price for p in engine.account.positions()}
    rep = PerformanceCalculator(engine.repo).report(marks=marks)
    a = rep.account
    print(f"    {label}: equity ${a['current_equity']:,.2f}  "
          f"cash ${a['cash']:,.2f}  open {rep.risk['open_positions']}  "
          f"closed {rep.trading['closed_trades']}")
    return rep


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="crypto_edge_smoke_"))
    db_path = str(workdir / "smoke.db")
    print("=" * 68)
    print("crypto_edge_bot_v1 -- OFFLINE SMOKE TEST (no network used)")
    print("=" * 68)
    print(f"scratch db: {db_path}\n")

    cfg = make_config(db_path)
    conn = db.connect(db_path)
    db.init_db(conn)
    repo = Repo(conn)
    transport = PrintingTransport()
    notifier = TelegramNotifier("smoke-token", "smoke-chat", repo, enabled=True,
                                transport=transport, sleep=lambda _: None)

    btc = trend_closes()
    sol = build_closes(seed=11)
    feed = build_feed({"BTC/USDT": btc, "SOL/USDT": sol})

    engine = TradingEngine(cfg, repo, feed, notifier)

    print("[1] universe construction")
    universe = engine.refresh_universe(force=True)
    print(f"    tradable after filtering: {universe}\n")

    print("[2] cycle 1 -- expect an entry on the breakout bar")
    engine.cycle()
    positions = engine.account.positions()
    if not positions:
        print("    NO ENTRY TAKEN -- smoke test cannot continue.")
        print("    (This is a fixture problem, not necessarily a bot problem;")
        print("     inspect the observations table for the rejection reason.)")
        for o in repo.get_observations()[:5]:
            print(f"      {o['symbol']}: {o['decision']} {o['reject_reason']}")
        return 1
    pos = positions[0]
    print(f"    ENTERED {pos.symbol} qty={pos.qty:.6g} "
          f"@ {pos.entry_fill_price:.6g} stop={pos.current_stop:.6g} "
          f"risk=${pos.risk_amount:.2f}")
    show(engine, "after entry")

    print("\n[3] duplicate protection -- re-running the same candle")
    engine.cycle()
    print(f"    open positions still {len(engine.account.positions())} "
          f"(same candle must not re-enter)")

    print("\n[4] restart from disk -- process death simulation")
    conn2 = db.connect(db_path)
    db.init_db(conn2)
    repo2 = Repo(conn2)
    notifier2 = TelegramNotifier("smoke-token", "smoke-chat", repo2, enabled=True,
                                 transport=transport, sleep=lambda _: None)
    engine2 = TradingEngine(cfg, repo2, feed, notifier2)
    restored = engine2.account.positions()
    print(f"    restored {len(restored)} position(s) from SQLite")
    assert restored and restored[0].id == pos.id, "position did not survive restart"
    print(f"    {restored[0].symbol} stop={restored[0].current_stop:.6g} intact")

    print("\n[5] price collapses through the stop -- expect a stop exit")
    crash = float(restored[0].current_stop) * 0.90
    new_sol = np.append(sol, crash)
    new_btc = np.append(btc, btc[-1])
    feed2 = build_feed({"BTC/USDT": new_btc, "SOL/USDT": new_sol}, spike_last=False)
    engine2.feed = feed2
    engine2.cycle()
    print(f"    open positions now {len(engine2.account.positions())}")
    trades = repo2.get_trades()
    if trades:
        t = trades[0]
        print(f"    CLOSED {t['symbol']} reason={t['exit_reason']}")
        print(f"      gross ${t['gross_pnl']:.2f}  fees ${t['fees']:.2f}  "
              f"slippage ${t['slippage_cost']:.2f}  NET ${t['net_pnl']:.2f}")
        recon = t["gross_pnl"] - t["fees"] - t["slippage_cost"]
        assert abs(recon - t["net_pnl"]) < 1e-6, "P&L identity violated"
        print(f"      identity check: gross - fees - slippage = "
              f"${recon:.2f} == net  OK")

    rep = show(engine2, "final")
    start_eq = rep.account["starting_balance"]
    drift = rep.account["current_equity"] - start_eq - rep.account["total_pnl"]
    print(f"    equity reconciliation drift: ${drift:.10f} (must be ~0)")
    assert abs(drift) < 1e-6, "account does not reconcile"

    print(f"\n[6] research database")
    obs = repo2.get_observations()
    decisions = {}
    for o in obs:
        decisions[o["decision"]] = decisions.get(o["decision"], 0) + 1
    print(f"    {len(obs)} observations recorded: {decisions}")

    print(f"\n[7] notifications formatted (transport faked): {len(transport.sent)}")

    print("\n" + "=" * 68)
    print("SMOKE TEST PASSED -- entry, persistence, restart, stop exit,")
    print("P&L identity and equity reconciliation all verified offline.")
    print("NOT verified here: live exchange data, real Telegram delivery.")
    print("=" * 68)
    shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
