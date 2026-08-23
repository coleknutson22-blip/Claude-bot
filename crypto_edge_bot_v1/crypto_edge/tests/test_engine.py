"""End-to-end engine tests against the deterministic fixture feed.

These exercise the real cycle: universe -> data -> context -> position
management -> signals -> ranking -> risk -> entry -> persistence.
"""
import unittest

import numpy as np

from crypto_edge.data.fixture_feed import FixtureFeed, make_series
from crypto_edge.engine import TradingEngine
from crypto_edge.notify.telegram import TelegramNotifier
from crypto_edge.timeutils import now_ms, tf_ms
from helpers import (breakout_closes, default_meta, engine_config, open_repo,
                     recent_start_ms, temp_repo, trend_closes)

HOUR = 3_600_000


class RecordingTransport:
    """Captures messages instead of sending them."""

    def __init__(self, fail: bool = False):
        self.sent = []
        self.fail = fail

    def send(self, token, chat_id, text, timeout):
        if self.fail:
            return False, "simulated failure"
        self.sent.append(text)
        return True, "ok"


def build_engine(closes_by_symbol, cfg=None, n_visible=None, repo=None,
                 transport=None, vol_spike=True):
    """Wire an engine to a fixture feed built from the given close series."""
    cfg = cfg or engine_config()
    repo = repo or temp_repo()[0]
    n = len(next(iter(closes_by_symbol.values())))
    start = recent_start_ms(n, "1h")

    series, markets = {}, {}
    for sym, closes in closes_by_symbol.items():
        vols = np.full(len(closes), 1000.0)
        if vol_spike:
            vols[-1] = 2600.0
        s1 = make_series(sym, "1h", closes, start, volume=vols)
        if n_visible is not None:
            s1 = make_series(sym, "1h", closes[:n_visible], start,
                             volume=vols[:n_visible])
        series[(sym, "1h")] = s1
        c4 = closes[::4] if n_visible is None else closes[:n_visible][::4]
        series[(sym, "4h")] = make_series(sym, "4h", c4, start, volume=4000.0)
        markets[sym] = default_meta(sym)

    feed = FixtureFeed(series, markets)
    transport = transport or RecordingTransport()
    notifier = TelegramNotifier("tok", "chat", repo, enabled=True,
                                transport=transport, sleep=lambda _: None)
    engine = TradingEngine(cfg, repo, feed, notifier)
    return engine, feed, transport, repo


class TestEngineEntry(unittest.TestCase):
    def setUp(self):
        self.closes = {
            "BTC/USDT": trend_closes(400, seed=4),
            "SOL/USDT": breakout_closes(400, seed=11),
        }

    def test_engine_opens_a_position_on_a_valid_breakout(self):
        engine, feed, transport, repo = build_engine(self.closes)
        engine.cycle()
        positions = engine.account.positions()
        self.assertEqual(len(positions), 1, "expected exactly one entry")
        p = positions[0]
        self.assertEqual(p.symbol, "SOL/USDT")
        self.assertGreater(p.qty, 0)
        self.assertLess(p.initial_stop, p.entry_fill_price)
        self.assertGreater(p.signal_score, 0)

    def test_entry_risk_respects_the_configured_budget(self):
        engine, *_ = build_engine(self.closes)
        engine.cycle()
        p = engine.account.positions()[0]
        equity = float(engine.account.state["starting_equity"])
        max_risk = equity * engine.cfg.risk.risk_per_trade_pct / 100.0
        self.assertLessEqual(p.risk_amount, max_risk + 1e-6)

    def test_entry_sends_a_telegram_message(self):
        engine, feed, transport, repo = build_engine(self.closes)
        engine.cycle()
        entries = [m for m in transport.sent if "ENTRY" in m]
        self.assertEqual(len(entries), 1)
        self.assertIn("SOL/USDT", entries[0])
        self.assertIn("Stop:", entries[0])

    def test_same_candle_not_re_entered_on_a_second_cycle(self):
        engine, *_ = build_engine(self.closes)
        engine.cycle()
        engine.cycle()
        self.assertEqual(len(engine.account.positions()), 1)
        self.assertEqual(len(repo_trades(engine)), 0)

    def test_observations_recorded_for_rejections(self):
        engine, *_ = build_engine(self.closes)
        engine.cycle()
        obs = engine.repo.get_observations()
        self.assertGreater(len(obs), 0)
        decisions = {o["decision"] for o in obs}
        self.assertIn("ENTERED", decisions)


def repo_trades(engine):
    return engine.repo.get_trades()


def rebuild_feed(feed, closes_by_symbol, vol_spike_idx=None):
    """Rewrite every fixture series so the LAST bar is the just-closed bar.

    Appending a bar without shifting `start` leaves it in the future, where
    `drop_unclosed` correctly discards it -- the engine would never see it.
    """
    n = len(next(iter(closes_by_symbol.values())))
    start = recent_start_ms(n, "1h")
    for sym, closes in closes_by_symbol.items():
        vols = np.full(len(closes), 1000.0)
        if vol_spike_idx is not None:
            vols[vol_spike_idx] = 2600.0
        feed._series[(sym, "1h")] = make_series(sym, "1h", closes, start, volume=vols)
        feed._series[(sym, "4h")] = make_series(sym, "4h", closes[::4], start,
                                                volume=4000.0)
    return feed


class TestEngineExit(unittest.TestCase):
    def test_stop_exit_closes_position_and_reconciles_cash(self):
        closes = {
            "BTC/USDT": trend_closes(401, seed=4),
            "SOL/USDT": breakout_closes(400, seed=11),
        }
        # first 400 bars visible -> entry on bar 400
        sol = closes["SOL/USDT"]
        engine, feed, transport, repo = build_engine(
            {"BTC/USDT": closes["BTC/USDT"][:400], "SOL/USDT": sol})
        engine.cycle()
        self.assertEqual(len(engine.account.positions()), 1)
        pos = engine.account.positions()[0]
        start_equity = engine.cfg.execution.starting_equity

        # append a crash bar that trades well through the stop, shifting the
        # whole series back one bar so the new bar is the just-closed one
        crash_price = pos.current_stop * 0.90
        rebuild_feed(feed, {
            "BTC/USDT": closes["BTC/USDT"][:401],
            "SOL/USDT": np.append(sol, crash_price),
        }, vol_spike_idx=-2)

        engine.cycle()
        self.assertEqual(len(engine.account.positions()), 0, "stop should have closed it")
        trades = engine.repo.get_trades()
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertIn(t["exit_reason"], ("stop_loss", "stop_gap"))
        self.assertLess(t["net_pnl"], 0)
        # With the book flat, the full round trip must reconcile exactly:
        # every dollar of cash movement is explained by the ledger's net P&L.
        self.assertAlmostEqual(engine.account.cash() - start_equity,
                               t["net_pnl"], places=6)
        self.assertAlmostEqual(engine.account.equity({}), engine.account.cash(),
                               places=6)
        # and net must equal gross less the two frictions
        self.assertAlmostEqual(t["net_pnl"],
                               t["gross_pnl"] - t["fees"] - t["slippage_cost"],
                               places=6)

    def test_exit_sends_telegram_with_cost_breakdown(self):
        closes = {"BTC/USDT": trend_closes(400, seed=4),
                  "SOL/USDT": breakout_closes(400, seed=11)}
        engine, feed, transport, repo = build_engine(closes)
        engine.cycle()
        pos = engine.account.positions()[0]
        sol = closes["SOL/USDT"]
        rebuild_feed(feed, {
            "BTC/USDT": np.append(closes["BTC/USDT"], closes["BTC/USDT"][-1]),
            "SOL/USDT": np.append(sol, pos.current_stop * 0.9),
        }, vol_spike_idx=-2)
        engine.cycle()
        exits = [m for m in transport.sent if "LOSS" in m or "PROFIT" in m]
        self.assertTrue(exits, "an exit notification should have been sent")
        msg = exits[0]
        for field in ("Gross:", "Fees:", "Slippage:", "NET", "Exit reason:"):
            self.assertIn(field, msg)


class TestEngineSafety(unittest.TestCase):
    def test_stale_data_suspends_signal_evaluation(self):
        # series that ended long ago -> stale
        closes = {"BTC/USDT": trend_closes(400, seed=4),
                  "SOL/USDT": breakout_closes(400, seed=11)}
        cfg = engine_config()
        repo, _ = temp_repo()
        series, markets = {}, {}
        old_start = 1_600_000_000_000        # 2020
        for sym, c in closes.items():
            series[(sym, "1h")] = make_series(sym, "1h", c, old_start, volume=1000.0)
            series[(sym, "4h")] = make_series(sym, "4h", c[::4], old_start, volume=4000.0)
            markets[sym] = default_meta(sym)
        feed = FixtureFeed(series, markets)
        transport = RecordingTransport()
        notifier = TelegramNotifier("t", "c", repo, enabled=True, transport=transport,
                                    sleep=lambda _: None)
        engine = TradingEngine(cfg, repo, feed, notifier)
        engine.cycle()
        self.assertEqual(len(engine.account.positions()), 0,
                         "must not trade on stale data")
        self.assertTrue(any("STALE DATA" in m for m in transport.sent))

    def test_missing_btc_data_fails_closed(self):
        closes = {"SOL/USDT": breakout_closes(400, seed=11)}
        engine, feed, transport, repo = build_engine(closes)
        engine.cycle()
        self.assertEqual(len(engine.account.positions()), 0)

    def test_halted_account_blocks_new_entries(self):
        closes = {"BTC/USDT": trend_closes(400, seed=4),
                  "SOL/USDT": breakout_closes(400, seed=11)}
        engine, feed, transport, repo = build_engine(closes)
        repo.set_halt(True, "MAX DRAWDOWN kill switch: manual test")
        engine.cycle()
        self.assertEqual(len(engine.account.positions()), 0,
                         "entries must be suppressed while halted")

    def test_drawdown_breaker_trips_and_persists(self):
        closes = {"BTC/USDT": trend_closes(400, seed=4),
                  "SOL/USDT": breakout_closes(400, seed=11)}
        engine, feed, transport, repo = build_engine(closes)
        # simulate a catastrophic loss of capital
        repo.update_account(cash=7_000.0, peak_equity=10_000.0,
                            daily_start_equity=7_000.0)
        engine.cycle()
        acct = repo.get_account()
        self.assertTrue(int(acct["halted"]))
        self.assertIn("DRAWDOWN", acct["halt_reason"])
        self.assertTrue(any("CIRCUIT BREAKER" in m for m in transport.sent))


class TestEngineRestart(unittest.TestCase):
    def test_open_position_survives_engine_restart(self):
        closes = {"BTC/USDT": trend_closes(400, seed=4),
                  "SOL/USDT": breakout_closes(400, seed=11)}
        repo, path = temp_repo()
        engine, feed, transport, _ = build_engine(closes, repo=repo)
        engine.cycle()
        self.assertEqual(len(engine.account.positions()), 1)
        before = engine.account.positions()[0]
        cash_before = engine.account.cash()

        repo.conn.close()
        repo2 = open_repo(path)
        engine2, feed2, _, _ = build_engine(closes, repo=repo2)
        after = engine2.account.positions()
        self.assertEqual(len(after), 1)
        self.assertEqual(after[0].symbol, before.symbol)
        self.assertAlmostEqual(after[0].qty, before.qty, places=9)
        self.assertAlmostEqual(engine2.account.cash(), cash_before, places=9)

        # and the same candle must not be traded again after restart
        engine2.cycle()
        self.assertEqual(len(engine2.account.positions()), 1)


if __name__ == "__main__":
    unittest.main()
