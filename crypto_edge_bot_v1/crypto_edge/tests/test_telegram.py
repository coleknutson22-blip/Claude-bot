"""Notification tests.

Two separate concerns:
  * formatters  -- pure string builders, fully VERIFIED OFFLINE
  * notifier    -- retry / dedupe / fail-safe LOGIC, VERIFIED OFFLINE via an
                   injected transport. Actual delivery to Telegram's servers
                   REQUIRES VERIFICATION ON THE USER'S MACHINE.
"""
import unittest

from crypto_edge.notify import formatters as F
from crypto_edge.notify.telegram import NullTransport, TelegramNotifier
from helpers import temp_repo


class RecordingTransport:
    def __init__(self, results=None):
        self.sent = []
        self.calls = 0
        self.results = list(results) if results else None

    def send(self, token, chat_id, text, timeout):
        self.calls += 1
        self.sent.append(text)
        if self.results:
            return self.results.pop(0)
        return True, "ok"


class RaisingTransport:
    def send(self, token, chat_id, text, timeout):
        raise RuntimeError("network exploded")


class TestFormatters(unittest.TestCase):
    def test_entry_message_contains_every_required_field(self):
        msg = F.entry(symbol="SOL/USDT", side="long", entry_price=142.31,
                      qty=10.5, position_value=1494.26, pct_of_account=14.9,
                      stop=133.77, dollar_risk=50.0, pct_risk=0.5, score=72.4,
                      htf_regime="bull", btc_regime="bull", breadth=61.0,
                      reasons=["donchian breakout", "adx 27"], equity=10_000.0)
        for field in ("ENTRY", "SOL/USDT", "Entry:", "Qty:", "Value:", "Stop:",
                      "Risk:", "Score:", "Regime:", "Equity:"):
            self.assertIn(field, msg)

    def test_exit_message_shows_the_full_cost_breakdown(self):
        msg = F.exit_trade(symbol="SOL/USDT", entry_price=142.31, exit_price=133.9,
                           position_value=1494.26, held_s=9_000, gross=-88.3,
                           fees=2.2, slippage=1.9, net=-92.4, return_pct=-6.18,
                           account_return_pct=-0.92, exit_reason="stop_loss",
                           equity=9_907.6, total_pnl=-92.4)
        for field in ("LOSS", "Gross:", "Fees:", "Slippage:", "NET LOSS:",
                      "Exit reason:", "Account equity:"):
            self.assertIn(field, msg)

    def test_win_and_loss_are_visually_distinct(self):
        common = dict(symbol="X/USDT", entry_price=10.0, exit_price=11.0,
                      position_value=100.0, held_s=60, gross=10.0, fees=0.1,
                      slippage=0.1, return_pct=9.8, account_return_pct=0.1,
                      exit_reason="trend_invalidation", equity=10_100.0,
                      total_pnl=9.8)
        win = F.exit_trade(net=9.8, **common)
        loss = F.exit_trade(**{**common, "net": -9.8})
        self.assertIn("PROFIT", win)
        self.assertIn("LOSS", loss)
        self.assertNotEqual(win, loss)

    def test_stop_update_reports_locked_in_profit_only_when_positive(self):
        up = F.stop_update(symbol="S/USDT", prev_stop=9.0, new_stop=11.0,
                           price=12.0, entry_price=10.0, qty=5.0, kind="chandelier")
        self.assertIn("Locked-in profit", up)
        below = F.stop_update(symbol="S/USDT", prev_stop=8.0, new_stop=9.0,
                              price=10.5, entry_price=10.0, qty=5.0, kind="chandelier")
        self.assertNotIn("Locked-in profit", below)

    def test_circuit_breaker_states_the_action_taken(self):
        msg = F.circuit_breaker(kind="MAX DRAWDOWN", reason="drawdown 20.4%",
                                equity=7_960.0)
        self.assertIn("CIRCUIT BREAKER", msg)
        self.assertIn("Action:", msg)

    def test_formatters_are_pure(self):
        kw = dict(mode="PAPER", exchange="binance", equity=10_000.0,
                  cash=10_000.0, open_positions=0, strategy="trend_breakout",
                  version="abc123", universe_size=40, telegram_ok=True)
        self.assertEqual(F.bot_start(**kw), F.bot_start(**kw))

    def test_start_message_states_paper_mode_unmistakably(self):
        msg = F.bot_start(mode="PAPER", exchange="binance", equity=10_000.0,
                          cash=10_000.0, open_positions=0,
                          strategy="trend_breakout", version="abc123",
                          universe_size=40, telegram_ok=True)
        self.assertIn("PAPER TRADING", msg)
        self.assertIn("no real orders", msg)


class TestNotifierFailSafe(unittest.TestCase):
    def test_transport_exception_never_propagates(self):
        n = TelegramNotifier("tok", "chat", enabled=True,
                             transport=RaisingTransport(), sleep=lambda _: None)
        self.assertFalse(n.send("hello"))          # must not raise
        self.assertEqual(n.failed_count, 1)

    def test_retries_then_succeeds(self):
        t = RecordingTransport([(False, "500"), (False, "500"), (True, "ok")])
        n = TelegramNotifier("tok", "chat", enabled=True, transport=t,
                             sleep=lambda _: None)
        self.assertTrue(n.send("hi"))
        self.assertEqual(t.calls, 3)

    def test_gives_up_after_max_retries(self):
        t = RecordingTransport([(False, "500")] * 9)
        n = TelegramNotifier("tok", "chat", enabled=True, transport=t,
                             max_retries=3, sleep=lambda _: None)
        self.assertFalse(n.send("hi"))
        self.assertEqual(t.calls, 3)
        self.assertEqual(n.failed_count, 1)

    def test_disabled_notifier_sends_nothing(self):
        t = RecordingTransport()
        n = TelegramNotifier("", "", enabled=True, transport=t)
        self.assertFalse(n.send("hi"))
        self.assertEqual(t.calls, 0)

    def test_null_transport_is_a_silent_no_op(self):
        # Deliberate: when Telegram is switched off, a "send" is a successful
        # no-op rather than an error, so nothing upstream treats it as a fault.
        ok, detail = NullTransport().send("t", "c", "x", 1.0)
        self.assertTrue(ok)
        self.assertEqual(detail, "disabled")


class TestNotifierDedupe(unittest.TestCase):
    def setUp(self):
        self.repo, self.path = temp_repo()

    def test_same_dedupe_key_sends_once(self):
        t = RecordingTransport()
        n = TelegramNotifier("tok", "chat", self.repo, enabled=True,
                             transport=t, sleep=lambda _: None)
        self.assertTrue(n.send("entry SOL", dedupe_key="entry:SOL:123", kind="entry"))
        self.assertTrue(n.send("entry SOL", dedupe_key="entry:SOL:123", kind="entry"))
        self.assertEqual(t.calls, 1, "the second send must be suppressed")

    def test_different_keys_both_send(self):
        t = RecordingTransport()
        n = TelegramNotifier("tok", "chat", self.repo, enabled=True,
                             transport=t, sleep=lambda _: None)
        n.send("a", dedupe_key="entry:SOL:1", kind="entry")
        n.send("b", dedupe_key="entry:SOL:2", kind="entry")
        self.assertEqual(t.calls, 2)

    def test_dedupe_survives_restart(self):
        t1 = RecordingTransport()
        n1 = TelegramNotifier("tok", "chat", self.repo, enabled=True,
                              transport=t1, sleep=lambda _: None)
        n1.send("x", dedupe_key="daily:2026-08-22", kind="daily")
        from helpers import open_repo
        t2 = RecordingTransport()
        n2 = TelegramNotifier("tok", "chat", open_repo(self.path), enabled=True,
                              transport=t2, sleep=lambda _: None)
        n2.send("x", dedupe_key="daily:2026-08-22", kind="daily")
        self.assertEqual(t2.calls, 0, "duplicate must stay suppressed across restart")


class TestErrorCooldown(unittest.TestCase):
    def test_repeat_error_is_rate_limited(self):
        t = RecordingTransport()
        n = TelegramNotifier("tok", "chat", enabled=True, transport=t,
                             error_cooldown_s=900, sleep=lambda _: None)
        self.assertTrue(n.send_error("fetch", "connection refused"))
        self.assertFalse(n.send_error("fetch", "connection refused"))
        self.assertEqual(t.calls, 1, "one failing loop must not spam the chat")

    def test_distinct_errors_are_not_suppressed(self):
        t = RecordingTransport()
        n = TelegramNotifier("tok", "chat", enabled=True, transport=t,
                             sleep=lambda _: None)
        n.send_error("fetch", "connection refused")
        n.send_error("order", "insufficient cash")
        self.assertEqual(t.calls, 2)


if __name__ == "__main__":
    unittest.main()
