"""Durable Telegram outbox: PENDING -> SENT.

DEFECT UNDER TEST
-----------------
The previous implementation claimed a dedupe key by INSERTing into
`telegram_outbox` *before* calling the transport. A failed delivery therefore
left a row that was indistinguishable from a successful one, so:

  * the message was never retried -- the key was already "taken"
  * a restart could not recover it -- the text was never stored
  * an entry/exit/circuit-breaker alert could be lost in silence

These tests pin the corrected behaviour: nothing reaches SENT until the
transport confirms it, everything short of that stays recoverable, and a
delivered message is deduplicated forever.

All of this is VERIFIED OFFLINE against an injected transport. Real delivery to
Telegram's servers still REQUIRES VERIFICATION ON THE USER'S MACHINE.
"""
import unittest

from crypto_edge.notify.telegram import TelegramNotifier
from crypto_edge.storage.repo import Repo
from crypto_edge.timeutils import now_ms
from helpers import open_repo, temp_repo


class ScriptedTransport:
    """Transport whose success/failure is controlled by the test."""

    def __init__(self, ok: bool = True):
        self.ok = ok
        self.sent: list[str] = []
        self.calls = 0

    def send(self, token, chat_id, text, timeout):
        self.calls += 1
        if not self.ok:
            return False, "simulated total outage"
        self.sent.append(text)
        return True, "ok"


def notifier(repo, transport, **kw):
    return TelegramNotifier("tok", "chat", repo, enabled=True, transport=transport,
                            sleep=lambda _: None, max_retries=kw.pop("max_retries", 2),
                            **kw)


class TestTotalFailure(unittest.TestCase):
    """1. Total Telegram failure."""

    def setUp(self):
        self.repo, self.path = temp_repo()

    def test_failed_delivery_is_not_marked_sent(self):
        t = ScriptedTransport(ok=False)
        n = notifier(self.repo, t)
        self.assertFalse(n.send("ENTRY SOL", dedupe_key="entry:sol:1", kind="entry"))
        row = self.repo.telegram_status("entry:sol:1")
        self.assertIsNotNone(row, "a failed send must still leave an outbox row")
        self.assertEqual(row["status"], "PENDING")
        self.assertEqual(row["sent_ms"], 0)
        self.assertFalse(self.repo.telegram_already_sent("entry:sol:1"))

    def test_failed_delivery_stores_the_payload_for_replay(self):
        t = ScriptedTransport(ok=False)
        n = notifier(self.repo, t)
        n.send("ENTRY SOL/USDT @ 142.31", dedupe_key="entry:sol:1", kind="entry")
        row = self.repo.telegram_status("entry:sol:1")
        self.assertEqual(row["text"], "ENTRY SOL/USDT @ 142.31")
        self.assertEqual(row["kind"], "entry")
        self.assertIn("outage", row["last_error"])

    def test_every_transport_attempt_is_exhausted_before_giving_up(self):
        t = ScriptedTransport(ok=False)
        n = notifier(self.repo, t, max_retries=3)
        n.send("x", dedupe_key="k", kind="entry")
        self.assertEqual(t.calls, 3)

    def test_failure_shows_up_in_health(self):
        t = ScriptedTransport(ok=False)
        n = notifier(self.repo, t)
        n.send("x", dedupe_key="k", kind="entry")
        self.assertEqual(n.health()["outbox_pending"], 1)
        self.assertEqual(n.pending_count(), 1)


class TestRecovery(unittest.TestCase):
    """2. Later recovery."""

    def setUp(self):
        self.repo, self.path = temp_repo()

    def test_pending_message_is_delivered_once_telegram_returns(self):
        t = ScriptedTransport(ok=False)
        n = notifier(self.repo, t, outbox_lease_s=0)
        n.send("ENTRY SOL", dedupe_key="entry:sol:1", kind="entry")
        self.assertEqual(t.sent, [])

        t.ok = True
        self.assertEqual(n.flush_pending(), 1)
        self.assertEqual(t.sent, ["ENTRY SOL"], "the original text must be replayed")
        self.assertEqual(self.repo.telegram_status("entry:sol:1")["status"], "SENT")
        self.assertEqual(n.recovered_count, 1)

    def test_recovered_message_is_not_sent_twice(self):
        t = ScriptedTransport(ok=False)
        n = notifier(self.repo, t, outbox_lease_s=0)
        n.send("ENTRY SOL", dedupe_key="entry:sol:1", kind="entry")
        t.ok = True
        n.flush_pending()
        n.flush_pending()
        n.flush_pending()
        self.assertEqual(len(t.sent), 1, "a recovered message must not repeat")

    def test_flush_is_a_no_op_when_nothing_is_pending(self):
        t = ScriptedTransport(ok=True)
        n = notifier(self.repo, t, outbox_lease_s=0)
        n.send("hi", dedupe_key="k", kind="entry")
        t.sent.clear()
        self.assertEqual(n.flush_pending(), 0)
        self.assertEqual(t.sent, [])

    def test_still_failing_message_stays_pending_after_a_flush(self):
        t = ScriptedTransport(ok=False)
        n = notifier(self.repo, t, outbox_lease_s=0)
        n.send("x", dedupe_key="k", kind="entry")
        n.flush_pending()
        self.assertEqual(self.repo.telegram_status("k")["status"], "PENDING")
        self.assertGreaterEqual(self.repo.telegram_status("k")["attempts"], 2)

    def test_attempt_budget_eventually_parks_the_message_as_failed(self):
        """An undeliverable message must stop consuming cycles, but must stay
        visible rather than vanishing."""
        t = ScriptedTransport(ok=False)
        n = notifier(self.repo, t, outbox_lease_s=0, outbox_max_attempts=3)
        n.send("x", dedupe_key="k", kind="entry")
        for _ in range(10):
            n.flush_pending()
        row = self.repo.telegram_status("k")
        self.assertEqual(row["status"], "FAILED")
        self.assertEqual(n.health()["outbox_failed"], 1)
        self.assertIsNotNone(row["text"], "the payload stays on disk for audit")


class TestRestartBetweenFailureAndRecovery(unittest.TestCase):
    """3. Restart between failure and recovery."""

    def setUp(self):
        self.repo, self.path = temp_repo()

    def test_unsent_message_survives_a_process_death_and_is_delivered(self):
        # --- process 1: Telegram is down, the send fails --------------------
        down = ScriptedTransport(ok=False)
        n1 = notifier(self.repo, down, outbox_lease_s=0)
        n1.send("🔴 CIRCUIT BREAKER tripped", dedupe_key="cb:2026-08-23",
                kind="circuit_breaker")
        self.assertEqual(down.sent, [])
        self.repo.conn.close()          # ungraceful death

        # --- process 2: fresh repo, fresh notifier, Telegram back up --------
        repo2 = open_repo(self.path)
        up = ScriptedTransport(ok=True)
        n2 = notifier(repo2, up, outbox_lease_s=0)
        self.assertEqual(n2.flush_pending(), 1)
        self.assertEqual(up.sent, ["🔴 CIRCUIT BREAKER tripped"],
                         "a critical message must not be lost to a restart")
        self.assertEqual(repo2.telegram_status("cb:2026-08-23")["status"], "SENT")

    def test_restart_does_not_resend_what_was_already_delivered(self):
        up = ScriptedTransport(ok=True)
        n1 = notifier(self.repo, up, outbox_lease_s=0)
        n1.send("EXIT SOL", dedupe_key="exit:trd-1", kind="exit")
        self.repo.conn.close()

        repo2 = open_repo(self.path)
        up2 = ScriptedTransport(ok=True)
        n2 = notifier(repo2, up2, outbox_lease_s=0)
        self.assertEqual(n2.flush_pending(), 0)
        n2.send("EXIT SOL", dedupe_key="exit:trd-1", kind="exit")
        self.assertEqual(up2.sent, [], "delivered messages stay deduped across restart")

    def test_the_original_defect_would_have_lost_this_message(self):
        """Regression guard stated as the defect it replaces.

        Old behaviour: claim-then-send left the key marked as taken, so after a
        failure the same message could never be delivered by anyone, ever.
        """
        down = ScriptedTransport(ok=False)
        n1 = notifier(self.repo, down, outbox_lease_s=0)
        n1.send("ENTRY BTC", dedupe_key="entry:btc:9", kind="entry")

        self.repo.conn.close()
        repo2 = open_repo(self.path)
        up = ScriptedTransport(ok=True)
        n2 = notifier(repo2, up, outbox_lease_s=0)
        # a plain re-send (not just a flush) must also be allowed through
        self.assertTrue(n2.send("ENTRY BTC", dedupe_key="entry:btc:9", kind="entry"))
        self.assertEqual(up.sent, ["ENTRY BTC"])


class TestDuplicateSuppression(unittest.TestCase):
    """4. Duplicate suppression after successful delivery."""

    def setUp(self):
        self.repo, self.path = temp_repo()

    def test_second_send_after_success_is_suppressed(self):
        t = ScriptedTransport(ok=True)
        n = notifier(self.repo, t)
        self.assertTrue(n.send("a", dedupe_key="entry:1", kind="entry"))
        self.assertTrue(n.send("a", dedupe_key="entry:1", kind="entry"))
        self.assertTrue(n.send("a", dedupe_key="entry:1", kind="entry"))
        self.assertEqual(len(t.sent), 1)
        self.assertEqual(n.suppressed_count, 2)

    def test_sent_state_is_terminal_even_if_the_text_differs(self):
        t = ScriptedTransport(ok=True)
        n = notifier(self.repo, t)
        n.send("first wording", dedupe_key="daily:2026-08-23", kind="daily_report")
        n.send("second wording", dedupe_key="daily:2026-08-23", kind="daily_report")
        self.assertEqual(t.sent, ["first wording"])

    def test_an_in_flight_send_blocks_a_concurrent_duplicate(self):
        """The lease is the anti-spam guard: while one attempt holds the key,
        a second caller for the same key must not also hit the transport."""
        t = ScriptedTransport(ok=True)
        n = notifier(self.repo, t, outbox_lease_s=300)
        # simulate an attempt that claimed the key and has not reported back
        self.assertEqual(self.repo.claim_telegram("entry:1", "entry", "a",
                                                  lease_ms=300_000),
                         Repo.CLAIMED)
        self.assertFalse(n.send("a", dedupe_key="entry:1", kind="entry"))
        self.assertEqual(t.calls, 0, "a duplicate must not reach the transport")

    def test_distinct_keys_are_unaffected(self):
        t = ScriptedTransport(ok=True)
        n = notifier(self.repo, t)
        n.send("a", dedupe_key="entry:1", kind="entry")
        n.send("b", dedupe_key="entry:2", kind="entry")
        self.assertEqual(len(t.sent), 2)

    def test_undeduped_messages_are_never_written_to_the_outbox(self):
        t = ScriptedTransport(ok=True)
        n = notifier(self.repo, t)
        n.send("heartbeat", kind="heartbeat")
        n.send("heartbeat", kind="heartbeat")
        self.assertEqual(len(t.sent), 2, "heartbeats are intentionally repeatable")
        self.assertEqual(self.repo.telegram_outbox_counts(), {})


class TestOutboxHousekeeping(unittest.TestCase):
    def setUp(self):
        self.repo, self.path = temp_repo()

    def test_pruning_removes_delivered_rows_only(self):
        t = ScriptedTransport(ok=True)
        n = notifier(self.repo, t, outbox_lease_s=0)
        n.send("delivered", dedupe_key="sent:1", kind="entry")
        t.ok = False
        n.send("undelivered", dedupe_key="pending:1", kind="entry")

        self.repo.prune_telegram_outbox(now_ms() + 60_000)
        self.assertIsNone(self.repo.telegram_status("sent:1"))
        self.assertIsNotNone(self.repo.telegram_status("pending:1"),
                             "an undelivered message must never be pruned away")

    def test_disabled_notifier_writes_no_outbox_rows(self):
        t = ScriptedTransport(ok=True)
        n = TelegramNotifier("", "", self.repo, enabled=True, transport=t)
        self.assertFalse(n.send("x", dedupe_key="k", kind="entry"))
        self.assertEqual(self.repo.telegram_outbox_counts(), {})
        self.assertEqual(n.flush_pending(), 0)

    def test_notifier_without_a_repo_still_sends(self):
        t = ScriptedTransport(ok=True)
        n = TelegramNotifier("tok", "chat", None, enabled=True, transport=t,
                             sleep=lambda _: None)
        self.assertTrue(n.send("x", dedupe_key="k", kind="entry"))
        self.assertEqual(len(t.sent), 1)

    def test_flush_never_raises_even_if_the_database_is_gone(self):
        t = ScriptedTransport(ok=True)
        n = notifier(self.repo, t, outbox_lease_s=0)
        self.repo.conn.close()
        self.assertEqual(n.flush_pending(), 0)   # must not raise


class TestSchemaMigration(unittest.TestCase):
    def test_v1_outbox_rows_migrate_to_sent(self):
        """A pre-existing v1 row means "key claimed"; we cannot tell whether it
        was delivered, so it migrates to SENT. Suppressing one possibly-delivered
        old alert is safe; replaying a backlog of stale alerts on upgrade is not.
        """
        import sqlite3
        import tempfile
        from pathlib import Path

        from crypto_edge.storage import db

        path = Path(tempfile.mkdtemp()) / "v1.db"
        conn = sqlite3.connect(str(path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO meta(key, value) VALUES('schema_version', '1');
            CREATE TABLE telegram_outbox (
                dedupe_key TEXT PRIMARY KEY,
                sent_ms INTEGER NOT NULL,
                kind TEXT NOT NULL DEFAULT '');
            INSERT INTO telegram_outbox VALUES('entry:old', 1700000000000, 'entry');
        """)
        db.init_db(conn)

        repo = Repo(conn)
        self.assertTrue(repo.telegram_already_sent("entry:old"))
        self.assertEqual(repo.telegram_status("entry:old")["status"], "SENT")
        self.assertEqual(
            int(conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]),
            db.SCHEMA_VERSION)

    def test_unknown_future_schema_version_is_still_refused(self):
        import sqlite3
        import tempfile
        from pathlib import Path

        from crypto_edge.storage import db

        path = Path(tempfile.mkdtemp()) / "v99.db"
        conn = sqlite3.connect(str(path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.executescript(
            "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            "INSERT INTO meta(key, value) VALUES('schema_version', '99');")
        with self.assertRaises(RuntimeError):
            db.init_db(conn)


if __name__ == "__main__":
    unittest.main()
