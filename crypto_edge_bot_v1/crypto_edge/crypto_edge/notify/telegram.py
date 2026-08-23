"""Telegram delivery.

Hard guarantee: NOTHING in this module can crash the trading engine. Every
send is wrapped, every failure is logged and swallowed, and the transport is
injectable so the tests exercise the full retry/dedupe/rate-limit logic
without touching the network.

DELIVERY MODEL
--------------
Deduplicated messages go through a durable outbox with an explicit state
machine, held in SQLite:

    (no row) --claim--> PENDING --transport says OK--> SENT   [terminal]
                           |
                           +--transport failed--------> PENDING (retryable)
                           +--attempt budget spent----> FAILED [terminal]

The ordering matters. The row is written PENDING *before* the transport is
called, so a crash mid-send leaves evidence; it is only flipped to SENT once
the transport confirms delivery, so a failed send is never mistaken for a
delivered one. That gives all four properties at once:

  * failed deliveries stay recoverable  -- the row stays PENDING with its text
  * successful deliveries stay deduped  -- SENT is terminal and permanent
  * a restart loses nothing critical    -- `flush_pending()` replays PENDING
                                           rows from disk on the next cycle
  * duplicates cannot spam the chat     -- claiming takes a time-limited lease
                                           via an atomic compare-and-swap, so a
                                           second caller for the same key is
                                           suppressed while one send is in
                                           flight, and forever after it succeeds

STATUS: the send path is REQUIRES VERIFICATION ON YOUR MACHINE (no network in
the build environment). The queueing, outbox state machine, deduplication,
restart recovery, error-cooldown and message-content logic are VERIFIED
OFFLINE against a fake transport.
"""
from __future__ import annotations

import time
from typing import Callable, Protocol

from ..logging_setup import log_event
from ..storage.repo import Repo
from ..timeutils import now_ms

API = "https://api.telegram.org"


class Transport(Protocol):
    def send(self, token: str, chat_id: str, text: str, timeout: float) -> tuple[bool, str]: ...


class RequestsTransport:
    """Real HTTP transport. REQUIRES NETWORK."""

    def send(self, token: str, chat_id: str, text: str,
             timeout: float) -> tuple[bool, str]:
        import requests
        try:
            r = requests.post(
                f"{API}/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text,
                      "disable_web_page_preview": True},
                timeout=timeout)
            if r.status_code == 200:
                return True, "ok"
            return False, f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            return False, str(e)


class NullTransport:
    """Used when Telegram is disabled. Records nothing, sends nothing."""

    def send(self, token: str, chat_id: str, text: str,
             timeout: float) -> tuple[bool, str]:
        return True, "disabled"


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, repo: Repo | None = None,
                 enabled: bool = True, transport: Transport | None = None,
                 timeout_s: float = 10.0, max_retries: int = 3,
                 error_cooldown_s: int = 900,
                 sleep: Callable[[float], None] = time.sleep,
                 outbox_lease_s: int = 120, outbox_max_attempts: int = 12,
                 outbox_flush_limit: int = 20) -> None:
        self.token = token
        self.chat_id = chat_id
        self.repo = repo
        self.enabled = enabled and bool(token and chat_id)
        self.transport = transport or (RequestsTransport() if self.enabled else NullTransport())
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.error_cooldown_s = error_cooldown_s
        self._sleep = sleep
        self.outbox_lease_ms = max(0, int(outbox_lease_s * 1000))
        self.outbox_max_attempts = max(1, int(outbox_max_attempts))
        self.outbox_flush_limit = max(1, int(outbox_flush_limit))
        self._last_error_ms: dict[str, int] = {}
        self.sent_count = 0
        self.failed_count = 0
        self.suppressed_count = 0
        self.recovered_count = 0
        self.last_error: str = ""

    # ------------------------------------------------------------- sending
    def send(self, text: str, *, dedupe_key: str | None = None,
             kind: str = "") -> bool:
        """Returns True if delivered (or intentionally suppressed as duplicate).

        Never raises. A Telegram outage degrades notifications, not trading.
        """
        try:
            if not self.enabled:
                log_event("telegram", "DEBUG", "telegram disabled; message dropped",
                          kind=kind)
                return False

            if dedupe_key is None or self.repo is None:
                return self._deliver(text, kind=kind)

            # Reserve the key FIRST, but only as PENDING -- never as delivered.
            claim = self.repo.claim_telegram(
                dedupe_key, kind=kind, text=text,
                lease_ms=self.outbox_lease_ms,
                max_attempts=self.outbox_max_attempts)
            if claim == Repo.ALREADY_SENT:
                self.suppressed_count += 1
                log_event("telegram", "DEBUG", "duplicate suppressed",
                          dedupe_key=dedupe_key)
                return True
            if claim == Repo.IN_FLIGHT:
                self.suppressed_count += 1
                log_event("telegram", "DEBUG", "send already in flight; suppressed",
                          dedupe_key=dedupe_key)
                return False
            if claim == Repo.GAVE_UP:
                log_event("telegram", "ERROR", "message abandoned after max attempts",
                          dedupe_key=dedupe_key, kind=kind)
                return False

            return self._attempt_claimed(dedupe_key, text, kind)
        except Exception as e:                       # belt and braces
            self.failed_count += 1
            self.last_error = str(e)
            try:
                log_event("telegram", "ERROR", "notifier raised (suppressed)",
                          error=str(e))
            except Exception:
                pass
            return False

    def _attempt_claimed(self, dedupe_key: str, text: str, kind: str) -> bool:
        """Deliver a message whose outbox row we already hold the lease on.

        Success flips the row to SENT; failure releases the lease and leaves it
        PENDING, which is what keeps the message recoverable across a restart.
        """
        if self._deliver(text, kind=kind):
            self.repo.mark_telegram_delivered(dedupe_key)
            return True
        self.repo.release_telegram_claim(
            dedupe_key, self.last_error, max_attempts=self.outbox_max_attempts)
        log_event("telegram", "WARNING", "delivery failed; message left pending",
                  dedupe_key=dedupe_key, kind=kind, error=self.last_error)
        return False

    def _deliver(self, text: str, kind: str = "") -> bool:
        """Transport call plus retry/backoff. No outbox bookkeeping here."""
        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            ok, detail = self.transport.send(self.token, self.chat_id, text,
                                             self.timeout_s)
            if ok:
                self.sent_count += 1
                log_event("telegram", "INFO", "sent", kind=kind,
                          attempt=attempt, chars=len(text))
                return True
            log_event("telegram", "WARNING", "send failed",
                      kind=kind, attempt=attempt, error=detail)
            self.last_error = detail
            if attempt < self.max_retries:
                self._sleep(delay)
                delay *= 2
        self.failed_count += 1
        return False

    # ----------------------------------------------------------- recovery
    def flush_pending(self, limit: int | None = None) -> int:
        """Retry messages that were written but never delivered.

        Safe to call every cycle and safe to call after a restart: the text was
        persisted at claim time, so a process that died between "failed send"
        and "retry" still has the message on disk. Returns the number delivered.
        """
        if not self.enabled or self.repo is None:
            return 0
        delivered = 0
        try:
            ready_before = now_ms() - self.outbox_lease_ms
            rows = self.repo.pending_telegram(
                limit=limit or self.outbox_flush_limit,
                ready_before_ms=ready_before)
            for row in rows:
                key = row["dedupe_key"]
                if not row["text"]:
                    # pre-migration row with no payload: nothing to resend
                    self.repo.release_telegram_claim(
                        key, "no stored payload", max_attempts=0)
                    continue
                claim = self.repo.claim_telegram(
                    key, kind=row["kind"], text=row["text"],
                    lease_ms=self.outbox_lease_ms,
                    max_attempts=self.outbox_max_attempts)
                if claim != Repo.CLAIMED:
                    continue
                if self._attempt_claimed(key, row["text"], row["kind"]):
                    delivered += 1
                    self.recovered_count += 1
                    log_event("telegram", "INFO", "pending message recovered",
                              dedupe_key=key, kind=row["kind"])
        except Exception as e:                       # never break the cycle
            log_event("telegram", "ERROR", "outbox flush failed (suppressed)",
                      error=str(e))
        return delivered

    def pending_count(self) -> int:
        if self.repo is None:
            return 0
        try:
            return self.repo.telegram_outbox_counts().get("PENDING", 0)
        except Exception:
            return 0

    def send_error(self, where: str, message: str) -> bool:
        """Rate-limited error channel so one failing loop cannot spam the chat."""
        from .formatters import error as fmt_error
        key = f"{where}:{message[:60]}"
        last = self._last_error_ms.get(key, 0)
        now = now_ms()
        if now - last < self.error_cooldown_s * 1000:
            log_event("telegram", "DEBUG", "error suppressed by cooldown", where=where)
            return False
        self._last_error_ms[key] = now
        return self.send(fmt_error(where=where, message=message), kind="error")

    def health(self) -> dict:
        counts = {}
        if self.repo is not None:
            try:
                counts = self.repo.telegram_outbox_counts()
            except Exception:
                counts = {}
        return {"enabled": self.enabled, "sent": self.sent_count,
                "failed": self.failed_count, "suppressed": self.suppressed_count,
                "recovered": self.recovered_count,
                "outbox_pending": counts.get("PENDING", 0),
                "outbox_failed": counts.get("FAILED", 0),
                "last_error": self.last_error}

    # -------------------------------------------------------- connectivity
    def test_connectivity(self) -> tuple[bool, str]:
        """Used by the startup self-check. REQUIRES NETWORK."""
        if not self.enabled:
            return False, "telegram disabled or credentials missing"
        ok, detail = self.transport.send(
            self.token, self.chat_id,
            "🔍 Crypto Edge self-check: Telegram connectivity OK", self.timeout_s)
        return ok, detail
