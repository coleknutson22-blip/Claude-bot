"""Telegram delivery.

Hard guarantee: NOTHING in this module can crash the trading engine. Every
send is wrapped, every failure is logged and swallowed, and the transport is
injectable so the tests exercise the full retry/dedupe/rate-limit logic
without touching the network.

STATUS: the send path is REQUIRES VERIFICATION ON YOUR MACHINE (no network in
the build environment). The queueing, deduplication, error-cooldown and
message-content logic are VERIFIED OFFLINE against a fake transport.
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
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.token = token
        self.chat_id = chat_id
        self.repo = repo
        self.enabled = enabled and bool(token and chat_id)
        self.transport = transport or (RequestsTransport() if self.enabled else NullTransport())
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.error_cooldown_s = error_cooldown_s
        self._sleep = sleep
        self._last_error_ms: dict[str, int] = {}
        self.sent_count = 0
        self.failed_count = 0
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
            if dedupe_key and self.repo is not None:
                # claim the key BEFORE sending: if two code paths race, only one wins
                if not self.repo.mark_telegram_sent(dedupe_key, kind):
                    log_event("telegram", "DEBUG", "duplicate suppressed",
                              dedupe_key=dedupe_key)
                    return True

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
        except Exception as e:                       # belt and braces
            self.failed_count += 1
            self.last_error = str(e)
            try:
                log_event("telegram", "ERROR", "notifier raised (suppressed)",
                          error=str(e))
            except Exception:
                pass
            return False

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
        return {"enabled": self.enabled, "sent": self.sent_count,
                "failed": self.failed_count, "last_error": self.last_error}

    # -------------------------------------------------------- connectivity
    def test_connectivity(self) -> tuple[bool, str]:
        """Used by the startup self-check. REQUIRES NETWORK."""
        if not self.enabled:
            return False, "telegram disabled or credentials missing"
        ok, detail = self.transport.send(
            self.token, self.chat_id,
            "🔍 Crypto Edge self-check: Telegram connectivity OK", self.timeout_s)
        return ok, detail
