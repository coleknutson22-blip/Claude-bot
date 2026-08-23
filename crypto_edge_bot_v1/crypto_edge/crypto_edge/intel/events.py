"""Scheduled token-event calendar (unlocks, upgrades, governance votes).

Stored with observed_at so a backtest can only see a calendar entry from the
moment we actually learned of it. Upcoming high-risk events suppress new
entries within a configurable window rather than forcing an exit.
"""
from __future__ import annotations

from ..storage.repo import Repo


class TokenEventCalendar:
    def __init__(self, repo: Repo, block_hours: int = 12) -> None:
        self.repo = repo
        self.block_hours = block_hours

    def add(self, symbol: str, kind: str, event_time: int, observed_at: int,
            description: str = "", source: str = "", severity: int = 2) -> None:
        self.repo.add_token_event({
            "symbol": symbol, "kind": kind, "event_time": event_time,
            "observed_at": observed_at, "description": description,
            "source": source, "severity": severity})

    def blocking_event(self, symbol: str, as_of_ms: int) -> dict | None:
        """A high-severity event inside the forward window blocks new entries."""
        window = self.block_hours * 3_600_000
        for ev in self.repo.token_events_visible(symbol, as_of_ms, window):
            if int(ev.get("severity", 0)) >= 2:
                return ev
        return None
