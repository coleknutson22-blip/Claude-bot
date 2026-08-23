"""Market-data interface.

Two implementations exist:
  * CCXTFeed      -- live public exchange data (REQUIRES NETWORK)
  * FixtureFeed   -- deterministic in-memory data for tests and offline smoke runs

The engine only ever talks to this interface, which is why the entire trading
core can be exercised and verified with no internet connection.
"""
from __future__ import annotations

from typing import Protocol

from ..models import MarketMeta, Quote, Series


class DataUnavailable(Exception):
    """Raised when data cannot be trusted. Callers must fail closed."""


class MarketFeed(Protocol):
    name: str

    def load_markets(self) -> dict[str, MarketMeta]: ...
    def fetch_tickers(self) -> dict[str, dict]: ...
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> Series: ...
    def fetch_quote(self, symbol: str) -> Quote | None: ...
    def server_time_ms(self) -> int | None: ...
