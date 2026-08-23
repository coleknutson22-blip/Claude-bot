"""Live exchange data via CCXT, public endpoints only.

NO API KEY IS EVER READ OR PASSED HERE. The client is constructed without
credentials, which means it is structurally incapable of placing an order even
if some future bug tried to.

STATUS: REQUIRES VERIFICATION ON YOUR MACHINE. The build environment has no
outbound network, so this class has been written and import-checked but its
network calls have NOT been executed. `scripts/selfcheck.py` exercises them.
"""
from __future__ import annotations

import time

from ..logging_setup import log_event
from ..models import MarketMeta, Quote, Series
from ..timeutils import now_ms
from .feed import DataUnavailable


class CCXTFeed:
    def __init__(self, exchange: str = "binance", quote: str = "USDT",
                 rate_limit_ms: int = 250, timeout_s: int = 20) -> None:
        try:
            import ccxt  # imported lazily so the core runs without the dependency
        except ImportError as e:
            raise DataUnavailable(
                "ccxt is not installed -- run: pip install -r requirements.txt") from e
        if not hasattr(ccxt, exchange):
            raise DataUnavailable(f"unknown exchange '{exchange}'")
        self.name = exchange
        self.quote = quote
        self.rate_limit_ms = rate_limit_ms
        # No apiKey / secret. Public data only. This is deliberate.
        self.client = getattr(ccxt, exchange)({
            "enableRateLimit": True,
            "timeout": timeout_s * 1000,
            "options": {"defaultType": "spot"},
        })
        self._markets: dict[str, MarketMeta] = {}

    def _sleep(self) -> None:
        time.sleep(self.rate_limit_ms / 1000.0)

    # ------------------------------------------------------------- markets
    def load_markets(self) -> dict[str, MarketMeta]:
        try:
            raw = self.client.load_markets()
        except Exception as e:
            raise DataUnavailable(f"load_markets failed: {e}") from e
        out: dict[str, MarketMeta] = {}
        for sym, m in raw.items():
            if not m.get("spot", False):
                continue
            if m.get("quote") != self.quote:
                continue
            precision = m.get("precision") or {}
            limits = m.get("limits") or {}
            amount_p = precision.get("amount")
            price_p = precision.get("price")
            out[sym] = MarketMeta(
                symbol=sym, base=m.get("base", ""), quote=m.get("quote", ""),
                active=bool(m.get("active", True)),
                amount_precision=int(amount_p) if isinstance(amount_p, int) else 8,
                price_precision=int(price_p) if isinstance(price_p, int) else 8,
                min_amount=float((limits.get("amount") or {}).get("min") or 0.0),
                min_cost=float((limits.get("cost") or {}).get("min") or 0.0),
            )
        self._markets = out
        return out

    def fetch_tickers(self) -> dict[str, dict]:
        try:
            return self.client.fetch_tickers()
        except Exception as e:
            raise DataUnavailable(f"fetch_tickers failed: {e}") from e

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> Series:
        try:
            rows = self.client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as e:
            raise DataUnavailable(f"fetch_ohlcv {symbol} {timeframe} failed: {e}") from e
        if not rows:
            raise DataUnavailable(f"empty OHLCV for {symbol} {timeframe}")
        return Series.from_ohlcv(symbol, timeframe, rows)

    def fetch_quote(self, symbol: str) -> Quote | None:
        try:
            t = self.client.fetch_ticker(symbol)
        except Exception as e:
            log_event("data", "WARNING", "fetch_ticker failed",
                      symbol=symbol, error=str(e))
            return None
        bid = float(t.get("bid") or 0.0)
        ask = float(t.get("ask") or 0.0)
        last = float(t.get("last") or t.get("close") or 0.0)
        if last <= 0 and bid > 0 and ask > 0:
            last = (bid + ask) / 2
        if last <= 0:
            return None
        return Quote(symbol, bid, ask, last, int(t.get("timestamp") or now_ms()))

    def server_time_ms(self) -> int | None:
        """Used by the startup self-check to detect a badly wrong local clock."""
        try:
            if self.client.has.get("fetchTime"):
                return int(self.client.fetch_time())
        except Exception as e:
            log_event("data", "WARNING", "fetch_time failed", error=str(e))
        return None
