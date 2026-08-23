"""Derivatives data collection.

Phase One posture: COLLECT, DO NOT TRADE ON IT. Funding, open interest and
liquidations are recorded against every observation so that we can later test
whether they improve expectancy (spec section 42). Nothing in the entry path
consumes these values yet -- deliberately.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..logging_setup import log_event
from ..storage.repo import Repo
from ..timeutils import now_ms


@dataclass
class DerivSnapshot:
    symbol: str
    event_time: int          # when the exchange stamped it
    observed_at: int         # when we received it
    funding_rate: float | None = None
    funding_change: float | None = None
    open_interest: float | None = None
    oi_change_pct: float | None = None
    liq_long: float | None = None
    liq_short: float | None = None
    basis: float | None = None
    source: str = ""


class DerivativesProvider(Protocol):
    name: str
    def fetch(self, symbols: list[str]) -> list[DerivSnapshot]: ...


class DerivativesEngine:
    def __init__(self, repo: Repo, providers: list[DerivativesProvider] | None = None) -> None:
        self.repo = repo
        self.providers = providers or []

    def poll(self, symbols: list[str]) -> int:
        n = 0
        for p in self.providers:
            try:
                for snap in p.fetch(symbols):
                    self.repo.add_derivatives({
                        "symbol": snap.symbol, "event_time": snap.event_time,
                        "observed_at": snap.observed_at,
                        "funding_rate": snap.funding_rate,
                        "funding_change": snap.funding_change,
                        "open_interest": snap.open_interest,
                        "oi_change_pct": snap.oi_change_pct,
                        "liq_long": snap.liq_long, "liq_short": snap.liq_short,
                        "basis": snap.basis, "source": snap.source or p.name})
                    n += 1
            except Exception as e:
                log_event("app", "WARNING", "derivatives provider failed",
                          provider=getattr(p, "name", "?"), error=str(e))
        return n

    def context_for(self, symbols: list[str], as_of_ms: int) -> dict[str, dict]:
        out = {}
        for sym in symbols:
            rec = self.repo.latest_derivatives(sym, as_of_ms)
            if rec:
                out[sym] = rec
        return out


class CCXTDerivativesProvider:
    """Funding + open interest from a CCXT perpetual market.

    STATUS: REQUIRES VERIFICATION ON YOUR MACHINE -- no network in the build
    environment. Disabled by default (`intel.derivatives_enabled = false`).
    """
    name = "ccxt_perp"

    def __init__(self, exchange: str = "binance", quote: str = "USDT") -> None:
        import ccxt
        self.client = getattr(ccxt, exchange)({
            "enableRateLimit": True, "options": {"defaultType": "swap"}})
        self.quote = quote

    def _perp(self, symbol: str) -> str:
        base = symbol.split("/")[0]
        return f"{base}/{self.quote}:{self.quote}"

    def fetch(self, symbols: list[str]) -> list[DerivSnapshot]:
        out: list[DerivSnapshot] = []
        for sym in symbols:
            perp = self._perp(sym)
            try:
                fr = self.client.fetch_funding_rate(perp)
            except Exception:
                continue
            oi_val = None
            try:
                oi = self.client.fetch_open_interest(perp)
                oi_val = oi.get("openInterestValue") or oi.get("openInterestAmount")
            except Exception:
                pass
            out.append(DerivSnapshot(
                symbol=sym,
                event_time=int(fr.get("timestamp") or now_ms()),
                observed_at=now_ms(),
                funding_rate=fr.get("fundingRate"),
                open_interest=float(oi_val) if oi_val else None,
                source=self.name))
        return out
