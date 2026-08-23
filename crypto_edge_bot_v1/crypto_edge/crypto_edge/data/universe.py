"""Dynamic market universe: scan broadly, filter aggressively.

The top-N list is the RESEARCH universe. Only assets that survive every
liquidity/tradability filter enter the strategy pipeline, and every rejection
is recorded with its reason so we can later audit whether the filters help or
merely cost us trades.
"""
from __future__ import annotations

import re

import numpy as np

from ..config import UniverseCfg
from ..logging_setup import log_event
from ..models import MarketMeta, Series
from ..timeutils import now_ms, tf_ms

LEVERAGED_RE = re.compile(r"(\d+[LS])$")


class UniverseBuilder:
    def __init__(self, cfg: UniverseCfg) -> None:
        self.cfg = cfg

    # ------------------------------------------------------- static filters
    def static_reject(self, meta: MarketMeta) -> str:
        """Filters that need no market data -- cheapest, applied first."""
        c = self.cfg
        base = meta.base.upper()
        if not meta.active:
            return "market inactive/suspended"
        if meta.symbol in c.blacklist or base in c.blacklist:
            return "blacklisted"
        if c.exclude_stablecoins and base in {s.upper() for s in c.stablecoin_symbols}:
            return "stablecoin"
        if c.exclude_leveraged:
            if LEVERAGED_RE.search(base):
                return "leveraged token"
            for mark in c.leveraged_markers:
                if base.endswith(mark) and base != mark:
                    return "leveraged token"
        if c.exclude_wrapped and base in {w.upper() for w in c.wrapped_markers}:
            return "wrapped/staked duplicate"
        return ""

    # ------------------------------------------------------ liquidity screen
    def rank_by_volume(self, tickers: dict[str, dict],
                       markets: dict[str, MarketMeta]) -> list[tuple[str, float]]:
        rows = []
        for sym, meta in markets.items():
            t = tickers.get(sym) or {}
            qv = t.get("quoteVolume")
            if qv is None:
                last = t.get("last") or t.get("close")
                bv = t.get("baseVolume")
                qv = (float(last) * float(bv)) if (last and bv) else 0.0
            try:
                qv = float(qv or 0.0)
            except (TypeError, ValueError):
                qv = 0.0
            rows.append((sym, qv))
        rows.sort(key=lambda r: r[1], reverse=True)
        return rows

    @staticmethod
    def spread_bps(ticker: dict) -> float:
        bid = float(ticker.get("bid") or 0.0)
        ask = float(ticker.get("ask") or 0.0)
        if bid > 0 and ask > bid:
            mid = (bid + ask) / 2
            return (ask - bid) / mid * 10_000.0
        return float("nan")

    def build_candidates(self, markets: dict[str, MarketMeta],
                         tickers: dict[str, dict]) -> tuple[list[str], list[dict]]:
        """Stage 1: symbol-level and liquidity filters. Returns
        (candidate symbols, audit rows)."""
        c = self.cfg
        audit: list[dict] = []
        ranked = self.rank_by_volume(tickers, markets)
        keep: list[str] = []

        for rank, (sym, qv) in enumerate(ranked, start=1):
            meta = markets[sym]
            row = {"symbol": sym, "rank": rank, "dollar_volume": qv,
                   "spread_bps": None, "included": False, "reject_reason": ""}
            if rank > c.top_n and sym not in c.always_include:
                row["reject_reason"] = f"outside top {c.top_n} by volume"
                audit.append(row)
                continue
            reason = self.static_reject(meta)
            if reason and sym not in c.always_include:
                row["reject_reason"] = reason
                audit.append(row)
                continue
            if qv < c.min_dollar_volume_24h and sym not in c.always_include:
                row["reject_reason"] = (f"24h volume ${qv:,.0f} < "
                                        f"${c.min_dollar_volume_24h:,.0f}")
                audit.append(row)
                continue
            sp = self.spread_bps(tickers.get(sym) or {})
            row["spread_bps"] = None if np.isnan(sp) else sp
            if np.isfinite(sp) and sp > c.max_spread_bps and sym not in c.always_include:
                row["reject_reason"] = f"spread {sp:.1f}bps > {c.max_spread_bps}bps"
                audit.append(row)
                continue
            row["included"] = True
            keep.append(sym)
            audit.append(row)
        return keep, audit

    def filter_by_history(self, symbol: str, series: Series | None,
                          atr_pct: float | None = None) -> str:
        """Stage 2: needs candles. Applied after OHLCV is fetched."""
        c = self.cfg
        if series is None or len(series) == 0:
            return "no candle data"
        if len(series) < c.min_candles_1h:
            return f"only {len(series)} 1h candles (< {c.min_candles_1h})"
        if not series.is_sane():
            return "candle data failed sanity check"
        age_ms = int(series.open_ms[-1] - series.open_ms[0])
        min_age_ms = c.min_market_age_days * 86_400_000
        if age_ms < min_age_ms:
            return f"market too new ({age_ms / 86_400_000:.0f}d < {c.min_market_age_days}d)"
        if atr_pct is not None and np.isfinite(atr_pct):
            if atr_pct < c.min_atr_pct:
                return f"too quiet (ATR {atr_pct:.2f}% < {c.min_atr_pct}%)"
            if atr_pct > c.max_atr_pct:
                return f"too volatile (ATR {atr_pct:.2f}% > {c.max_atr_pct}%)"
        return ""

    def log_audit(self, audit: list[dict]) -> None:
        included = sum(1 for r in audit if r["included"])
        log_event("data", "INFO", "universe refreshed",
                  scanned=len(audit), included=included,
                  top_rejections=_top_reasons(audit))


def _top_reasons(audit: list[dict], n: int = 6) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in audit:
        if r["included"] or not r["reject_reason"]:
            continue
        key = r["reject_reason"].split("(")[0].split("$")[0].strip()
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:n])
