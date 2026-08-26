"""Venue-level universe filtering: intersect, then filter aggressively.

This is stage 1 of the pipeline. What counts as a legitimate asset is decided
upstream, by `broad_universe.BroadUniverseService` (a market-cap ranking that
does not depend on our exchange); this module decides which of those assets we
can actually trade here, and how well.

    BROAD TOP ~200 (upstream)
        -> intersect with this venue's markets      <-- this module
        -> static filters: stablecoin/wrapped/staked/leveraged/blacklist
        -> liquidity + spread filters
        -> [after OHLCV] history, age and volatility filters
        -> strategy ranking

Ordering into the strategy pipeline follows the BROAD rank (market cap) when a
broad universe is supplied, not the venue's volume table. Exchange volume is
still used as a liquidity FILTER -- it just no longer defines membership,
because "what is being churned hardest today" is not a definition of an asset.

Passing `broad=None` falls back to pure volume ranking. That path exists for
unit-testing the filters in isolation; the engine never uses it, because
without a broad universe it fails closed instead (see `TradingEngine`).

Every rejection is recorded with its reason so we can later audit whether the
filters help or merely cost us trades.
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

    def rank_by_broad_universe(self, broad, tickers: dict[str, dict],
                               markets: dict[str, MarketMeta]
                               ) -> list[tuple[str, float, int | None]]:
        """Order the venue's markets by their BROAD (market-cap) rank.

        Markets whose base asset is absent from the broad universe get rank
        None and are rejected downstream -- that is the intersection. Volume is
        carried along for the liquidity filter and as a tiebreak among assets
        the broad list does not separate.
        """
        rows: list[tuple[str, float, int | None, str]] = []
        for sym, qv in self.rank_by_volume(tickers, markets):
            res = broad.resolve(markets[sym].base)
            rows.append((sym, qv, res.rank, res.reason))
        # unranked assets sort last, then by descending volume within a rank
        rows.sort(key=lambda r: (r[2] is None, r[2] if r[2] is not None else 0, -r[1]))
        return rows

    def build_candidates(self, markets: dict[str, MarketMeta],
                         tickers: dict[str, dict],
                         broad=None) -> tuple[list[str], list[dict]]:
        """Stage 1: intersection, symbol-level and liquidity filters.

        Returns (candidate symbols in pipeline order, audit rows). With `broad`
        supplied, membership and ordering come from the broad universe; without
        it, the legacy volume ranking is used (unit tests only -- see module
        docstring).
        """
        c = self.cfg
        audit: list[dict] = []
        keep: list[str] = []

        if broad is not None:
            ordered = self.rank_by_broad_universe(broad, tickers, markets)
        else:
            ordered = [(sym, qv, None, "") for sym, qv in
                       self.rank_by_volume(tickers, markets)]

        for position, (sym, qv, broad_rank, why) in enumerate(ordered, start=1):
            meta = markets[sym]
            rank = broad_rank if broad_rank is not None else position
            row = {"symbol": sym, "rank": rank, "dollar_volume": qv,
                   "spread_bps": None, "included": False, "reject_reason": ""}

            if broad is not None and broad_rank is None:
                if sym not in c.always_include:
                    # `why` distinguishes "this asset is not in the top-N" from
                    # "this TICKER is claimed by several assets and we refuse to
                    # guess which one this venue means" -- very different facts.
                    row["reject_reason"] = why or "not in the broad top-N asset universe"
                    audit.append(row)
                    continue
                row["rank"] = rank = position
            if rank > c.top_n and sym not in c.always_include:
                row["reject_reason"] = (
                    f"outside top {c.top_n} by market cap" if broad is not None
                    else f"outside top {c.top_n} by volume")
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
                          atr_pct: float | None = None, *,
                          requested_bars: int | None = None) -> str:
        """Stage 2: needs candles. Applied after OHLCV is fetched.

        `requested_bars` is how many bars were asked of the venue. It is what
        separates "this market is genuinely young" from "this venue would not
        give us more history", which look identical in the data and are very
        different facts. Both still block an entry -- we fail closed either way
        -- but they must not be reported as the same thing, because one is a
        property of the asset and the other is a property of our data feed.
        """
        c = self.cfg
        if series is None or len(series) == 0:
            return "no candle data"
        truncated = (requested_bars is not None and len(series) < requested_bars)
        if len(series) < c.min_candles_1h:
            if truncated:
                return (f"venue supplied only {len(series)} of {requested_bars} "
                        f"requested bars (need {c.min_candles_1h})")
            return f"only {len(series)} 1h candles (< {c.min_candles_1h})"
        if not series.is_sane():
            return "candle data failed sanity check"
        age_ms = int(series.open_ms[-1] - series.open_ms[0])
        min_age_ms = c.min_market_age_days * 86_400_000
        if age_ms < min_age_ms:
            if truncated:
                # We did not receive enough history to demonstrate the market's
                # age. Saying "too new" here would blame the asset for a limit
                # of our own fetch -- the mistake that made every Kraken market
                # look days old regardless of how long it had existed.
                return (f"history truncated by venue: {len(series)} of "
                        f"{requested_bars} bars spans {age_ms / 86_400_000:.0f}d, "
                        f"cannot verify {c.min_market_age_days}d age")
            return f"market too new ({age_ms / 86_400_000:.0f}d < {c.min_market_age_days}d)"
        if atr_pct is not None and np.isfinite(atr_pct):
            if atr_pct < c.min_atr_pct:
                return f"too quiet (ATR {atr_pct:.2f}% < {c.min_atr_pct}%)"
            if atr_pct > c.max_atr_pct:
                return f"too volatile (ATR {atr_pct:.2f}% > {c.max_atr_pct}%)"
        return ""

    def log_audit(self, audit: list[dict], broad=None) -> None:
        included = sum(1 for r in audit if r["included"])
        extra = {"broad_source": broad.source, "broad_assets": len(broad),
                 "broad_from_cache": broad.from_cache, "broad_stale": broad.stale}\
            if broad is not None else {}
        log_event("data", "INFO", "universe refreshed",
                  scanned=len(audit), included=included,
                  top_rejections=_top_reasons(audit), **extra)


def _top_reasons(audit: list[dict], n: int = 6) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in audit:
        if r["included"] or not r["reject_reason"]:
            continue
        key = r["reject_reason"].split("(")[0].split("$")[0].strip()
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:n])
