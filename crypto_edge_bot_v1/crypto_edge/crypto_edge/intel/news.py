"""News and event intelligence.

Purpose in Phase One is RISK MANAGEMENT AND DATA COLLECTION, not prediction.
Nothing here tries to guess direction from a headline.

Two invariants:
  1. Every event stores `event_time` (when it happened) and `observed_at`
     (when this system could first have known). All reads go through
     `Repo.news_visible_at()`, so a backtest cannot use tomorrow's news.
  2. Classification is DETERMINISTIC -- keyword and source-tier rules, no
     model in the loop. The same headline always produces the same severity.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from ..logging_setup import log_event
from ..storage.repo import Repo
from ..timeutils import now_ms

# ---------------------------------------------------------------- tiers
TIER_OFFICIAL = 1      # project/exchange official, regulators
TIER_MAJOR_PRESS = 2   # reputable financial/crypto publications
TIER_AGGREGATOR = 3    # aggregators, secondary reporting
TIER_SOCIAL = 4        # social media, anonymous accounts, rumour

TIER_CONFIDENCE = {TIER_OFFICIAL: 0.95, TIER_MAJOR_PRESS: 0.75,
                   TIER_AGGREGATOR: 0.5, TIER_SOCIAL: 0.25}

# severity 5 = confirmed catastrophic, 0 = no risk signal
CATEGORY_RULES: list[tuple[str, int, str, tuple[str, ...]]] = [
    ("exploit",      5, "severe_negative", ("exploit", "hacked", "hack", "drained",
                                            "stolen funds", "bridge attack",
                                            "reentrancy", "private key compromise")),
    ("insolvency",   5, "severe_negative", ("insolvency", "bankruptcy", "chapter 11",
                                            "halts withdrawals", "suspends withdrawals",
                                            "proof of reserves failure")),
    ("depeg",        5, "severe_negative", ("depeg", "de-peg", "lost peg",
                                            "breaks peg")),
    ("outage",       4, "negative",        ("outage", "chain halted", "network halt",
                                            "block production stopped")),
    ("delisting",    4, "negative",        ("delist", "delisting", "trading suspended")),
    ("regulatory",   3, "negative",        ("lawsuit", "sec charges", "indicted",
                                            "enforcement action", "banned",
                                            "cease and desist", "subpoena")),
    ("unlock",       2, "negative",        ("token unlock", "vesting", "cliff unlock")),
    ("upgrade",      1, "positive",        ("mainnet", "upgrade", "hard fork",
                                            "hardfork")),
    ("listing",      1, "positive",        ("listing", "will list", "lists")),
    ("partnership",  1, "positive",        ("partnership", "integration",
                                            "collaboration")),
    ("etf",          1, "strong_positive", ("etf approved", "etf approval",
                                            "spot etf")),
]


@dataclass
class NewsItem:
    """What a provider must return. `observed_at` defaults to fetch time."""
    headline: str
    source: str
    source_tier: int
    event_time: int
    observed_at: int
    url: str = ""
    assets: tuple[str, ...] = ()


class NewsProvider(Protocol):
    name: str
    def fetch(self, since_ms: int) -> list[NewsItem]: ...


def classify(headline: str) -> tuple[str, int, str]:
    """Deterministic keyword classification -> (category, severity, sentiment)."""
    h = headline.lower()
    best = ("other", 0, "neutral")
    for category, severity, sentiment, keywords in CATEGORY_RULES:
        if any(k in h for k in keywords):
            if severity > best[1]:
                best = (category, severity, sentiment)
    return best


def event_id(item: NewsItem) -> str:
    """Stable hash so re-fetching the same headline never duplicates a row."""
    raw = f"{item.source}|{item.headline}|{item.event_time}"
    return "news-" + hashlib.sha256(raw.encode()).hexdigest()[:20]


def confidence_for(tier: int, corroborating_sources: int = 1) -> float:
    """Multiple independent sources raise confidence; a lone social post stays low."""
    base = TIER_CONFIDENCE.get(tier, 0.25)
    bonus = min(0.2, 0.1 * max(0, corroborating_sources - 1))
    return min(0.99, base + bonus)


class NewsEngine:
    def __init__(self, repo: Repo, providers: list[NewsProvider] | None = None,
                 block_severity: int = 4, block_hours: int = 24) -> None:
        self.repo = repo
        self.providers = providers or []
        self.block_severity = block_severity
        self.block_hours = block_hours

    def ingest(self, items: list[NewsItem]) -> int:
        """Store and, where warranted, restrict trading. Returns count stored."""
        stored = 0
        seen_headlines: dict[str, int] = {}
        for it in items:
            key = it.headline.lower()[:80]
            seen_headlines[key] = seen_headlines.get(key, 0) + 1

        for it in items:
            category, severity, sentiment = classify(it.headline)
            conf = confidence_for(it.source_tier,
                                  seen_headlines.get(it.headline.lower()[:80], 1))
            restricted = 0
            # Only a high-severity event from a source we actually trust may
            # block trading. A rumour on social media is logged, never acted on.
            if severity >= self.block_severity and it.source_tier <= TIER_MAJOR_PRESS:
                for asset in it.assets:
                    self.repo.block_symbol(
                        asset, f"{category}: {it.headline[:80]}",
                        it.observed_at + self.block_hours * 3_600_000)
                    restricted = 1
                    log_event("strategy", "WARNING", "asset blocked by news event",
                              symbol=asset, category=category, severity=severity,
                              source=it.source)
            self.repo.add_news_event({
                "id": event_id(it), "event_time": it.event_time,
                "observed_at": it.observed_at, "source": it.source,
                "source_tier": it.source_tier, "headline": it.headline[:500],
                "url": it.url, "assets": list(it.assets), "category": category,
                "sentiment": sentiment, "severity": severity, "confidence": conf,
                "caused_restriction": restricted})
            stored += 1
        return stored

    def poll(self, since_ms: int) -> int:
        total = 0
        for p in self.providers:
            try:
                total += self.ingest(p.fetch(since_ms))
            except Exception as e:
                log_event("app", "WARNING", "news provider failed",
                          provider=getattr(p, "name", "?"), error=str(e))
        return total

    def context_for(self, symbols: list[str], as_of_ms: int,
                    lookback_h: int = 48) -> dict[str, dict]:
        """Worst visible event per symbol. Uses observed_at, never event_time."""
        since = as_of_ms - lookback_h * 3_600_000
        out: dict[str, dict] = {}
        bases = {s.split("/")[0].upper(): s for s in symbols}
        for ev in self.repo.news_visible_at(as_of_ms, since):
            if ev["severity"] <= 0:
                continue
            for asset in ev["assets"]:
                sym = bases.get(asset.split("/")[0].upper())
                if not sym:
                    continue
                cur = out.get(sym)
                if cur is None or ev["severity"] > cur["severity"]:
                    out[sym] = ev
        return out


class StubNewsProvider:
    """No-op provider so the engine runs with news enabled but nothing wired.

    To add a real feed, implement `fetch(since_ms) -> list[NewsItem]` and set
    `observed_at` to the moment YOUR system received it -- never the article's
    publication time, which may precede your access to it.
    """
    name = "stub"

    def fetch(self, since_ms: int) -> list[NewsItem]:
        return []
