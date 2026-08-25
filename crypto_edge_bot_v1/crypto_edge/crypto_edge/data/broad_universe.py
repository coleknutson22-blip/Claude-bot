"""The BROAD universe: what counts as a legitimate crypto asset at all.

WHY THIS EXISTS
---------------
The previous definition of "top 200" was "the 200 markets with the highest 24h
volume on our exchange". That is not a universe of assets, it is a snapshot of
where this week's speculation happens to be pointed. It systematically admits
whatever is being churned hardest right now -- a fresh listing being farmed, a
memecoin mid-mania, a token whose volume is mostly wash -- and it silently
changes composition with venue promotions and fee campaigns. Backtests run
against it are not reproducible, because the list is a function of activity
that is itself downstream of the moves we are trying to trade.

The corrected pipeline, in order:

    BROAD TOP ~200 LEGITIMATE ASSETS   (market-cap ranked, venue-independent)
        -> intersect with the selected exchange's tradable markets
        -> drop stablecoins / wrapped / staked / leveraged / blacklisted
        -> liquidity, spread, history, age and volatility filters
        -> strategy ranking

Market capitalisation is not a quality judgement, but it is a defensible,
externally-published, slow-moving definition of "this asset is a real thing
people hold" -- which is exactly the property the venue's volume table lacks.
The intersection direction matters: the broad list decides WHAT is legitimate,
the exchange decides only WHAT WE CAN TRADE of it.

OPERATIONAL RULES
-----------------
  * every fetch is cached with its source, its provider timestamp and a content
    hash, so any past universe decision can be reproduced exactly
  * a provider outage falls back to the last valid cached snapshot, flagged stale
  * with neither a live nor a usable cached universe, NEW ENTRIES FAIL CLOSED
  * none of the above can touch OPEN POSITIONS. Managing risk we already hold
    must never depend on a third-party ranking API being reachable; that is
    enforced in the engine, which always fetches data for open positions
    regardless of what this module returns.

STATUS: `CoinGeckoUniverseProvider` REQUIRES VERIFICATION ON YOUR MACHINE --
the build environment has no outbound network, so it is written and
import-checked but its HTTP call has never been executed. Everything else in
this module (validation, caching, fallback, staleness, fail-closed) is VERIFIED
OFFLINE.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Protocol

from ..logging_setup import log_event
from ..storage.repo import Repo
from ..timeutils import now_ms

# Assets that are legitimately large but are not a directional crypto bet:
# fiat-pegged stablecoins and wrapped/staked receipts for an asset we may
# already hold. They are dropped here as well as in the venue-level filters,
# because they should never have entered the broad list in the first place.
NON_ASSET_BASES = frozenset({
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USDP", "USDD", "PYUSD",
    "EURT", "EURS", "GUSD", "LUSD", "SUSD", "USTC", "FRAX", "USDE", "RLUSD",
    "USDS", "USD1", "BUIDL", "USDY", "USDF", "XAUT", "PAXG",
    "WBTC", "WETH", "WBETH", "STETH", "WSTETH", "CBETH", "RETH", "BETH",
    "SAVAX", "STSOL", "MSOL", "JITOSOL", "WEETH", "EZETH", "RSETH", "LBTC",
    "SOLVBTC", "CBBTC", "WBETH", "SUSDE", "SUSDS", "WBNB", "WHYPE", "WSOL",
})


@dataclass(frozen=True)
class RankedAsset:
    """One entry in the broad ranking.

    `symbol` is a BASE asset ticker (BTC), not a market pair -- the pairing is
    the exchange's business. `asset_id` is the provider's STABLE IDENTIFIER
    ("bitcoin"), and it is the field that actually identifies the asset:
    tickers are not globally unique and are routinely reused by unrelated
    projects, so anything that must be correct keys off `asset_id`.
    """
    rank: int
    symbol: str
    name: str = ""
    market_cap: float = 0.0
    asset_id: str = ""

    def identity(self) -> str:
        return self.asset_id or f"symbol:{self.symbol.upper()}"

    def as_dict(self) -> dict:
        return {"rank": self.rank, "symbol": self.symbol, "name": self.name,
                "market_cap": self.market_cap, "asset_id": self.asset_id}


@dataclass
class Resolution:
    """Outcome of mapping one exchange base symbol to a broad-universe asset."""
    rank: int | None
    asset_id: str = ""
    reason: str = ""          # empty when resolved

    @property
    def ok(self) -> bool:
        return self.rank is not None


@dataclass
class BroadUniverse:
    """A resolved broad universe plus its full provenance.

    `ambiguous` maps a ticker to every distinct asset identity seen claiming it
    across the whole collision scan -- not merely within the top-N. A ticker in
    that map cannot be resolved by symbol alone and is REFUSED rather than
    guessed at, unless an explicit override pins it to one asset id.
    """
    assets: list[RankedAsset]
    source: str
    fetched_ms: int
    as_of_ms: int
    content_hash: str
    from_cache: bool = False
    stale: bool = False
    age_s: float = 0.0
    ambiguous: dict[str, list[str]] = field(default_factory=dict)
    scanned: int = 0
    overrides: dict[str, str] = field(default_factory=dict, repr=False)
    _ranks: dict[str, int] = field(default_factory=dict, repr=False)
    _by_id: dict[str, RankedAsset] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self._ranks:
            self._ranks = {a.symbol.upper(): a.rank for a in self.assets}
        if not self._by_id:
            self._by_id = {a.asset_id: a for a in self.assets if a.asset_id}
        self.ambiguous = {k.upper(): v for k, v in (self.ambiguous or {}).items()}
        self.overrides = {k.upper(): v for k, v in (self.overrides or {}).items()}

    def __len__(self) -> int:
        return len(self.assets)

    @property
    def bases(self) -> set[str]:
        """Unambiguously resolvable base tickers."""
        return {b for b in self._ranks if self.resolve(b).ok}

    def resolve(self, base: str) -> Resolution:
        """Map an exchange base ticker to a ranked asset, or explain why not.

        Order matters: an explicit override is consulted FIRST, because it is
        the only deterministic statement of identity available to us and it
        must be able to rescue a ticker that is otherwise ambiguous.
        """
        b = base.upper()

        pinned = self.overrides.get(b)
        if pinned:
            asset = self._by_id.get(pinned)
            if asset is not None:
                return Resolution(asset.rank, asset.asset_id)
            return Resolution(None, "",
                              f"override pins {b} to asset id '{pinned}', which is "
                              f"not in the broad universe")

        claimants = self.ambiguous.get(b)
        if claimants:
            return Resolution(None, "",
                              f"ticker '{b}' is claimed by {len(claimants)} distinct "
                              f"assets ({', '.join(claimants[:3])}"
                              f"{'...' if len(claimants) > 3 else ''}); "
                              f"set universe.broad_symbol_overrides to disambiguate")

        rank = self._ranks.get(b)
        if rank is None:
            return Resolution(None, "", "not in the broad top-N asset universe")
        asset = next((a for a in self.assets if a.symbol.upper() == b), None)
        return Resolution(rank, asset.asset_id if asset else "")

    def rank_of(self, base: str) -> int | None:
        return self.resolve(base).rank

    def provenance(self) -> dict:
        return {"source": self.source, "fetched_ms": self.fetched_ms,
                "as_of_ms": self.as_of_ms, "content_hash": self.content_hash,
                "n_assets": len(self.assets), "from_cache": self.from_cache,
                "stale": self.stale, "age_s": round(self.age_s, 1),
                "scanned": self.scanned, "ambiguous_tickers": len(self.ambiguous)}


class BroadUniverseProvider(Protocol):
    name: str

    def fetch(self, limit: int) -> list[RankedAsset]: ...


# --------------------------------------------------------------- providers
class CoinGeckoUniverseProvider:
    """Top-N assets by market capitalisation from CoinGecko's public API.

    Chosen because it is publicly documented, needs no key for this endpoint,
    covers assets rather than venue pairs, and publishes a ranking that changes
    on the scale of weeks rather than hours.

    STATUS: REQUIRES VERIFICATION ON YOUR MACHINE. No network in the build
    environment, so this method has never actually run.
    """
    name = "coingecko_market_cap"
    URL = "https://api.coingecko.com/api/v3/coins/markets"

    def __init__(self, timeout_s: float = 20.0, vs_currency: str = "usd") -> None:
        self.timeout_s = timeout_s
        self.vs_currency = vs_currency

    def fetch(self, limit: int = 250) -> list[RankedAsset]:
        import requests
        out: list[RankedAsset] = []
        per_page = min(250, max(1, limit))
        page = 1
        while len(out) < limit:
            r = requests.get(
                self.URL,
                params={"vs_currency": self.vs_currency,
                        "order": "market_cap_desc", "per_page": per_page,
                        "page": page, "sparkline": "false"},
                timeout=self.timeout_s,
                headers={"Accept": "application/json"})
            if r.status_code != 200:
                raise BroadUniverseUnavailable(
                    f"{self.name}: HTTP {r.status_code}: {r.text[:200]}")
            rows = r.json()
            if not isinstance(rows, list) or not rows:
                break
            for row in rows:
                sym = str(row.get("symbol") or "").upper()
                if not sym:
                    continue
                out.append(RankedAsset(
                    rank=int(row.get("market_cap_rank") or (len(out) + 1)),
                    symbol=sym, name=str(row.get("name") or ""),
                    market_cap=float(row.get("market_cap") or 0.0),
                    # CoinGecko's "id" is the stable, globally unique handle
                    # ("bitcoin"). It is what we key identity off; the ticker is
                    # only a convenience for matching exchange bases.
                    asset_id=str(row.get("id") or "")))
            if len(rows) < per_page:
                break
            page += 1
        return out[:limit]


class StaticBroadUniverseProvider:
    """A fixed list of base assets. Used by the offline tests and the smoke
    test, and available as a deliberate operator override.

    It is NOT a substitute for a live ranking: a hand-maintained list goes stale
    silently, which is the failure mode this whole module exists to avoid.
    """

    def __init__(self, symbols: list[str], name: str = "static") -> None:
        self.name = name
        self._symbols = [s.upper() for s in symbols]

    def fetch(self, limit: int = 250) -> list[RankedAsset]:
        return [RankedAsset(rank=i, symbol=s, name=s, market_cap=0.0,
                            asset_id=f"static:{s.lower()}")
                for i, s in enumerate(self._symbols[:limit], start=1)]


class BroadUniverseUnavailable(Exception):
    """The provider could not supply a usable ranking."""


# ----------------------------------------------------------------- service
def content_hash(assets: list[RankedAsset]) -> str:
    """Stable hash of the ranking's content, for reproducibility checks.

    Includes the asset identity, not just the ticker: two different assets that
    happen to share a ticker must not hash to the same universe.
    """
    blob = json.dumps([[a.rank, a.symbol, a.identity()] for a in assets],
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


class BroadUniverseService:
    """Fetch / validate / cache / fall back. The engine talks only to this."""

    def __init__(self, repo: Repo, provider: BroadUniverseProvider | None = None,
                 *, limit: int = 200, min_assets: int = 50,
                 refresh_hours: int = 12, max_cache_age_hours: int = 168,
                 drop_non_assets: bool = True, collision_scan_limit: int = 1000,
                 symbol_overrides: dict[str, str] | None = None) -> None:
        self.repo = repo
        self.provider = provider
        self.limit = limit
        self.min_assets = min_assets
        self.refresh_ms = max(0, refresh_hours) * 3_600_000
        self.max_cache_age_ms = max(0, max_cache_age_hours) * 3_600_000
        self.drop_non_assets = drop_non_assets
        # We deliberately scan further down the ranking than we trade. A ticker
        # in our top-200 may also be used by an asset at rank 640 that this
        # exchange happens to list -- and matching on the ticker alone would
        # then buy the wrong thing. Seeing the collision requires looking past
        # the trading cut-off.
        self.collision_scan_limit = max(limit, collision_scan_limit)
        self.symbol_overrides = {k.upper(): v for k, v in
                                 (symbol_overrides or {}).items()}
        self.last_error: str = ""

    # ------------------------------------------------------------ validation
    def _clean(self, assets: list[RankedAsset]) -> list[RankedAsset]:
        """Normalise, drop non-assets, sort by rank, de-duplicate by IDENTITY.

        Note what is NOT done here: duplicate TICKERS are not collapsed. Two
        distinct assets sharing a ticker are both kept, so `_collisions()` can
        see them. Collapsing them -- which is what the previous implementation
        did -- silently picked whichever ranked higher and threw away the
        evidence that the ticker was ambiguous at all.
        """
        seen_ids: set[str] = set()
        out: list[RankedAsset] = []
        for a in sorted(assets, key=lambda x: x.rank):
            sym = a.symbol.upper().strip()
            if not sym:
                continue
            if self.drop_non_assets and sym in NON_ASSET_BASES:
                continue
            ident = (a.asset_id or f"symbol:{sym}").strip()
            if ident in seen_ids:
                continue           # the same asset listed twice: genuinely a dup
            seen_ids.add(ident)
            out.append(RankedAsset(a.rank, sym, a.name, a.market_cap, a.asset_id))
        return out

    def _collisions(self, scanned: list[RankedAsset]) -> dict[str, list[str]]:
        """Tickers claimed by more than one distinct asset in the whole scan."""
        by_ticker: dict[str, list[str]] = {}
        for a in scanned:
            by_ticker.setdefault(a.symbol.upper(), [])
            ident = a.identity()
            if ident not in by_ticker[a.symbol.upper()]:
                by_ticker[a.symbol.upper()].append(ident)
        return {t: ids for t, ids in by_ticker.items() if len(ids) > 1}

    def _usable(self, assets: list[RankedAsset]) -> bool:
        return len(assets) >= self.min_assets

    # --------------------------------------------------------------- reading
    def _from_cache(self, now: int) -> BroadUniverse | None:
        row = self.repo.latest_broad_universe(min_assets=self.min_assets)
        if row is None:
            return None
        age = now - int(row["fetched_ms"])
        if self.max_cache_age_ms and age > self.max_cache_age_ms:
            log_event("data", "ERROR",
                      "cached broad universe is too old to trust",
                      source=row["source"], age_h=round(age / 3_600_000, 1),
                      limit_h=round(self.max_cache_age_ms / 3_600_000, 1))
            return None
        assets = [RankedAsset(int(a["rank"]), str(a["symbol"]),
                              str(a.get("name") or ""),
                              float(a.get("market_cap") or 0.0),
                              str(a.get("asset_id") or ""))
                  for a in row["assets"]]
        return BroadUniverse(
            assets=assets, source=row["source"], fetched_ms=int(row["fetched_ms"]),
            as_of_ms=int(row["as_of_ms"]), content_hash=str(row["content_hash"]),
            from_cache=True, stale=age > self.refresh_ms if self.refresh_ms else False,
            age_s=age / 1000.0,
            # The ambiguity map is cached alongside the ranking: a fallback
            # universe must be exactly as careful about identity as a live one.
            ambiguous=row.get("ambiguous") or {},
            scanned=int(row.get("scanned") or len(assets)),
            overrides=self.symbol_overrides)

    # ----------------------------------------------------------------- get
    def get(self, now: int | None = None, force: bool = False) -> BroadUniverse | None:
        """Resolve the broad universe.

        Returns the freshest usable snapshot, or the last valid cached one if
        the provider is unreachable, or None if neither exists -- in which case
        the caller MUST fail closed for new entries.
        """
        now = now_ms() if now is None else now
        cached = self._from_cache(now)

        if cached is not None and not force and not cached.stale:
            return cached

        if self.provider is None:
            if cached is not None:
                return cached
            self.last_error = "no broad-universe provider configured"
            log_event("data", "ERROR", "broad universe unavailable",
                      reason=self.last_error)
            return None

        try:
            scanned = self._clean(self.provider.fetch(self.collision_scan_limit))
            ambiguous = self._collisions(scanned)
            fetched = scanned[:self.limit]
            if not self._usable(fetched):
                raise BroadUniverseUnavailable(
                    f"{self.provider.name} returned only {len(fetched)} usable "
                    f"assets (need >= {self.min_assets})")
        except Exception as e:
            self.last_error = str(e)
            log_event("data", "ERROR", "broad universe fetch failed",
                      provider=getattr(self.provider, "name", "?"), error=str(e),
                      falling_back_to_cache=cached is not None)
            if cached is not None:
                cached.stale = True
                return cached
            return None

        h = content_hash(fetched)
        # Only collisions that actually touch a tradable ticker are worth
        # carrying: a clash between two assets neither of which is in our
        # top-N cannot mislead us.
        top_tickers = {a.symbol.upper() for a in fetched}
        relevant = {t: ids for t, ids in ambiguous.items() if t in top_tickers}
        self.repo.save_broad_universe(
            source=self.provider.name, fetched_ms=now, as_of_ms=now,
            assets=[a.as_dict() for a in fetched], content_hash=h,
            ambiguous=relevant, scanned=len(scanned))
        self.repo.prune_broad_universe()
        self.last_error = ""
        if relevant:
            log_event("data", "WARNING",
                      "ambiguous tickers in the broad universe; they will be "
                      "refused unless an override pins them",
                      count=len(relevant), tickers=sorted(relevant)[:10])
        log_event("data", "INFO", "broad universe refreshed",
                  source=self.provider.name, n_assets=len(fetched),
                  scanned=len(scanned), ambiguous=len(relevant), content_hash=h)
        return BroadUniverse(assets=fetched, source=self.provider.name,
                             fetched_ms=now, as_of_ms=now, content_hash=h,
                             from_cache=False, stale=False, age_s=0.0,
                             ambiguous=relevant, scanned=len(scanned),
                             overrides=self.symbol_overrides)
