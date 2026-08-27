"""Configuration: TOML for parameters, .env for secrets. Nothing sensitive is
ever read from the TOML file, and no credential is ever hard-coded."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .timeutils import tf_ms


class ConfigError(Exception):
    pass


# utf-8-sig, not utf-8: Windows Notepad writes a UTF-8 BYTE ORDER MARK at the
# start of the file. Read as plain utf-8 that BOM becomes a \ufeff character
# glued to the first key, so TELEGRAM_BOT_TOKEN silently becomes
# "\ufeffTELEGRAM_BOT_TOKEN", is never found, and Telegram quietly disables
# itself with no error anywhere. utf-8-sig strips it and is a no-op otherwise.
TEXT_ENCODING = "utf-8-sig"


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader so we take no third-party dependency for secrets.
    Existing environment variables always win."""
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding=TEXT_ENCODING).splitlines():
        line = raw.strip().lstrip("\ufeff")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if not key:
            continue
        os.environ.setdefault(key, val)


@dataclass
class ExchangeCfg:
    name: str = "binance"
    quote: str = "USDT"
    # Public market-data endpoints only. No API key is required for paper mode
    # and none is read here -- see safety.live_trading_enabled.
    rate_limit_ms: int = 250
    # Bars per REQUEST -- a paging hint, not the total history we want. Venues
    # cap responses (Kraken at 720, and it ignores this value entirely), so the
    # feed pages until it has the depth the universe filters require. See
    # Config.required_history_bars().
    ohlcv_limit: int = 300
    # Closed candles kept in memory per (symbol, timeframe). A closed candle is
    # immutable, so while no newer one has closed the cache IS the current state
    # of the market and no request is made at all. 0 disables the cache.
    ohlcv_cache_bars: int = 2000
    max_history_bars: int = 2000        # hard ceiling on paging per symbol


@dataclass
class UniverseCfg:
    top_n: int = 200                    # research/scan universe size
    max_tradable: int = 40              # how many survive filtering into the pipeline
    refresh_minutes: int = 360
    min_dollar_volume_24h: float = 5_000_000.0
    max_spread_bps: float = 15.0
    min_candles_1h: int = 400           # history requirement
    min_market_age_days: int = 45
    min_atr_pct: float = 0.8            # too quiet to pay costs
    max_atr_pct: float = 25.0           # untradeable chaos
    exclude_stablecoins: bool = True
    exclude_leveraged: bool = True
    exclude_wrapped: bool = True
    stablecoin_symbols: list[str] = field(default_factory=lambda: [
        "USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD", "USDP", "USDD", "PYUSD",
        "EURT", "EURS", "GUSD", "LUSD", "SUSD", "USTC", "FRAX", "USDE", "RLUSD"])
    leveraged_markers: list[str] = field(default_factory=lambda: [
        "UP", "DOWN", "BULL", "BEAR", "3L", "3S", "5L", "5S", "2L", "2S"])
    wrapped_markers: list[str] = field(default_factory=lambda: [
        "WBTC", "WETH", "WBETH", "STETH", "WSTETH", "CBETH", "RETH", "BETH",
        "SAVAX", "STSOL", "MSOL", "JITOSOL"])
    # Configured as BASE assets, not pairs: the pair is base + whatever quote
    # currency this venue is being operated in. Hardcoding "BTC/USDT" here made
    # the whole always-include list silently wrong on a USD venue.
    always_include_bases: list[str] = field(default_factory=lambda: ["BTC", "ETH"])
    # Derived from always_include_bases + the effective quote at load time.
    # Setting it explicitly in the TOML overrides that, and is validated.
    always_include: list[str] = field(default_factory=list)
    blacklist: list[str] = field(default_factory=list)
    # --- broad (venue-independent) asset universe ------------------------
    # "Top 200" means the top ~200 legitimate crypto ASSETS by market cap,
    # intersected with what this exchange lists -- NOT the exchange's own
    # highest-volume markets. See data/broad_universe.py.
    broad_source: str = "coingecko"     # "coingecko" | "static" | "none"
    broad_limit: int = 200              # how many assets the broad list holds
    broad_min_assets: int = 50          # fewer than this = provider is broken
    broad_refresh_hours: int = 12       # market caps move slowly; so do we
    broad_max_cache_age_hours: int = 168  # a week-old cache stops being usable
    broad_static_assets: list[str] = field(default_factory=list)
    # Ticker symbols are NOT globally unique. We scan further down the ranking
    # than we trade so that a ticker shared with a lower-ranked asset is
    # visible as a collision; a colliding ticker is REFUSED rather than guessed.
    broad_collision_scan_limit: int = 1000
    # Deterministic disambiguation: {exchange base symbol: provider asset id},
    # e.g. {"GRT" = "the-graph"}. Consulted before anything else, so it also
    # rescues a ticker that would otherwise be refused as ambiguous.
    broad_symbol_overrides: dict = field(default_factory=dict)
    # --- market age, established independently of indicator history -------
    # A venue's OHLCV cap is PER TIMEFRAME: Kraken's 720-bar limit is 30 days at
    # 1h but 720 days at 1d. Age therefore asks at a coarse resolution, where
    # reach matters, while the indicators keep asking at 1h, where density does.
    age_probe_timeframe: str = "1d"
    age_probe_bars: int = 400          # 400 daily bars ~= 400 days of reach
    age_cache_hours: int = 24          # age changes slowly; re-probe rarely
    # With no live and no usable cached broad universe, NEW ENTRIES stop.
    # Open positions are managed regardless -- see TradingEngine.fetch_data.
    require_broad_universe: bool = True


@dataclass
class StrategyCfg:
    name: str = "trend_breakout"
    version: str = "1.0.0"
    entry_timeframe: str = "1h"
    regime_timeframe: str = "4h"
    donchian_lookback: int = 48
    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200
    atr_period: int = 14
    adx_period: int = 14
    rsi_period: int = 14
    min_adx: float = 20.0
    max_rsi: float = 78.0               # exhaustion filter
    min_rel_volume: float = 1.2
    momentum_periods: list[int] = field(default_factory=lambda: [24, 72, 168])
    max_extension_atr: float = 3.5      # refuse entries far above the breakout level
    min_score: float = 55.0             # 0-100 opportunity score threshold
    stop_atr_mult: float = 2.2
    trail_atr_mult: float = 3.0         # chandelier
    breakeven_at_r: float = 1.0
    breakeven_offset_r: float = 0.1
    time_stop_bars: int = 96
    time_stop_min_r: float = 0.5        # exit if we haven't made this much by then
    regime_ema: int = 50
    # The reference asset for the market-regime read, as a BASE. The pair is
    # completed with the effective quote currency at load time, so operating
    # kraken/USD uses BTC/USD rather than a USDT pair the venue may not even
    # list -- which would silently disable the regime filter.
    btc_base: str = "BTC"
    btc_symbol: str = ""        # derived; see Config.resolve_symbols()
    warmup_bars: int = 250


@dataclass
class RiskCfg:
    risk_per_trade_pct: float = 0.5
    max_position_pct: float = 15.0      # max allocation to one position
    max_portfolio_exposure_pct: float = 60.0
    max_open_positions: int = 6
    max_new_entries_per_cycle: int = 2
    daily_loss_limit_pct: float = 3.0
    max_drawdown_pct: float = 20.0      # kill switch
    max_correlation: float = 0.85
    correlation_lookback: int = 168     # hours
    min_stop_distance_pct: float = 0.3  # reject absurdly tight stops


@dataclass
class ExecutionCfg:
    starting_equity: float = 10_000.0
    taker_fee_bps: float = 7.5          # 0.075%
    maker_fee_bps: float = 7.5
    slippage_bps: float = 6.0           # market-order slippage assumption
    stop_slippage_bps: float = 15.0     # stops fill worse than limit orders
    use_book_spread: bool = True        # cross the spread when a quote exists
    max_spread_bps_entry: float = 25.0  # refuse to trade a blown-out book
    # --- entry quote validation: NEW ENTRIES FAIL CLOSED ------------------
    # A new position may only be opened against a live quote we can trust.
    # Missing, invalid, stale, malformed or unavailable -> no entry. These do
    # NOT gate exits: an open position must always be manageable, quote or not.
    max_quote_age_s: int = 90           # older than this is stale -> no entry
    max_quote_future_skew_s: int = 30   # a quote stamped ahead of us is broken
    # Timestamp policy. Some venues do not stamp their ticker at all; CCXT
    # reports that as None and the adapter substitutes local receive time,
    # flagging it. An age check against a local stamp is vacuous, so it is
    # SKIPPED rather than allowed to return a meaningless "fresh" -- see
    # PaperBroker.validate_entry_quote. Set this true on a venue whose stamps
    # you have verified, to refuse quotes that lack one.
    require_venue_quote_timestamp: bool = False
    # What the CCXT adapter does when a ticker has no venue timestamp:
    # "local" (stamp on receipt, flag it) or "reject" (discard the quote).
    quote_ts_fallback: str = "local"
    max_quote_deviation_pct: float = 10.0   # quote vs signal reference sanity
    # Final sizing is done against the SIMULATED FILL, not the signal candle's
    # close. This is the tolerance on the post-sizing risk revalidation, and
    # exists only to absorb float/rounding noise -- not real risk drift.
    risk_overshoot_tolerance_pct: float = 1.0


@dataclass
class SafetyCfg:
    mode: str = "PAPER"                 # PAPER only; see validate()
    live_trading_enabled: bool = False  # hard-wired off, see validate()
    max_data_staleness_s: int = 900
    candle_close_buffer_s: int = 20     # clock-skew guard before trusting a close
    max_clock_skew_s: int = 60
    require_btc_data: bool = True


@dataclass
class TelegramCfg:
    enabled: bool = True
    heartbeat_minutes: int = 60
    daily_report_hour_utc: int = 0
    stop_update_min_pct: float = 0.75   # only notify meaningful stop moves
    error_cooldown_s: int = 900         # anti-spam
    timeout_s: int = 10
    max_retries: int = 3                # transport retries within one attempt
    # --- durable outbox (see notify/telegram.py) ------------------------
    outbox_lease_s: int = 120           # how long one in-flight send holds a key
    outbox_max_attempts: int = 12       # cycles of retry before parking as FAILED
    outbox_flush_limit: int = 20        # pending messages replayed per cycle
    outbox_retention_days: int = 30     # how long DELIVERED rows are kept


@dataclass
class EngineCfg:
    poll_seconds: int = 30
    # A FLOOR on the gap between cycles, so a cycle that overruns `poll_seconds`
    # cannot degrade into a back-to-back loop against the venue's rate limiter.
    # The realised cadence is therefore max(poll_seconds, cycle + min_pause_seconds).
    min_pause_seconds: float = 5.0
    db_path: str = "data/crypto_edge.db"
    log_dir: str = "logs"
    equity_snapshot_minutes: int = 15
    counterfactual_horizons_h: list[int] = field(default_factory=lambda: [1, 4, 12, 24, 48, 168])


@dataclass
class IntelCfg:
    """News / token-event / derivatives collection. Providers are pluggable and
    default to disabled so Phase One never blocks on a third-party feed."""
    news_enabled: bool = False
    news_poll_minutes: int = 15
    news_block_severity: int = 4        # >= this severity blocks new entries
    news_block_hours: int = 24
    derivatives_enabled: bool = False
    derivatives_poll_minutes: int = 30
    token_event_block_hours: int = 12   # avoid new entries near an unlock


@dataclass
class Config:
    exchange: ExchangeCfg = field(default_factory=ExchangeCfg)
    universe: UniverseCfg = field(default_factory=UniverseCfg)
    strategy: StrategyCfg = field(default_factory=StrategyCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)
    execution: ExecutionCfg = field(default_factory=ExecutionCfg)
    safety: SafetyCfg = field(default_factory=SafetyCfg)
    telegram: TelegramCfg = field(default_factory=TelegramCfg)
    engine: EngineCfg = field(default_factory=EngineCfg)
    intel: IntelCfg = field(default_factory=IntelCfg)

    def __post_init__(self) -> None:
        # A Config built directly (tests, the smoke test) must be as coherent as
        # one loaded from disk, or the two disagree about which market is BTC.
        self.resolve_symbols()

    # --- secrets, populated from the environment only -------------------
    telegram_token: str = ""
    telegram_chat_id: str = ""

    # Where the effective exchange came from. Recorded so every surface can say
    # not just WHICH venue is in use but WHY -- the failure this guards against
    # is running verification against one venue and the trader against another,
    # which is silent and easy when the venue is a per-invocation flag.
    exchange_source: str = "config file"

    def validate(self) -> list[str]:
        errs: list[str] = []
        if self.safety.mode.upper() != "PAPER":
            errs.append("safety.mode must be PAPER -- live trading is not implemented")
        if self.safety.live_trading_enabled:
            errs.append("safety.live_trading_enabled must be false")
        if self.execution.starting_equity <= 0:
            errs.append("execution.starting_equity must be > 0")
        if not (0 < self.risk.risk_per_trade_pct <= 5):
            errs.append("risk.risk_per_trade_pct must be in (0, 5]")
        if self.risk.max_position_pct > self.risk.max_portfolio_exposure_pct:
            errs.append("max_position_pct cannot exceed max_portfolio_exposure_pct")
        if self.risk.max_open_positions < 1:
            errs.append("risk.max_open_positions must be >= 1")
        if not (0 < self.risk.max_drawdown_pct < 100):
            errs.append("risk.max_drawdown_pct must be in (0, 100)")
        if self.strategy.ema_fast >= self.strategy.ema_slow:
            errs.append("strategy.ema_fast must be < ema_slow")
        if self.strategy.warmup_bars < self.strategy.ema_trend + 10:
            errs.append("strategy.warmup_bars must exceed ema_trend by a margin")
        if self.universe.max_tradable > self.universe.top_n:
            errs.append("universe.max_tradable cannot exceed top_n")
        q = self.quote_currency
        if not q or "/" in q:
            errs.append(f"exchange.quote must be a bare currency code, got {q!r}")
        if not self.strategy.btc_symbol.endswith(f"/{q}"):
            errs.append(
                f"strategy.btc_symbol ({self.strategy.btc_symbol}) is not quoted in "
                f"{q}; the market-regime reference must trade in the same currency "
                f"as everything else or the regime filter reads a different market")
        mismatched = [s for s in self.universe.always_include
                      if not s.endswith(f"/{q}")]
        if mismatched:
            errs.append(f"universe.always_include contains symbols not quoted in "
                        f"{q}: {mismatched}")
        if self.universe.broad_source not in ("coingecko", "static", "none"):
            errs.append("universe.broad_source must be coingecko, static or none")
        if self.universe.broad_source == "static" and not self.universe.broad_static_assets:
            errs.append("universe.broad_source='static' requires broad_static_assets")
        if self.universe.broad_min_assets < 1:
            errs.append("universe.broad_min_assets must be >= 1")
        if self.universe.broad_collision_scan_limit < self.universe.broad_limit:
            errs.append("universe.broad_collision_scan_limit must be >= broad_limit")
        if not isinstance(self.universe.broad_symbol_overrides, dict):
            errs.append("universe.broad_symbol_overrides must be a table")
        else:
            for k, v in self.universe.broad_symbol_overrides.items():
                if not isinstance(v, str) or not v.strip():
                    errs.append(f"broad_symbol_overrides['{k}'] must be a "
                                f"non-empty provider asset id")
        if (self.universe.require_broad_universe
                and self.universe.broad_source == "none"):
            errs.append("universe.require_broad_universe needs a broad_source "
                        "(set one, or explicitly disable the requirement)")
        if self.execution.taker_fee_bps < 0 or self.execution.slippage_bps < 0:
            errs.append("fees and slippage must be non-negative")
        probe_span_days = (self.universe.age_probe_bars
                           * tf_ms(self.universe.age_probe_timeframe) / 86_400_000.0)
        if probe_span_days < self.universe.min_market_age_days:
            errs.append(
                f"universe.age_probe_bars ({self.universe.age_probe_bars} x "
                f"{self.universe.age_probe_timeframe}) reaches only "
                f"{probe_span_days:.0f} days, below the "
                f"{self.universe.min_market_age_days}-day age gate; every market "
                f"would be unverifiable")
        needed = self.required_history_bars()
        if self.exchange.max_history_bars < needed:
            errs.append(
                f"exchange.max_history_bars ({self.exchange.max_history_bars}) is "
                f"below the {needed} bars the universe filters require "
                f"(min_candles_1h={self.universe.min_candles_1h}, "
                f"min_market_age_days={self.universe.min_market_age_days}); every "
                f"symbol would be rejected for history it was never fetched")
        if self.exchange.ohlcv_limit < 1:
            errs.append("exchange.ohlcv_limit must be >= 1")
        if self.execution.max_quote_age_s <= 0:
            errs.append("execution.max_quote_age_s must be > 0")
        if self.execution.quote_ts_fallback not in ("local", "reject"):
            errs.append("execution.quote_ts_fallback must be 'local' or 'reject'")
        if self.execution.max_quote_deviation_pct <= 0:
            errs.append("execution.max_quote_deviation_pct must be > 0")
        if not (0 <= self.execution.risk_overshoot_tolerance_pct <= 25):
            errs.append("execution.risk_overshoot_tolerance_pct must be in [0, 25]")
        if self.telegram.enabled and not (self.telegram_token and self.telegram_chat_id):
            errs.append("telegram enabled but TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID are unset")
        return errs

    def required_history_bars(self, timeframe: str | None = None) -> int:
        """Bars of INDICATOR history a symbol needs -- nothing to do with age.

        Deliberately EXCLUDES min_market_age_days. Age is a calendar span and a
        property of the asset; folding it in here demanded 1080 hourly bars from
        venues that cap responses at 720, so every market failed forever, on
        every asset. Age is established separately at a coarse timeframe, where
        the same cap buys years instead of days -- see data/market_age.py.

          * min_candles_1h -- a bar count, directly
          * warmup_bars    -- indicator warm-up

        Deliberately NOT clamped to max_history_bars either: clamping would
        silently return a depth that cannot satisfy the filters, reintroducing
        the same class of bug one layer down. The ceiling is a validation error.
        """
        return max(self.universe.min_candles_1h, self.strategy.warmup_bars) + 5

    @property
    def quote_currency(self) -> str:
        """THE quote currency. Every pair in the system is built from this."""
        return self.exchange.quote

    def market_for(self, base: str) -> str:
        """The market symbol for a base asset on this venue, e.g. BTC -> BTC/USD."""
        return f"{base.upper()}/{self.quote_currency}"

    def resolve_symbols(self, explicit: set[str] | None = None) -> None:
        """Complete every configured BASE into a pair using the effective quote.

        Called after the TOML and any --exchange/--quote override are applied,
        so the quote currency has exactly one chance to be decided and every
        symbol in the system is built from that one decision. Anything the
        operator wrote explicitly is left alone and checked by validate().
        """
        explicit = explicit or set()
        if "btc_symbol" not in explicit or not self.strategy.btc_symbol:
            self.strategy.btc_symbol = self.market_for(self.strategy.btc_base)
        if "always_include" not in explicit or not self.universe.always_include:
            self.universe.always_include = [
                self.market_for(b) for b in self.universe.always_include_bases]

    def exchange_label(self) -> str:
        """The single string every surface should show for "which venue"."""
        return f"{self.exchange.name}/{self.exchange.quote}"

    def strategy_fingerprint(self) -> dict[str, Any]:
        """Recorded against every trade so historical results stay bound to the
        settings that produced them (spec section 15)."""
        return {
            "strategy": asdict(self.strategy),
            "risk": asdict(self.risk),
            "execution": asdict(self.execution),
        }


def _apply(section: Any, data: dict) -> None:
    for k, v in data.items():
        if not hasattr(section, k):
            raise ConfigError(f"unknown config key: {k}")
        cur = getattr(section, k)
        if isinstance(cur, bool) and not isinstance(v, bool):
            raise ConfigError(f"config key {k} must be a boolean")
        setattr(section, k, v)


def load_config(path: str | Path = "config/config.toml",
                env_path: str | Path = ".env") -> Config:
    load_dotenv(env_path)
    cfg = Config()
    explicit: set[str] = set()
    p = Path(path)
    if p.exists():
        raw = tomllib.loads(p.read_text(encoding=TEXT_ENCODING))
        for name, data in raw.items():
            if not hasattr(cfg, name):
                raise ConfigError(f"unknown config section: [{name}]")
            _apply(getattr(cfg, name), data)
            explicit.update(data.keys())
    # Operational overrides. Switching venue is the single most common thing an
    # operator needs to do (geo-blocks, outages), so it must not require editing
    # a config file -- a mistyped TOML line is a worse failure than a wrong venue.
    exchange = os.environ.get("CRYPTO_EDGE_EXCHANGE", "").strip()
    if exchange:
        cfg.exchange.name = exchange
        cfg.exchange_source = "--exchange / CRYPTO_EDGE_EXCHANGE override"
    quote_ccy = os.environ.get("CRYPTO_EDGE_QUOTE", "").strip()
    if quote_ccy:
        cfg.exchange.quote = quote_ccy
        if not exchange:
            cfg.exchange_source = "--quote / CRYPTO_EDGE_QUOTE override"

    cfg.telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    cfg.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if os.environ.get("TELEGRAM_ENABLED", "").lower() in ("0", "false", "no"):
        cfg.telegram.enabled = False
    # Every pair in the system is completed here, AFTER the quote currency has
    # been finally decided by the TOML plus any override. One decision, one
    # place, so nothing can be left pointing at a different currency's market.
    cfg.resolve_symbols(explicit)

    # Belt and braces: the environment can never switch on live trading.
    cfg.safety.mode = "PAPER"
    cfg.safety.live_trading_enabled = False
    return cfg
