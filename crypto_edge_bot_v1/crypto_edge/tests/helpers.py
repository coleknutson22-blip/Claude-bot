"""Shared test fixtures."""
from __future__ import annotations

import logging

# keep the suite output clean: the engine logs a lot by design
logging.getLogger("ce").addHandler(logging.NullHandler())
logging.getLogger("ce").propagate = False
for _ch in ("app", "data", "strategy", "trades", "performance", "telegram"):
    _lg = logging.getLogger(f"ce.{_ch}")
    _lg.addHandler(logging.NullHandler())
    _lg.propagate = False

import tempfile
from pathlib import Path

import numpy as np

from crypto_edge.config import Config
from crypto_edge.data.fixture_feed import FixtureFeed, make_series
from crypto_edge.models import MarketMeta
from crypto_edge.storage import db
from crypto_edge.storage.repo import Repo

START_MS = 1_700_000_000_000  # 2023-11-14T22:13:20Z, aligned to the hour below
HOUR = 3_600_000
ALIGNED = (START_MS // HOUR) * HOUR


def temp_repo() -> tuple[Repo, str]:
    path = Path(tempfile.mkdtemp()) / "test.db"
    conn = db.connect(path)
    db.init_db(conn)
    return Repo(conn), str(path)


def open_repo(path: str) -> Repo:
    conn = db.connect(path)
    db.init_db(conn)
    return Repo(conn)


def default_meta(symbol: str = "SOL/USDT") -> MarketMeta:
    return MarketMeta(symbol, symbol.split("/")[0], "USDT", True,
                      amount_precision=4, price_precision=4,
                      min_amount=0.0001, min_cost=10.0)


def test_config(**overrides) -> Config:
    cfg = Config()
    cfg.telegram.enabled = False
    cfg.execution.starting_equity = 10_000.0
    cfg.execution.use_book_spread = False
    cfg.engine.poll_seconds = 0
    cfg.safety.candle_close_buffer_s = 0
    for path, val in overrides.items():
        section, _, key = path.partition(".")
        setattr(getattr(cfg, section), key, val)
    return cfg


def uptrend_closes(n: int = 400, start: float = 100.0, drift: float = 0.004,
                   seed: int = 7, breakout_at: int | None = None) -> np.ndarray:
    """Deterministic gently-trending series with low noise, so a breakout
    strategy has something to find."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 0.004, n)
    closes = start * np.exp(np.cumsum(steps))
    if breakout_at is not None and breakout_at < n:
        closes[breakout_at:] *= 1.05
    return closes


def flat_closes(n: int = 400, start: float = 100.0, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(0.0, 0.002, n)))


def build_feed(symbols: dict[str, np.ndarray], clock_ms: int | None = None,
               htf_from_1h: bool = True) -> FixtureFeed:
    """Build a FixtureFeed with 1h and 4h series for each symbol."""
    series = {}
    markets = {}
    for sym, closes in symbols.items():
        s1 = make_series(sym, "1h", closes, ALIGNED, volume=1000.0)
        series[(sym, "1h")] = s1
        if htf_from_1h:
            c4 = closes[::4]
            series[(sym, "4h")] = make_series(sym, "4h", c4, ALIGNED, volume=4000.0)
        markets[sym] = default_meta(sym)
    return FixtureFeed(series, markets, clock_ms=clock_ms)


def breakout_closes(n: int = 400, drift: float = 0.0006, vol: float = 0.006,
                    seed: int = 11, breakout: float = 1.012,
                    start: float = 100.0) -> np.ndarray:
    """A realistic noisy uptrend whose FINAL bar clears the prior 48-bar high.

    Deliberately noisy: a perfectly monotonic series drives RSI and ADX to 100,
    which no real market produces and which the exhaustion filter would reject.
    """
    rng = np.random.default_rng(seed)
    c = start * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    prior_high = c[-49:-1].max() * 1.004
    c[-1] = max(c[-2], prior_high) * breakout
    return c


def trend_closes(n: int = 400, drift: float = 0.0006, vol: float = 0.006,
                 seed: int = 4, start: float = 30_000.0) -> np.ndarray:
    """Plain noisy uptrend, no engineered breakout (used for BTC regime)."""
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(drift, vol, n)))


def recent_start_ms(n_bars: int, tf: str = "1h", now: int | None = None) -> int:
    """Start timestamp such that the LAST bar of an n-bar series has just
    closed relative to `now`. Keeps engine staleness checks satisfied."""
    from crypto_edge.timeutils import floor_to_tf, now_ms, tf_ms
    now = now if now is not None else now_ms()
    step = tf_ms(tf)
    last_open = floor_to_tf(now, tf) - step
    return last_open - (n_bars - 1) * step


def engine_config():
    """Config tuned for deterministic offline engine tests.

    Only the universe-age/volatility gates are relaxed, because synthetic
    fixtures are short-lived by nature. Every risk, execution and strategy
    parameter is left at its production default.
    """
    cfg = test_config()
    cfg.universe.min_candles_1h = 300
    cfg.universe.min_market_age_days = 5
    cfg.universe.min_atr_pct = 0.05
    cfg.universe.refresh_minutes = 0
    cfg.telegram.enabled = False
    cfg.safety.candle_close_buffer_s = 0
    return cfg
