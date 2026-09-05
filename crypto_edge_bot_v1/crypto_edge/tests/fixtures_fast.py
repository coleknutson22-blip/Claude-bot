"""Synthetic multi-timeframe fixtures for the aggressive strategy.

Built from a log-return path so the three timeframes are CONSISTENT with each
other -- a 15m bar spans three 5m bars and a 1h bar spans twelve -- rather than
three unrelated random walks that happen to share a symbol.

`mirror()` reflects a path geometrically around its first close (p -> p0^2/p),
which negates every log return exactly and swaps highs with lows. That is what
makes a long/short symmetry test meaningful: the bearish fixture is not merely
"a different downward series", it is the arithmetic mirror of the bullish one.
"""
from __future__ import annotations

import numpy as np

from crypto_edge.models import Series
from crypto_edge.timeutils import tf_ms

BASE_MS = 1_700_000_000_000
TF_BARS = {"5m": 1, "15m": 3, "1h": 12}      # 5m bars per bar of that timeframe
REGIME_BARS = {"4h": 48}                     # opt-in; Strategy A's regime frame


def _ohlc_from_closes(symbol, tf, closes, seed, vol_mult=1.0):
    n = len(closes)
    step = tf_ms(tf)
    start = BASE_MS // step * step - (n - 1) * step
    opens_ms = np.array([start + i * step for i in range(n)], dtype=np.int64)
    c = np.asarray(closes, dtype=float)
    o = np.concatenate([[c[0]], c[:-1]])
    rng = np.random.default_rng(seed)
    wick = np.abs(rng.normal(0, 0.0030 * vol_mult, n))
    hi = np.maximum(o, c) * (1.0 + wick)
    lo = np.minimum(o, c) * (1.0 - wick)
    volume = np.full(n, 1000.0) * (1.0 + rng.normal(0, 0.04, n))
    return Series(symbol, tf, opens_ms, o, hi, lo, c, np.abs(volume))


def path(n_5m: int, drift: float, seed: int, sigma: float = 0.0018) -> np.ndarray:
    """A 5m log-return path. `drift` is the per-5m-bar log drift."""
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(drift, sigma, n_5m)))


def choppy_path(n_5m: int, seed: int, sigma: float = 0.0018,
                pull: float = 0.05) -> np.ndarray:
    """A MEAN-REVERTING path: chop, not a random walk.

    A zero-drift random walk is not flat -- it wanders, and over 300 bars it
    trends by chance often enough that a "flat market" fixture built that way
    produces genuine directional readings. Pulling each step back toward the
    mean gives a market with no persistent direction, which is what "choppy"
    actually means.
    """
    rng = np.random.default_rng(seed)
    x = 0.0
    out = np.empty(n_5m)
    for i in range(n_5m):
        x += -pull * x + rng.normal(0.0, sigma)
        out[i] = x
    return 100.0 * np.exp(out)


def oscillating_path(n_5m: int, period: int = 24, amp: float = 0.02,
                     seed: int = 5) -> np.ndarray:
    """A market with NO net direction at all: a clean cycle plus light noise.

    The period is deliberately SHORT -- two hours by default. A slow cycle is
    not directionless when you measure it: a sine wave sampled on its rising leg
    has been going up for six hours by every window that matters, and the
    strategy is right to say so. Only a cycle fast enough for the 2h/3h/6h
    windows to straddle several turns has genuinely no direction to find.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_5m)
    wave = amp * np.sin(2 * np.pi * t / period)
    return 100.0 * np.exp(wave + rng.normal(0, 0.0004, n_5m))


def resample(closes_5m: np.ndarray, bars: int) -> np.ndarray:
    """Take every `bars`-th close: the close of each coarser bar."""
    usable = len(closes_5m) - (len(closes_5m) % bars)
    return closes_5m[bars - 1:usable:bars]


def frames(drift: float, seed: int = 11, symbol: str = "SOL/USD",
           n_5m: int = 3600, sigma: float = 0.0018,
           chop: bool = False, oscillate: bool = False,
           include_4h: bool = False) -> dict[str, Series]:
    """One coherent market across 5m, 15m and 1h."""
    if oscillate:
        p5 = oscillating_path(n_5m, seed=seed)
    elif chop:
        p5 = choppy_path(n_5m, seed, sigma)
    else:
        p5 = path(n_5m, drift, seed, sigma)
    out = {}
    wanted = dict(TF_BARS)
    if include_4h:
        # Strategy A's regime timeframe. Off by default so the aggressive
        # tests, which never read it, do not pay to build it.
        wanted.update(REGIME_BARS)
    for tf, bars in wanted.items():
        closes = resample(p5, bars) if bars > 1 else p5
        out[tf] = _ohlc_from_closes(symbol, tf, closes[-320:], seed + bars,
                                    vol_mult=np.sqrt(bars))
    return out


def mirror_series(s: Series, pivot: float) -> Series:
    """Reflect around `pivot`: p -> pivot^2 / p. Highs become lows."""
    k = pivot * pivot
    return Series(s.symbol, s.timeframe, s.open_ms.copy(),
                  k / s.open, k / s.low, k / s.high, k / s.close, s.volume.copy())


def mirror(fr: dict[str, Series]) -> dict[str, Series]:
    """Mirror every timeframe around ONE SHARED pivot.

    Reflecting each frame around its own first close would send the three
    timeframes to unrelated price levels -- they would no longer describe one
    market, and any cross-frame feature computed from them would be nonsense.
    """
    pivot = float(fr["15m"].close[0])
    return {tf: mirror_series(s, pivot) for tf, s in fr.items()}


def restamp_to_now(s: Series, now: int | None = None) -> Series:
    """Same prices, shifted so the LAST bar has just closed relative to `now`.

    The paths above are anchored to a fixed BASE_MS so they stay reproducible.
    An engine-level test also has to satisfy the freshness guards, which are
    about the clock rather than the prices, so the two concerns are separated:
    build the market once, then move it to the present.
    """
    from crypto_edge.timeutils import floor_to_tf, now_ms as _now
    step = tf_ms(s.timeframe)
    now = now if now is not None else _now()
    start = floor_to_tf(now, s.timeframe) - step - (len(s) - 1) * step
    ms = np.array([start + i * step for i in range(len(s))], dtype=np.int64)
    return Series(s.symbol, s.timeframe, ms, s.open, s.high, s.low,
                  s.close, s.volume)


def engine_feed(symbols: list[str], *, n_5m: int = 6000, seed: int = 700,
                now: int | None = None):
    """A FixtureFeed serving 5m, 15m, 1h AND 4h -- what a real venue serves.

    `helpers.build_feed` holds only 1h and 4h, and `FixtureFeed._derive` can
    only resample COARSER. So a 1h fixture cannot answer a 5m request at all,
    and Strategy B run against one silently evaluates nothing: every deep fetch
    raises DataUnavailable and lands in `fetch_failures`. That is a property of
    the fixture, not of the wiring, and this builder is what tells the two
    apart.
    """
    from crypto_edge.data.fixture_feed import FixtureFeed
    from crypto_edge.models import MarketMeta

    series, markets = {}, {}
    for i, sym in enumerate(symbols):
        # Spread the drift across the symbols so the shortlist has both sides
        # to rank rather than one direction repeated.
        fr = frames((i - 2) * 0.00035, seed=seed + i, symbol=sym,
                    n_5m=n_5m, include_4h=True)
        for tf, s in fr.items():
            series[(sym, tf)] = restamp_to_now(s, now)
        base, _, quote = sym.partition("/")
        markets[sym] = MarketMeta(sym, base, quote or "USDT", True,
                                  6, 6, 1e-6, 5.0)
    return FixtureFeed(series, markets), markets
