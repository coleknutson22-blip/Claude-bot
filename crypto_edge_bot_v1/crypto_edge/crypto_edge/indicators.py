"""Technical indicators on numpy arrays.

Every function here is *causal*: element i of the output depends only on input
elements 0..i. Warm-up regions are filled with NaN rather than a plausible-
looking number, so downstream code is forced to notice insufficient history
instead of silently trading on a half-formed indicator.
"""
from __future__ import annotations

import numpy as np


def _nan(n: int) -> np.ndarray:
    return np.full(n, np.nan, dtype=float)


def ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential MA seeded with an SMA of the first `period` values."""
    n = len(values)
    out = _nan(n)
    if n < period or period < 1:
        return out
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = float(np.mean(values[:period]))
    for i in range(period, n):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def sma(values: np.ndarray, period: int) -> np.ndarray:
    n = len(values)
    out = _nan(n)
    if n < period or period < 1:
        return out
    c = np.cumsum(np.insert(values.astype(float), 0, 0.0))
    out[period - 1:] = (c[period:] - c[:-period]) / period
    return out


def rma(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing -- the basis of ATR, ADX and RSI."""
    n = len(values)
    out = _nan(n)
    if n < period or period < 1:
        return out
    out[period - 1] = float(np.mean(values[:period]))
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + values[i]) / period
    return out


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(close)
    tr = _nan(n)
    if n == 0:
        return tr
    tr[0] = high[0] - low[0]
    if n > 1:
        prev = close[:-1]
        tr[1:] = np.maximum.reduce([
            high[1:] - low[1:],
            np.abs(high[1:] - prev),
            np.abs(low[1:] - prev),
        ])
    return tr


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    return rma(true_range(high, low, close), period)


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    out = _nan(n)
    if n <= period:
        return out
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = rma(gain, period)
    al = rma(loss, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(al > 0, ag / al, np.inf)
        vals = 100.0 - (100.0 / (1.0 + rs))
    vals = np.where(al == 0, 100.0, vals)
    vals = np.where((ag == 0) & (al == 0), 50.0, vals)
    out[1:] = vals
    return out


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
        period: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (adx, plus_di, minus_di)."""
    n = len(close)
    if n <= period + 1:
        return _nan(n), _nan(n), _nan(n)
    up = np.diff(high)
    dn = -np.diff(low)
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(high, low, close)[1:]
    atr_s = rma(tr, period)
    plus_s = rma(plus_dm, period)
    minus_s = rma(minus_dm, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100.0 * plus_s / atr_s
        mdi = 100.0 * minus_s / atr_s
        dx = 100.0 * np.abs(pdi - mdi) / (pdi + mdi)
    dx = np.where(np.isfinite(dx), dx, np.nan)
    # ADX = Wilder-smoothed DX, which itself only becomes valid after warm-up
    adx_vals = _nan(len(dx))
    valid = np.where(np.isfinite(dx))[0]
    if len(valid) >= period:
        start = valid[0]
        seg = dx[start:]
        sm = rma(np.nan_to_num(seg, nan=0.0), period)
        adx_vals[start:] = sm
    out_adx, out_p, out_m = _nan(n), _nan(n), _nan(n)
    out_adx[1:] = adx_vals
    out_p[1:] = pdi
    out_m[1:] = mdi
    return out_adx, out_p, out_m


def donchian(high: np.ndarray, low: np.ndarray, lookback: int,
             exclude_current: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Rolling channel. With `exclude_current=True` the band at bar i covers
    bars i-lookback .. i-1, so a breakout is measured against prior structure
    and the current bar cannot break its own high."""
    n = len(high)
    up, lo = _nan(n), _nan(n)
    if lookback < 1:
        return up, lo
    off = 1 if exclude_current else 0
    for i in range(n):
        end = i + 1 - off
        start = end - lookback
        if start < 0 or end <= 0:
            continue
        up[i] = float(np.max(high[start:end]))
        lo[i] = float(np.min(low[start:end]))
    return up, lo


def rel_volume(volume: np.ndarray, period: int = 24) -> np.ndarray:
    """Current volume relative to the average of the *preceding* `period` bars."""
    n = len(volume)
    out = _nan(n)
    if n <= period:
        return out
    avg = sma(volume, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = np.where(avg[:-1] > 0, volume[1:] / avg[:-1], np.nan)
    return out


def roc(close: np.ndarray, period: int) -> np.ndarray:
    """Rate of change over `period` bars, as a percentage."""
    n = len(close)
    out = _nan(n)
    if n <= period:
        return out
    prev = close[:-period]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[period:] = np.where(prev > 0, (close[period:] / prev - 1.0) * 100.0, np.nan)
    return out


def realized_vol(close: np.ndarray, period: int = 24) -> np.ndarray:
    """Annualisation-free stdev of log returns, in percent."""
    n = len(close)
    out = _nan(n)
    if n <= period + 1:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.diff(np.log(close))
    for i in range(period, len(lr)):
        w = lr[i - period + 1:i + 1]
        if np.all(np.isfinite(w)):
            out[i + 1] = float(np.std(w, ddof=1) * 100.0)
    return out


def log_returns(close: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diff(np.log(close))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two equal-length return series."""
    n = min(len(a), len(b))
    if n < 8:
        return 0.0
    x, y = a[-n:], b[-n:]
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8:
        return 0.0
    x, y = x[m], y[m]
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    c = float(np.corrcoef(x, y)[0, 1])
    return c if np.isfinite(c) else 0.0


def last_valid(arr: np.ndarray) -> float:
    """Most recent finite value, or NaN. Used to read an indicator's current
    reading without accidentally picking up a warm-up NaN."""
    if len(arr) == 0:
        return float("nan")
    v = arr[-1]
    return float(v) if np.isfinite(v) else float("nan")
