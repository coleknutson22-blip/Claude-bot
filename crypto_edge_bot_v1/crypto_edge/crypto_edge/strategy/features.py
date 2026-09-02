"""Multi-timeframe momentum features for the aggressive strategy.

EVERYTHING HERE IS CAUSAL. Each function reads only bars at or before the last
element of the array it is given, and the engine only ever hands it closed
candles. Rolling extremes exclude the current bar, because a "swing high" that
includes the bar being evaluated is just that bar's own high -- a comparison
with itself that always reads as "at the high".

WHY MOMENTUM IS MEASURED IN ATR UNITS
-------------------------------------
A 2% move means something entirely different on a stablecoin pair than on a
small-cap that routinely swings 15% a day. Ranking on raw percentage sorts by
volatility, not by opportunity, and fills the shortlist with whatever happens to
be the twitchiest asset that hour. Every momentum figure is therefore also
expressed as a multiple of the asset's own recent ATR, and the scoring uses that
normalised form.

WHICH TIMEFRAME SUPPLIES WHICH WINDOW
-------------------------------------
Each window is read from the coarsest timeframe that still resolves it, so a
24-hour figure is 24 hourly bars rather than 288 five-minute ones:

    30m, 1h  -> 5m bars      2h, 3h -> 15m bars      6h, 24h -> 1h bars
"""
from __future__ import annotations

import numpy as np

from ..indicators import (atr, donchian, ema, last_valid, realized_vol,
                          rel_volume)
from ..models import Series
from ..timeutils import tf_ms

# window name -> (timeframe, minutes) -- the single place this mapping lives
MOMENTUM_WINDOWS: dict[str, tuple[str, int]] = {
    "30m": ("5m", 30),
    "1h": ("5m", 60),
    "2h": ("15m", 120),
    "3h": ("15m", 180),
    "6h": ("1h", 360),
    "24h": ("1h", 1440),
}

NAN = float("nan")


def log_roc(close: np.ndarray, period: int) -> float:
    """Return over `period` bars as a LOG percentage: 100 * ln(p_t / p_t-n).

    Log rather than simple percentage, because a long/short strategy has to
    treat equal-and-opposite moves as equal. A simple return is asymmetric under
    price inversion -- a +10% move reflects to -9.09%, not -10% -- so scoring on
    it makes a short of identical magnitude look systematically weaker than the
    long, entirely as an artefact of the arithmetic. Log returns negate exactly.
    """
    if period < 1 or len(close) <= period:
        return NAN
    a, b = float(close[-1 - period]), float(close[-1])
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
        return NAN
    return float(np.log(b / a) * 100.0)


def bars_for(timeframe: str, minutes: int) -> int:
    """How many bars of `timeframe` span `minutes`. At least one."""
    step_min = tf_ms(timeframe) / 60_000.0
    return max(1, int(round(minutes / step_min)))


def _finite(x: float) -> bool:
    return bool(np.isfinite(x))


def trend_quality(close: np.ndarray, period: int) -> tuple[float, float]:
    """Least-squares slope of LOG price, and the R^2 of that fit.

    Slope answers "which way and how fast", R^2 answers "how orderly". A steep
    line through noise and a gentle line through a clean trend are different
    setups, and a single momentum number cannot tell them apart. Logs make the
    slope a compounding rate, so it is comparable across price scales.
    """
    if period < 3 or len(close) < period:
        return NAN, NAN
    y = np.asarray(close[-period:], dtype=float)
    if np.any(~np.isfinite(y)) or np.any(y <= 0):
        return NAN, NAN
    y = np.log(y)
    x = np.arange(period, dtype=float)
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = float(np.dot(x_c, x_c))
    if denom <= 0:
        return NAN, NAN
    slope = float(np.dot(x_c, y_c) / denom)
    ss_tot = float(np.dot(y_c, y_c))
    resid = y_c - slope * x_c
    ss_res = float(np.dot(resid, resid))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else NAN
    # per-bar compounding rate, in percent
    return (float(np.expm1(slope) * 100.0), r2)


def ema_structure(close: np.ndarray, fast: int = 9, mid: int = 21,
                  slow: int = 50) -> float:
    """Stack alignment in [-1, +1]. +1 fully bullish, -1 fully bearish.

    Graded rather than boolean: three of the four conditions holding is a real
    and common state, and collapsing it to False throws away the distinction
    between "not perfect" and "actively hostile" -- which is exactly the
    distinction this strategy exists to trade.
    """
    e_f, e_m, e_s = (last_valid(ema(close, n)) for n in (fast, mid, slow))
    price = float(close[-1]) if len(close) else NAN
    if not all(_finite(v) for v in (e_f, e_m, e_s, price)):
        return NAN
    votes = [
        1.0 if e_f > e_m else -1.0,
        1.0 if e_m > e_s else -1.0,
        1.0 if price > e_f else -1.0,
        1.0 if price > e_s else -1.0,
    ]
    return float(sum(votes) / len(votes))


def atr_expansion(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  period: int = 14, lookback: int = 24) -> float:
    """Current ATR relative to its own recent median. >1 expanding, <1 quiet."""
    a = atr(high, low, close, period)
    now = last_valid(a)
    window = a[-lookback:] if len(a) >= lookback else a
    finite = window[np.isfinite(window)]
    if not _finite(now) or finite.size == 0:
        return NAN
    med = float(np.median(finite))
    return now / med if med > 0 else NAN


def swing_proximity(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                    lookback: int, atr_value: float) -> tuple[float, float]:
    """Distance to the recent swing high and low, in ATR units.

    Both are returned as POSITIVE distances travelled to reach the level, so
    "close to the high" and "close to the low" are read the same way on either
    side. The current bar is excluded -- a level that includes the bar being
    measured is not a level.
    """
    if atr_value <= 0 or not _finite(atr_value):
        return NAN, NAN
    dch, dcl = donchian(high, low, lookback, exclude_current=True)
    hi, lo = last_valid(dch), last_valid(dcl)
    price = float(close[-1])
    if not all(_finite(v) for v in (hi, lo, price)):
        return NAN, NAN
    return ((hi - price) / atr_value, (price - lo) / atr_value)


def momentum_set(frames: dict[str, Series]) -> dict[str, float]:
    """Every configured momentum window, as a percentage return."""
    out: dict[str, float] = {}
    for name, (tf, minutes) in MOMENTUM_WINDOWS.items():
        s = frames.get(tf)
        n = bars_for(tf, minutes)
        if s is None or len(s) <= n:
            out[name] = NAN
            continue
        out[name] = log_roc(s.close, n)
    return out


def compute(frames: dict[str, Series], *, btc: dict[str, Series] | None = None,
            breadth_pct: float = 50.0, btc_regime: str = "unknown",
            btc_regime_score: float = 50.0, meta: dict | None = None) -> dict:
    """The full feature vector for one symbol.

    `frames` maps timeframe -> closed-candle Series ("5m", "15m", "1h").
    `btc` is the same structure for the regime reference asset, used for
    relative strength. Missing inputs produce NaN features rather than
    exceptions; the caller decides what is fatal.
    """
    meta = meta or {}
    f5, f15, f1h = frames.get("5m"), frames.get("15m"), frames.get("1h")
    out: dict = {"timeframes_present": sorted(k for k, v in frames.items() if v is not None)}

    price = float(f5.close[-1]) if f5 is not None and len(f5) else (
        float(f15.close[-1]) if f15 is not None and len(f15) else NAN)
    out["price"] = price

    # --- volatility -------------------------------------------------------
    if f15 is not None and len(f15) > 15:
        atr15 = last_valid(atr(f15.high, f15.low, f15.close, 14))
        out["atr"] = atr15
        # Divided by the 15m close, NOT the 5m price. The two frames close at
        # different instants, so mixing them makes ATR% a ratio of two
        # unrelated numbers -- and if the frames ever disagree badly (a stale
        # or mismatched fetch) the result is silently meaningless rather than
        # obviously wrong. Same series, same instant, one honest ratio.
        ref15 = float(f15.close[-1])
        out["atr_pct"] = (atr15 / ref15 * 100.0) if (ref15 > 0 and _finite(atr15)) else NAN
        out["atr_expansion"] = atr_expansion(f15.high, f15.low, f15.close, 14, 24)
    else:
        out["atr"] = out["atr_pct"] = out["atr_expansion"] = NAN
    out["short_vol_pct"] = (last_valid(realized_vol(f5.close, 24))
                            if f5 is not None and len(f5) > 25 else NAN)

    # --- momentum ---------------------------------------------------------
    mom = momentum_set(frames)
    out["momentum"] = mom
    atr_pct = out["atr_pct"]
    # Normalised: how many of this asset's own ATRs the move represents.
    out["momentum_atr"] = {
        k: (v / atr_pct if _finite(v) and _finite(atr_pct) and atr_pct > 0 else NAN)
        for k, v in mom.items()}

    # --- structure --------------------------------------------------------
    out["ema_struct_5m"] = ema_structure(f5.close) if f5 is not None and len(f5) > 50 else NAN
    out["ema_struct_15m"] = ema_structure(f15.close) if f15 is not None and len(f15) > 50 else NAN
    out["ema_struct_1h"] = ema_structure(f1h.close) if f1h is not None and len(f1h) > 50 else NAN

    slope15, r2_15 = (trend_quality(f15.close, 24) if f15 is not None else (NAN, NAN))
    slope1h, r2_1h = (trend_quality(f1h.close, 24) if f1h is not None else (NAN, NAN))
    out["slope_15m_pct"], out["trend_r2_15m"] = slope15, r2_15
    out["slope_1h_pct"], out["trend_r2_1h"] = slope1h, r2_1h

    # --- participation ----------------------------------------------------
    out["rel_volume"] = (last_valid(rel_volume(f15.volume, 24))
                         if f15 is not None and len(f15) > 25 else NAN)

    # --- swing structure --------------------------------------------------
    if f15 is not None and len(f15) > 30 and _finite(out["atr"]):
        hi_d, lo_d = swing_proximity(f15.high, f15.low, f15.close, 24, out["atr"])
    else:
        hi_d, lo_d = NAN, NAN
    out["dist_to_swing_high_atr"], out["dist_to_swing_low_atr"] = hi_d, lo_d

    # --- relative strength vs BTC ----------------------------------------
    btc_mom = momentum_set(btc) if btc else {}
    out["btc_momentum"] = btc_mom
    rs = {}
    for k in ("1h", "6h", "24h"):
        a, b = mom.get(k, NAN), btc_mom.get(k, NAN)
        rs[k] = (a - b) if (_finite(a) and _finite(b)) else NAN
    out["rel_strength"] = rs

    # --- market context ---------------------------------------------------
    out["breadth_pct"] = breadth_pct
    out["btc_regime"] = btc_regime
    out["btc_regime_score"] = btc_regime_score

    # --- tradability ------------------------------------------------------
    # How far the frames disagree on price. Legitimately non-zero -- each
    # timeframe's last CLOSED bar ends at a different instant -- but a large
    # value means the frames are not the same market at the same time.
    closes = [float(fr.close[-1]) for fr in (f5, f15, f1h)
              if fr is not None and len(fr)]
    out["frame_price_spread_pct"] = (
        (max(closes) - min(closes)) / min(closes) * 100.0
        if len(closes) > 1 and min(closes) > 0 else 0.0)
    out["dollar_volume"] = float(meta.get("dollar_volume", 0.0))
    out["spread_bps"] = float(meta.get("spread_bps", NAN))
    out["bar_open_ms"] = int(f15.open_ms[-1]) if f15 is not None and len(f15) else 0
    return out


def required_bars() -> dict[str, int]:
    """Minimum closed bars per timeframe for the feature set to be computable.

    Driven by the longest thing each timeframe is asked for -- the 50-period EMA
    and the 24-bar momentum window -- plus a margin so an indicator that is
    still warming up does not silently return NaN on its first valid bar.
    """
    need = {"5m": 55, "15m": 55, "1h": 55}
    for name, (tf, minutes) in MOMENTUM_WINDOWS.items():
        need[tf] = max(need.get(tf, 0), bars_for(tf, minutes) + 2)
    return need
