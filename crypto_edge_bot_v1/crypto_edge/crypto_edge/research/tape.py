"""The per-bar tape: every input the live exit engine reads, stored per bar.

WHY A TAPE RATHER THAN MORE FLAGS
---------------------------------
Stage A recorded whether each candidate TARGET was reached before the INITIAL
stop. That cannot reproduce the live strategy, because the live stop MOVES --
breakeven at 1R, then a chandelier trail from 1.5R -- and five further exits
(momentum invalidation, two time stops, hostile regime, forced short close) can
close a trade before any target is touched.

Adding a flag per question would commit this recorder to one fixed set of
policies. Storing the inputs instead means any future exit rule can be replayed
exactly, offline, without re-running anything or fetching the market again.

WHAT THE LIVE ENGINE ACTUALLY READS, PER EVALUATION
---------------------------------------------------
From `AggressiveRuntime.manage`:

    candle   = s.last_candle()                 # 5m OHLC -> the protective stop
    price    = float(s.close[-1])              # 5m close -> every non-stop exit
    struct   = feat.ema_structure(s15.close)   # 15m      -> momentum invalidation
    src      = s15 if len(s15) > 20 else s     # 15m      -> the chandelier ATR
    a_val    = last_valid(atr(src.high, src.low, src.close, 14))
    regime   = ctx.btc_regime                  # context  -> hostile-regime exit

So the tape stores the 5m OHLC, the 15m ATR the trail would have used, the 15m
EMA structure, and the BTC regime in force -- each resolved AS OF that bar.

CAUSALITY IS THE WHOLE DIFFICULTY
---------------------------------
The naive implementation computes `last_valid(atr(s15...))` once and writes that
number against every bar in the path. That is a look-ahead error of the worst
kind: it stamps a value derived from the END of the window onto bars at its
start, and the replayed trail would then be trailing on information the live
bot could not have had.

Every indicator here is therefore resolved at the last 15m bar that had CLOSED
by the 5m bar's own close time, using a forward-filled prefix -- which is what
`last_valid` means when you are standing at that bar rather than at the end.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..indicators import atr, ema
from ..timeutils import tf_ms

MIN_MS = 60_000
BAR_MS = tf_ms("5m")

# Matched to the live call sites. `manage()` uses ema_structure's defaults and
# atr(..., 14); these mirror them so a change there is a visible change here.
ATR_PERIOD = 14
EMA_FAST, EMA_MID, EMA_SLOW = 9, 21, 50
# The live guards: `s15 if len(s15) > 20 else s` for the ATR source, and
# `len(s15) > 50` before ema_structure is computed at all.
MIN_BARS_FOR_ATR15 = 20
MIN_BARS_FOR_STRUCT = 50


@dataclass
class TapeBar:
    """One closed 5m bar plus the context the exit engine would have seen."""
    open_ms: int
    elapsed_min: float
    open: float
    high: float
    low: float
    close: float
    atr: float | None           # the value the chandelier would have used
    atr_tf: str                 # which series it came from: "15m" or "5m"
    ema_struct_15m: float | None
    btc_regime: str


def _ffill(a: np.ndarray) -> np.ndarray:
    """Carry the last finite value forward.

    This is exactly what `last_valid` computes when evaluated at index i rather
    than at the end of the array, and it is the difference between a causal
    tape and one contaminated by its own future.
    """
    out = np.asarray(a, dtype=float).copy()
    valid = np.isfinite(out)
    if not valid.any():
        return out
    idx = np.where(valid, np.arange(len(out)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = out[idx]
    out[:np.argmax(valid)] = np.nan       # nothing valid yet at the very start
    return out


def _structure_series(close: np.ndarray) -> np.ndarray:
    """`ema_structure` evaluated at EVERY index, not just the last.

    Mirrors features.ema_structure exactly: four votes in [-1, +1], built from
    forward-filled EMAs so each index sees only what had happened by then.
    """
    e_f, e_m, e_s = (_ffill(ema(close, n)) for n in (EMA_FAST, EMA_MID, EMA_SLOW))
    price = np.asarray(close, dtype=float)
    votes = (np.where(e_f > e_m, 1.0, -1.0)
             + np.where(e_m > e_s, 1.0, -1.0)
             + np.where(price > e_f, 1.0, -1.0)
             + np.where(price > e_s, 1.0, -1.0))
    out = votes / 4.0
    bad = ~(np.isfinite(e_f) & np.isfinite(e_m) & np.isfinite(e_s)
            & np.isfinite(price))
    out[bad] = np.nan
    return out


class TapeContext:
    """Resolves the live exit engine's inputs as of any 5m bar close.

    Built once per symbol per cycle and then queried per bar, so the 15m
    indicator arrays are computed once rather than per bar.
    """

    def __init__(self, s5, s15=None, regime_at=None) -> None:
        self.s5 = s5
        self.s15 = s15
        self._regime_at = regime_at or (lambda ms: "unknown")

        self._atr15 = self._close15 = self._struct15 = None
        if s15 is not None and len(s15) > MIN_BARS_FOR_ATR15:
            self._close15 = np.asarray(s15.open_ms, dtype=np.int64) + tf_ms("15m")
            self._atr15 = _ffill(atr(s15.high, s15.low, s15.close, ATR_PERIOD))
            if len(s15) > MIN_BARS_FOR_STRUCT:
                self._struct15 = _structure_series(np.asarray(s15.close, float))

        # The 5m fallback the live code uses when the 15m series is too short.
        self._atr5 = (_ffill(atr(s5.high, s5.low, s5.close, ATR_PERIOD))
                      if s5 is not None and len(s5) else None)

    def _index_15m_as_of(self, bar_close_ms: int) -> int:
        """The last 15m bar that had CLOSED by `bar_close_ms`. -1 if none."""
        if self._close15 is None:
            return -1
        return int(np.searchsorted(self._close15, bar_close_ms, side="right")) - 1

    def _value(self, arr, i: int) -> float | None:
        if arr is None or i < 0 or i >= len(arr):
            return None
        v = float(arr[i])
        return v if np.isfinite(v) else None

    def bar_at(self, i5: int, signal_ms: int) -> TapeBar:
        """Build the tape row for 5m bar `i5` of the series."""
        s5 = self.s5
        open_ms = int(s5.open_ms[i5])
        bar_close = open_ms + BAR_MS
        j = self._index_15m_as_of(bar_close)

        a15 = self._value(self._atr15, j)
        if a15 is not None:
            a_val, a_tf = a15, "15m"
        else:
            a_val, a_tf = self._value(self._atr5, i5), "5m"

        return TapeBar(
            open_ms=open_ms,
            elapsed_min=(open_ms - signal_ms) / MIN_MS,
            open=float(s5.open[i5]), high=float(s5.high[i5]),
            low=float(s5.low[i5]), close=float(s5.close[i5]),
            atr=a_val, atr_tf=a_tf,
            ema_struct_15m=self._value(self._struct15, j),
            btc_regime=self._regime_at(bar_close))
