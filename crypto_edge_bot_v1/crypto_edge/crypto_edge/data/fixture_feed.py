"""Deterministic in-memory feed.

Used by the automated tests and by `scripts/smoke_test.py`, so the whole
engine -- signal generation, sizing, fills, stops, persistence, restart
recovery, notifications -- can be verified end to end with zero network.
"""
from __future__ import annotations

import numpy as np

from ..models import MarketMeta, Quote, Series
from ..timeutils import now_ms, tf_ms


class FixtureFeed:
    name = "fixture"

    def __init__(self, series: dict[tuple[str, str], Series],
                 markets: dict[str, MarketMeta] | None = None,
                 tickers: dict[str, dict] | None = None,
                 clock_ms: int | None = None,
                 spread_bps: float = 4.0) -> None:
        self._series = series
        self._markets = markets or {}
        self._tickers = tickers or {}
        self.clock_ms = clock_ms
        self.spread_bps = spread_bps
        self.calls: list[tuple] = []

    def load_markets(self) -> dict[str, MarketMeta]:
        if self._markets:
            return self._markets
        out = {}
        for (sym, _tf) in self._series:
            base = sym.split("/")[0]
            out[sym] = MarketMeta(sym, base, "USDT", True, 6, 4, 0.0, 10.0)
        self._markets = out
        return out

    def fetch_tickers(self) -> dict[str, dict]:
        if self._tickers:
            return self._tickers
        out = {}
        for (sym, tf), s in self._series.items():
            if len(s) == 0:
                continue
            last = float(s.close[-1])
            out[sym] = {"symbol": sym, "last": last,
                        "quoteVolume": 50_000_000.0,
                        "bid": last * (1 - self.spread_bps / 2e4),
                        "ask": last * (1 + self.spread_bps / 2e4)}
        return out

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> Series:
        from .feed import DataUnavailable
        s = self._series.get((symbol, timeframe))
        if s is None:
            # A real venue serves every timeframe it advertises, so a fixture
            # that only holds 1h would misrepresent one -- and the market-age
            # probe legitimately asks at a coarser resolution. Derive it by
            # resampling the finest series we do have.
            s = self._derive(symbol, timeframe)
        if s is None:
            raise DataUnavailable(f"no fixture series for {symbol} {timeframe}")
        if self.clock_ms is None:
            return s
        # only reveal candles whose OPEN time has passed -- the fixture feed
        # mimics an exchange returning a partially-formed final candle
        keep = int(np.searchsorted(s.open_ms, self.clock_ms, side="right"))
        keep = max(0, min(keep, len(s)))
        start = max(0, keep - limit)
        return Series(symbol, timeframe, s.open_ms[start:keep], s.open[start:keep],
                      s.high[start:keep], s.low[start:keep], s.close[start:keep],
                      s.volume[start:keep])

    def _derive(self, symbol: str, timeframe: str) -> Series | None:
        """Resample a coarser series from the finest one held for `symbol`."""
        target = tf_ms(timeframe)
        candidates = [(tf, ser) for (sym, tf), ser in self._series.items()
                      if sym == symbol and len(ser) and tf_ms(tf) < target]
        if not candidates:
            return None
        src_tf, src = min(candidates, key=lambda kv: tf_ms(kv[0]))
        step = tf_ms(src_tf)
        per = max(1, target // step)
        rows = []
        for i in range(0, len(src) - per + 1, per):
            chunk = slice(i, i + per)
            bucket = int(src.open_ms[i]) // target * target
            rows.append([bucket, float(src.open[i]),
                         float(np.max(src.high[chunk])),
                         float(np.min(src.low[chunk])),
                         float(src.close[i + per - 1]),
                         float(np.sum(src.volume[chunk]))])
        if not rows:
            return None
        derived = Series.from_ohlcv(symbol, timeframe, rows)
        self._series[(symbol, timeframe)] = derived
        return derived

    def fetch_quote(self, symbol: str) -> Quote | None:
        for tf in ("1h", "4h", "1d"):
            s = self._series.get((symbol, tf))
            if s is not None and len(s):
                last = float(s.close[-1])
                if self.clock_ms is not None:
                    sub = self.fetch_ohlcv(symbol, tf, 5)
                    if len(sub) == 0:
                        return None
                    last = float(sub.close[-1])
                half = self.spread_bps / 2e4
                # A real ticker is stamped when the exchange published it, i.e.
                # ~now -- NOT with the open time of the last candle, which would
                # look an entire timeframe stale to the entry quote gate.
                # `clock_ms` is the simulated now when a test pins the clock.
                ts = self.clock_ms if self.clock_ms is not None else now_ms()
                return Quote(symbol, last * (1 - half), last * (1 + half), last, ts)
        return None

    def server_time_ms(self) -> int | None:
        return self.clock_ms


def make_series(symbol: str, timeframe: str, closes, start_ms: int = 1_700_000_000_000,
                volume: float | list = 1000.0, wick_pct: float = 0.004) -> Series:
    """Build a clean synthetic OHLCV series from a list of closes."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    step = tf_ms(timeframe)
    open_ms = np.array([start_ms + i * step for i in range(n)], dtype=np.int64)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) * (1 + wick_pct)
    lows = np.minimum(opens, closes) * (1 - wick_pct)
    if isinstance(volume, (int, float)):
        vols = np.full(n, float(volume))
    else:
        vols = np.asarray(volume, dtype=float)
    return Series(symbol, timeframe, open_ms, opens, highs, lows, closes, vols)
