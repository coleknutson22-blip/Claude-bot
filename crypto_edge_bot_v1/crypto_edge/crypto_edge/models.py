"""Core domain objects. Deliberately plain dataclasses so they serialise cleanly
into SQLite and are trivial to construct in tests."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Candle:
    open_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def is_valid(self) -> bool:
        vals = (self.open, self.high, self.low, self.close)
        if any((v is None or not np.isfinite(v) or v <= 0) for v in vals):
            return False
        if self.volume is None or not np.isfinite(self.volume) or self.volume < 0:
            return False
        return self.high >= max(self.open, self.close) and self.low <= min(self.open, self.close)


@dataclass
class Series:
    """An OHLCV series for one symbol/timeframe, held as numpy arrays."""
    symbol: str
    timeframe: str
    open_ms: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __len__(self) -> int:
        return len(self.close)

    @classmethod
    def from_ohlcv(cls, symbol: str, timeframe: str, rows: list[list[float]]) -> "Series":
        """`rows` is CCXT's format: [open_ms, open, high, low, close, volume]."""
        if not rows:
            empty = np.array([], dtype=float)
            return cls(symbol, timeframe, np.array([], dtype=np.int64),
                       empty, empty, empty, empty, empty)
        arr = np.asarray(rows, dtype=float)
        return cls(
            symbol=symbol, timeframe=timeframe,
            open_ms=arr[:, 0].astype(np.int64),
            open=arr[:, 1], high=arr[:, 2], low=arr[:, 3],
            close=arr[:, 4], volume=arr[:, 5],
        )

    def last_candle(self) -> Candle:
        i = len(self) - 1
        return Candle(int(self.open_ms[i]), float(self.open[i]), float(self.high[i]),
                      float(self.low[i]), float(self.close[i]), float(self.volume[i]))

    def drop_unclosed(self, now_ms: int, buffer_ms: int = 0) -> "Series":
        """Remove any trailing candle that has not finished. This is the single
        most important guard against look-ahead bias in live operation."""
        from .timeutils import candle_close_ms
        keep = len(self)
        while keep > 0 and now_ms < candle_close_ms(int(self.open_ms[keep - 1]),
                                                    self.timeframe) + buffer_ms:
            keep -= 1
        if keep == len(self):
            return self
        return Series(self.symbol, self.timeframe, self.open_ms[:keep], self.open[:keep],
                      self.high[:keep], self.low[:keep], self.close[:keep], self.volume[:keep])

    def is_sane(self) -> bool:
        if len(self) == 0:
            return False
        for a in (self.open, self.high, self.low, self.close):
            if not np.all(np.isfinite(a)) or np.any(a <= 0):
                return False
        if not np.all(np.isfinite(self.volume)) or np.any(self.volume < 0):
            return False
        if np.any(self.high < np.maximum(self.open, self.close) - 1e-12):
            return False
        if np.any(self.low > np.minimum(self.open, self.close) + 1e-12):
            return False
        # timestamps must be strictly increasing and evenly spaced
        if len(self) > 1:
            d = np.diff(self.open_ms)
            if np.any(d <= 0) or len(np.unique(d)) != 1:
                return False
        return True


@dataclass
class MarketMeta:
    """Exchange metadata for one market. Without this we cannot size an order
    correctly, so its absence is a hard block on trading that symbol.

    Granularity is carried two ways because exchanges express it two ways:

      * `amount_precision` / `price_precision` -- decimal places, which is what
        CCXT's DECIMAL_PLACES mode reports
      * `amount_step` / `price_step` -- an absolute tick size, which is what
        CCXT's TICK_SIZE mode reports and what every major venue actually uses

    A step is strictly more expressive: 0.05 and 0.5 are legal ticks and cannot
    be written as a number of decimal places. When a step is present it wins;
    the decimal fields remain as a fallback for venues that only give those.
    `0.0` means "not specified".
    """
    symbol: str
    base: str
    quote: str
    active: bool = True
    amount_precision: int = 8      # decimal places for quantity
    price_precision: int = 8
    min_amount: float = 0.0
    min_cost: float = 0.0          # minimum notional in quote currency
    amount_step: float = 0.0       # absolute tick for quantity (0 = unspecified)
    price_step: float = 0.0        # absolute tick for price   (0 = unspecified)


# Where a quote's timestamp came from. This matters: if the venue did not
# supply one and we substituted our own receive time, then "quote age" measures
# our own clock, not the venue's freshness -- an age check against a local
# stamp is close to vacuous and must not be mistaken for a real staleness test.
TS_VENUE = "venue"     # the exchange stamped it; age is meaningful
TS_LOCAL = "local"     # we stamped it on receipt; age is ~0 by construction


@dataclass
class Quote:
    symbol: str
    bid: float
    ask: float
    last: float
    ts_ms: int
    ts_source: str = TS_VENUE

    @property
    def venue_timestamped(self) -> bool:
        return self.ts_source == TS_VENUE

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last

    @property
    def spread_bps(self) -> float:
        if self.bid > 0 and self.ask > self.bid:
            return (self.ask - self.bid) / self.mid * 10_000.0
        return 0.0


@dataclass
class Fill:
    """Result of a simulated order. `ref_price` is the price the strategy
    believed it would get; `fill_price` is what the paper engine actually gave
    it. The difference is slippage, tracked explicitly rather than hidden."""
    symbol: str
    side: str            # "buy" | "sell"
    qty: float
    ref_price: float
    fill_price: float
    fee: float
    slippage_cost: float
    notional: float
    ts_ms: int
    reason: str = ""


@dataclass
class Position:
    id: str
    symbol: str
    strategy: str
    strategy_version: str
    side: str
    qty: float
    entry_ref_price: float
    entry_fill_price: float
    entry_ms: int
    entry_fee: float
    entry_slippage: float
    initial_stop: float
    current_stop: float
    highest_price: float
    lowest_price: float
    risk_amount: float
    candle_id: str
    signal_score: float = 0.0
    mfe: float = 0.0             # max favourable excursion, price terms
    mae: float = 0.0             # max adverse excursion, price terms
    journal: dict[str, Any] = field(default_factory=dict)

    @property
    def entry_notional(self) -> float:
        return self.qty * self.entry_fill_price

    def mark_value(self, price: float) -> float:
        return self.qty * price

    def unrealized(self, price: float) -> float:
        return (price - self.entry_fill_price) * self.qty

    def to_row(self) -> dict:
        d = asdict(self)
        d["journal"] = json.dumps(d["journal"], default=str)
        return d


@dataclass
class ClosedTrade:
    id: str
    position_id: str
    symbol: str
    strategy: str
    strategy_version: str
    qty: float
    entry_ref_price: float
    entry_fill_price: float
    entry_ms: int
    exit_ref_price: float
    exit_fill_price: float
    exit_ms: int
    exit_reason: str
    initial_stop: float
    final_stop: float
    gross_pnl: float
    fees: float
    slippage_cost: float
    net_pnl: float
    return_pct: float
    account_return_pct: float
    mfe: float
    mae: float
    duration_s: float
    equity_after: float
    journal: dict[str, Any] = field(default_factory=dict)
