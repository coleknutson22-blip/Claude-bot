"""Simulated execution.

Accounting convention (this is the important part):

    ref_price   = the price the strategy *assumed* it would trade at
                  (signal candle close for entries, the stop level for stops)
    fill_price  = what the simulator actually gave it, after crossing the
                  spread and applying slippage

    gross_pnl      = (exit_ref  - entry_ref)  * qty
    slippage_cost  = (entry_fill - entry_ref) * qty
                   + (exit_ref  - exit_fill)  * qty
    fees           = entry_fee + exit_fee
    net_pnl        = gross_pnl - slippage_cost - fees

which is algebraically identical to

    net_pnl = (exit_fill - entry_fill) * qty - fees

so slippage is never double-counted, and the identity is asserted in the tests.
Cash movements always use *fill* prices, so the account can never drift from
the trade ledger.

No code path in this module can place a real order. There is no exchange
client here at all -- it is arithmetic over numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..models import Candle, Fill, MarketMeta, Quote
from ..timeutils import now_ms

BPS = 1e-4


def round_amount(qty: float, precision: int) -> float:
    """Always round DOWN. Rounding up could exceed available cash or the
    intended risk budget."""
    if precision < 0:
        return qty
    factor = 10 ** precision
    return math.floor(qty * factor) / factor


def round_price(price: float, precision: int) -> float:
    if precision < 0:
        return price
    factor = 10 ** precision
    return math.floor(price * factor + 0.5) / factor


@dataclass
class SizingResult:
    qty: float
    notional: float
    risk_amount: float
    stop_distance: float
    ok: bool
    reason: str = ""


class PaperBroker:
    """Deterministic simulated fills. Same inputs -> same outputs, always."""

    def __init__(self, taker_fee_bps: float, slippage_bps: float,
                 stop_slippage_bps: float, use_book_spread: bool = True,
                 max_spread_bps_entry: float = 25.0) -> None:
        self.taker_fee_bps = taker_fee_bps
        self.slippage_bps = slippage_bps
        self.stop_slippage_bps = stop_slippage_bps
        self.use_book_spread = use_book_spread
        self.max_spread_bps_entry = max_spread_bps_entry

    # ------------------------------------------------------------- fees
    def fee(self, notional: float) -> float:
        return abs(notional) * self.taker_fee_bps * BPS

    # ----------------------------------------------------------- sizing
    def size_position(self, equity: float, cash: float, entry_price: float,
                      stop_price: float, risk_pct: float, max_position_pct: float,
                      current_exposure: float, max_exposure_pct: float,
                      meta: MarketMeta, min_stop_distance_pct: float = 0.0) -> SizingResult:
        if entry_price <= 0 or not math.isfinite(entry_price):
            return SizingResult(0, 0, 0, 0, False, "invalid entry price")
        if stop_price <= 0 or stop_price >= entry_price:
            return SizingResult(0, 0, 0, 0, False, "stop must be below entry for a long")

        stop_dist = entry_price - stop_price
        stop_pct = stop_dist / entry_price * 100.0
        if stop_pct < min_stop_distance_pct:
            return SizingResult(0, 0, 0, stop_dist, False,
                                f"stop too tight ({stop_pct:.2f}% < {min_stop_distance_pct}%)")

        risk_budget = equity * risk_pct / 100.0
        qty = risk_budget / stop_dist

        # cap 1: single-position allocation
        max_notional_pos = equity * max_position_pct / 100.0
        qty = min(qty, max_notional_pos / entry_price)

        # cap 2: total portfolio exposure
        room = equity * max_exposure_pct / 100.0 - current_exposure
        if room <= 0:
            return SizingResult(0, 0, 0, stop_dist, False, "portfolio exposure limit reached")
        qty = min(qty, room / entry_price)

        # cap 3: cash actually available, including the entry fee
        affordable = cash / (entry_price * (1.0 + self.taker_fee_bps * BPS))
        qty = min(qty, affordable)

        qty = round_amount(qty, meta.amount_precision)
        if qty <= 0:
            return SizingResult(0, 0, 0, stop_dist, False, "size rounds to zero")
        if meta.min_amount and qty < meta.min_amount:
            return SizingResult(0, 0, 0, stop_dist, False,
                                f"below exchange min amount ({qty} < {meta.min_amount})")
        notional = qty * entry_price
        if meta.min_cost and notional < meta.min_cost:
            return SizingResult(0, 0, 0, stop_dist, False,
                                f"below exchange min notional (${notional:.2f} < ${meta.min_cost})")

        # actual risk after all the capping and rounding down
        actual_risk = qty * stop_dist
        return SizingResult(qty, notional, actual_risk, stop_dist, True)

    # ------------------------------------------------------------- entry
    def buy(self, symbol: str, qty: float, ref_price: float,
            quote: Quote | None = None, meta: MarketMeta | None = None,
            ts_ms: int | None = None, reason: str = "entry") -> Fill:
        """Simulated market buy. Crosses the spread if a book is available,
        then applies the configured slippage on top."""
        base = ref_price
        if quote and self.use_book_spread and quote.ask > 0:
            base = max(quote.ask, 0.0)
        fill_price = base * (1.0 + self.slippage_bps * BPS)
        if meta:
            fill_price = round_price(fill_price, meta.price_precision)
        notional = qty * fill_price
        fee = self.fee(notional)
        slip = (fill_price - ref_price) * qty
        return Fill(symbol, "buy", qty, ref_price, fill_price, fee, slip,
                    notional, ts_ms if ts_ms is not None else now_ms(), reason)

    # -------------------------------------------------------------- exit
    def sell(self, symbol: str, qty: float, ref_price: float,
             quote: Quote | None = None, meta: MarketMeta | None = None,
             ts_ms: int | None = None, reason: str = "exit",
             slippage_bps: float | None = None) -> Fill:
        base = ref_price
        if quote and self.use_book_spread and quote.bid > 0:
            base = quote.bid
        slip_bps = self.slippage_bps if slippage_bps is None else slippage_bps
        fill_price = base * (1.0 - slip_bps * BPS)
        if meta:
            fill_price = round_price(fill_price, meta.price_precision)
        notional = qty * fill_price
        fee = self.fee(notional)
        slip = (ref_price - fill_price) * qty
        return Fill(symbol, "sell", qty, ref_price, fill_price, fee, slip,
                    notional, ts_ms if ts_ms is not None else now_ms(), reason)

    def stop_exit(self, symbol: str, qty: float, stop_price: float, candle: Candle,
                  meta: MarketMeta | None = None, ts_ms: int | None = None) -> Fill | None:
        """Resolve a stop against a completed candle.

        Three cases, in order of severity:
          1. The candle opened at or below the stop -> the market GAPPED
             through it. We fill at the open, which may be far worse than the
             stop. Pretending otherwise is the classic backtest lie.
          2. The low traded through the stop -> fill at the stop, degraded by
             stop-slippage.
          3. The stop was never touched -> no fill.
        """
        if candle.low > stop_price:
            return None
        if candle.open <= stop_price:
            base = candle.open
            reason = "stop_gap"
        else:
            base = stop_price
            reason = "stop"
        fill_price = base * (1.0 - self.stop_slippage_bps * BPS)
        if meta:
            fill_price = round_price(fill_price, meta.price_precision)
        notional = qty * fill_price
        fee = self.fee(notional)
        slip = (stop_price - fill_price) * qty
        return Fill(symbol, "sell", qty, stop_price, fill_price, fee, slip,
                    notional, ts_ms if ts_ms is not None else candle.open_ms, reason)

    # ------------------------------------------------------------ checks
    def spread_acceptable(self, quote: Quote | None) -> tuple[bool, float]:
        if quote is None:
            return True, 0.0
        s = quote.spread_bps
        return (s <= self.max_spread_bps_entry), s


def realise_pnl(entry_ref: float, entry_fill: float, exit_ref: float,
                exit_fill: float, qty: float, entry_fee: float,
                exit_fee: float) -> dict:
    """Single source of truth for trade P&L decomposition."""
    gross = (exit_ref - entry_ref) * qty
    slippage = (entry_fill - entry_ref) * qty + (exit_ref - exit_fill) * qty
    fees = entry_fee + exit_fee
    net = gross - slippage - fees
    return {"gross_pnl": gross, "slippage_cost": slippage, "fees": fees, "net_pnl": net}
