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
    entry_price: float = 0.0     # the price the size was actually computed on
    risk_budget: float = 0.0     # equity * risk_pct, before capping/rounding
    entry_fee: float = 0.0       # modelled cost of getting in
    est_cost_at_stop: float = 0.0   # loss at the stop INCLUDING both fees


@dataclass
class QuoteCheck:
    """Result of validating a live quote for a NEW ENTRY. Fails closed."""
    ok: bool
    reason: str = ""
    spread_bps: float = 0.0
    age_s: float = 0.0


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

    # ------------------------------------------------------ expected fill
    def expected_entry_price(self, ref_price: float, quote: Quote | None = None,
                             meta: MarketMeta | None = None) -> float:
        """The price `buy()` WILL fill at, computed before we commit to a size.

        This mirrors `buy()` exactly -- same spread crossing, same slippage,
        same price rounding -- because sizing against anything else is how a
        position ends up risking more than the configured budget. `buy()` is
        deterministic, so this is a prediction only in name.
        """
        base = ref_price
        if quote and self.use_book_spread and quote.ask > 0:
            base = max(quote.ask, 0.0)
        fill_price = base * (1.0 + self.slippage_bps * BPS)
        if meta:
            fill_price = round_price(fill_price, meta.price_precision)
        return fill_price

    # ----------------------------------------------------------- sizing
    def size_position(self, equity: float, cash: float, entry_price: float,
                      stop_price: float, risk_pct: float, max_position_pct: float,
                      current_exposure: float, max_exposure_pct: float,
                      meta: MarketMeta, min_stop_distance_pct: float = 0.0) -> SizingResult:
        """Size a long so that a stop-out costs at most `equity * risk_pct`.

        `entry_price` MUST be the price the order is expected to actually fill
        at (see `expected_entry_price`), not the signal candle's close. Sizing
        on the reference price while filling higher silently inflates risk by
        the full spread-plus-slippage gap.
        """
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
        entry_fee = self.fee(notional)
        # What a stop-out really costs: price risk, plus the fee paid getting
        # in, plus the fee and slippage paid getting out. Reported so the
        # journal shows the true worst case rather than the price-only figure.
        exit_notional = qty * stop_price * (1.0 - self.stop_slippage_bps * BPS)
        est_cost_at_stop = (actual_risk + entry_fee + self.fee(exit_notional)
                            + qty * stop_price * self.stop_slippage_bps * BPS)
        return SizingResult(qty, notional, actual_risk, stop_dist, True,
                            entry_price=entry_price, risk_budget=risk_budget,
                            entry_fee=entry_fee, est_cost_at_stop=est_cost_at_stop)

    # ------------------------------------------------ post-sizing revalidation
    def revalidate_risk(self, qty: float, fill_price: float, stop_price: float,
                        equity: float, risk_pct: float,
                        tolerance_pct: float = 1.0) -> tuple[bool, str, float]:
        """Last gate before a position is committed.

        Recomputes risk from the price the trade ACTUALLY filled at. Anything
        that slipped past sizing -- a quote that moved, a rounding artefact, a
        caller that sized on the wrong price -- is caught here rather than
        discovered later in the trade ledger.
        """
        if qty <= 0 or not math.isfinite(qty):
            return False, "non-positive quantity", 0.0
        if fill_price <= 0 or not math.isfinite(fill_price):
            return False, "invalid fill price", 0.0
        if stop_price >= fill_price:
            return False, (f"stop ${stop_price:,.6g} is not below the simulated "
                           f"fill ${fill_price:,.6g}"), 0.0
        actual_risk = qty * (fill_price - stop_price)
        budget = equity * risk_pct / 100.0
        allowed = budget * (1.0 + tolerance_pct / 100.0)
        if actual_risk > allowed:
            return False, (f"risk at simulated fill ${actual_risk:,.2f} exceeds "
                           f"budget ${budget:,.2f} "
                           f"({actual_risk / budget * 100.0 - 100.0:+.1f}%)"), actual_risk
        return True, "", actual_risk

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
        """Spread gate. A missing quote is NOT an acceptable spread -- there is
        no book to judge, so this fails closed. Entry callers should use
        `validate_entry_quote`, which subsumes this and every other check."""
        if quote is None:
            return False, float("inf")
        s = quote.spread_bps
        return (s <= self.max_spread_bps_entry), s

    @staticmethod
    def quote_structure_error(quote: Quote | None) -> str:
        """Is this object a usable quote at all? Returns "" if it is.

        Shared by the entry gate and the exit path. Entries additionally demand
        freshness, a tight spread and plausibility (see `validate_entry_quote`);
        exits only need to know the numbers are not nonsense, because an exit
        falls back to the reference price rather than being blocked.
        """
        if quote is None:
            return "quote unavailable"
        for attr in ("bid", "ask", "last", "ts_ms"):
            if not hasattr(quote, attr):
                return f"quote malformed (no {attr})"
        try:
            bid, ask = float(quote.bid), float(quote.ask)
            last, ts = float(quote.last), int(quote.ts_ms)
        except (TypeError, ValueError):
            return "quote malformed (non-numeric fields)"
        if not all(math.isfinite(v) for v in (bid, ask, last)):
            return "quote invalid (non-finite price)"
        if bid <= 0 or ask <= 0 or last <= 0:
            return "quote invalid (non-positive price)"
        if ask < bid:
            return f"quote invalid (crossed book: bid {bid:g} > ask {ask:g})"
        if ts <= 0:
            return "quote invalid (no timestamp)"
        return ""

    def validate_entry_quote(self, quote: Quote | None, *, ref_price: float = 0.0,
                             now: int | None = None, max_age_s: float = 90.0,
                             max_future_skew_s: float = 30.0,
                             max_deviation_pct: float = 10.0) -> QuoteCheck:
        """Gate a NEW ENTRY on a live quote. Fails closed on every doubt.

        A new position is discretionary: if we cannot see a book we trust, the
        correct answer is simply not to trade. (Exits are the opposite case and
        deliberately do NOT go through here -- an open position must remain
        manageable whether or not a quote is available.)

        Rejects: unavailable, malformed, invalid, crossed, stale, future-stamped,
        wide-spread, and quotes implausibly far from the signal reference.
        """
        now = now_ms() if now is None else now

        structural = self.quote_structure_error(quote)
        if structural:
            return QuoteCheck(False, structural)

        ts = int(quote.ts_ms)
        age_s = (now - ts) / 1000.0
        if age_s > max_age_s:
            return QuoteCheck(False, f"quote stale by {age_s:.0f}s (limit {max_age_s:.0f}s)",
                              age_s=age_s)
        if age_s < -max_future_skew_s:
            return QuoteCheck(False,
                              f"quote timestamped {-age_s:.0f}s in the future",
                              age_s=age_s)

        spread = quote.spread_bps
        if not math.isfinite(spread):
            return QuoteCheck(False, "quote invalid (unmeasurable spread)", age_s=age_s)
        if spread > self.max_spread_bps_entry:
            return QuoteCheck(False,
                              f"spread {spread:.1f}bps exceeds entry limit "
                              f"({self.max_spread_bps_entry:.1f}bps)",
                              spread_bps=spread, age_s=age_s)

        if ref_price > 0 and math.isfinite(ref_price) and max_deviation_pct > 0:
            dev = abs(quote.mid - ref_price) / ref_price * 100.0
            if dev > max_deviation_pct:
                return QuoteCheck(False,
                                  f"quote deviates {dev:.1f}% from the signal "
                                  f"reference (limit {max_deviation_pct:.1f}%)",
                                  spread_bps=spread, age_s=age_s)

        return QuoteCheck(True, "", spread_bps=spread, age_s=age_s)


def realise_pnl(entry_ref: float, entry_fill: float, exit_ref: float,
                exit_fill: float, qty: float, entry_fee: float,
                exit_fee: float) -> dict:
    """Single source of truth for trade P&L decomposition."""
    gross = (exit_ref - entry_ref) * qty
    slippage = (entry_fill - entry_ref) * qty + (exit_ref - exit_fill) * qty
    fees = entry_fee + exit_fee
    net = gross - slippage - fees
    return {"gross_pnl": gross, "slippage_cost": slippage, "fees": fees, "net_pnl": net}
