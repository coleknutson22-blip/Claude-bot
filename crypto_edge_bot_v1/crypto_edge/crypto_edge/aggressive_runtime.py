"""Strategy B's trading loop: size with the ladder, manage with deterministic exits.

WHY THIS IS A SEPARATE CLASS AND NOT MORE ENGINE
-----------------------------------------------
`TradingEngine` is Strategy A's loop, and Strategy A is running with real
recorded history that must not move. Threading a second strategy's ladder,
short handling and exit set through the same 900 lines would put every one of
those changes on Strategy A's execution path, where the only thing standing
between a mistake and a corrupted ledger is that the equivalence tests happen
to cover it.

So Strategy B owns its own account, its own entries and its own exits, and
BORROWS the engine's data: the same feed, the same candle cache, the same
universe, the same market context. One fetch, two strategies. The engine calls
`manage()` and `scan_and_enter()` once per cycle and otherwise knows nothing
about it.

ORDER WITHIN A CYCLE, AND WHY
-----------------------------
Exits run BEFORE entries, always. A position closed this cycle frees both cash
and a ladder slot, and an entry sized before that close would be sized against
a balance that is about to change. It also means a forced short close cannot be
starved of attention by a scan that takes longer than expected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .indicators import atr, last_valid
from .logging_setup import log_event
from .models import Series
from .notify import formatters as fmt
from .portfolio import aggressive_exits as ex
from .portfolio import ladder as L
from .portfolio.account import PaperAccount
from .portfolio.risk import RiskManager
from .research.journal import ENTERED, REJECTED_RISK, REJECTED_STRATEGY
from .scan import scan
from .strategy import confidence as conf
from .strategy import features as feat
from .strategy.base import LONG, SHORT, MarketContext
from .timeutils import now_ms, tf_ms


@dataclass
class AggressiveStatus:
    entries: int = 0
    exits: int = 0
    evaluated: int = 0
    open_positions: int = 0
    last_scan_ms: int = 0
    halted: bool = False
    halt_reason: str = ""
    deep_fetches: int = 0
    binding: dict = field(default_factory=dict)


class AggressiveRuntime:
    def __init__(self, cfg: Config, repo, feed, notifier, broker,
                 journal) -> None:
        self.cfg = cfg
        self.repo = repo
        self.feed = feed
        self.notifier = notifier
        self.broker = broker
        self.journal = journal
        a = cfg.aggressive
        self.name = a.name
        self.account = PaperAccount(
            repo, broker, cfg.starting_equity_for(a.name), a.name,
            borrow_bps_per_day=a.short_borrow_bps_per_day)
        self.risk = RiskManager(cfg.risk)
        self.status = AggressiveStatus()

    # ------------------------------------------------------------- helpers
    def marks(self, series_5m: dict[str, Series]) -> dict[str, float]:
        out = {}
        for p in self.account.positions():
            s = series_5m.get(p.symbol)
            if s is not None and len(s):
                out[p.symbol] = float(s.close[-1])
        return out

    def equity(self, series_5m: dict[str, Series]) -> float:
        return self.account.equity(self.marks(series_5m))

    def daily_buffer(self, equity: float) -> float:
        """Loss still available before this strategy's own daily halt trips."""
        acct = self.account.state
        return self.risk.daily_loss_remaining(
            equity, float(acct["daily_start_equity"]))

    # ============================================================== exits
    def manage(self, ctx: MarketContext, frames_by_symbol: dict[str, dict],
               markets: dict) -> None:
        """Stops first, then the deterministic exit set, then trail.

        A stop is resolved against the candle's extremes rather than its close,
        because a stop fills where it was touched -- resolving it against a
        close that has already run past it would book a fill that never
        existed.
        """
        a = self.cfg.aggressive
        for pos in self.account.positions():
            frames = frames_by_symbol.get(pos.symbol) or {}
            s = frames.get("5m") or frames.get("15m")
            if s is None or len(s) == 0 or not s.is_sane():
                log_event("strategy", "WARNING",
                          "no usable data for open Strategy B position",
                          symbol=pos.symbol, strategy=self.name)
                continue
            meta = markets.get(pos.symbol)
            candle = s.last_candle()
            bar_end = int(s.open_ms[-1]) + tf_ms(s.timeframe)

            # --- 1. protective stop, direction-aware ----------------------
            fill = self.broker.stop_exit(pos.symbol, pos.qty, pos.current_stop,
                                         candle, meta, ts_ms=bar_end,
                                         direction=pos.direction)
            if fill is not None:
                self._close(pos, fill,
                            ex.STOP if fill.reason == "stop" else "stop_gap",
                            frames_by_symbol)
                continue

            price = float(s.close[-1])
            s15 = frames.get("15m")
            struct = (feat.ema_structure(s15.close)
                      if s15 is not None and len(s15) > 50 else None)
            if struct is not None and not np.isfinite(struct):
                struct = None

            # --- 2. the deterministic exit set ----------------------------
            reason = ex.check_exit(pos, price, now_ms=now_ms(), cfg=a,
                                   ema_struct=struct, btc_regime=ctx.btc_regime)
            if reason:
                quote = self._safe_quote(pos.symbol)
                if quote is not None and self.broker.quote_structure_error(quote):
                    # An exit is never blocked for want of a quote, but a
                    # broken one must not price the fill either.
                    quote = None
                exit_fill = self.broker.exit_fill(
                    pos.symbol, pos.qty, price, pos.direction, quote, meta,
                    ts_ms=now_ms(), reason=reason)
                self._close(pos, exit_fill, reason, frames_by_symbol)
                continue

            # --- 3. ratchet the stop --------------------------------------
            src = s15 if s15 is not None and len(s15) > 20 else s
            a_val = last_valid(atr(src.high, src.low, src.close, 14))
            upd = ex.update_stop(
                pos, price, a_val if np.isfinite(a_val) else 0.0,
                breakeven_at_r=a.breakeven_at_r,
                breakeven_offset_r=a.breakeven_offset_r,
                trail_start_r=a.trail_start_r, trail_atr_mult=a.trail_atr_mult)
            if upd.changed:
                prev = pos.current_stop
                self.account.set_stop(pos, upd.new_stop)
                log_event("trades", "INFO", "stop updated", symbol=pos.symbol,
                          strategy=self.name, previous=prev, new=upd.new_stop,
                          kind=upd.kind, side=pos.side)
        self.account.update_marks(self.marks(
            {k: (v.get("5m") or v.get("15m")) for k, v in frames_by_symbol.items()
             if v}))
        self.status.open_positions = len(self.account.positions())

    def _close(self, pos, fill, reason: str, frames_by_symbol: dict) -> None:
        marks = self.marks({k: (v.get("5m") or v.get("15m"))
                            for k, v in frames_by_symbol.items() if v})
        trade = self.account.close_position(pos, fill, reason, marks)
        if trade is None:
            return
        self.status.exits += 1
        equity = self.account.equity(marks)
        self.notifier.send(
            fmt.exit_trade(
                symbol=trade.symbol, entry_price=trade.entry_fill_price,
                exit_price=trade.exit_fill_price,
                position_value=trade.qty * trade.entry_fill_price,
                held_s=trade.duration_s, gross=trade.gross_pnl, fees=trade.fees,
                slippage=trade.slippage_cost, net=trade.net_pnl,
                return_pct=trade.return_pct,
                account_return_pct=trade.account_return_pct,
                exit_reason=f"{trade.exit_reason} ({trade.side})", equity=equity,
                total_pnl=float(self.account.state["realized_pnl"])),
            dedupe_key=f"exitB:{trade.id}", kind="exit")

    def _safe_quote(self, symbol: str):
        try:
            return self.feed.fetch_quote(symbol)
        except Exception as e:
            log_event("data", "WARNING", "quote fetch failed",
                      symbol=symbol, error=str(e))
            return None

    # ============================================================= entries
    def scan_and_enter(self, ctx: MarketContext, *, rank_series: dict,
                       btc_1h, meta_by_symbol: dict, markets: dict,
                       entries_allowed: bool = True) -> int:
        a = self.cfg.aggressive
        res = scan(self.cfg, self.feed, ctx, rank_series=rank_series,
                   btc_1h=btc_1h, meta_by_symbol=meta_by_symbol,
                   now_ms=now_ms(),
                   buffer_ms=self.cfg.safety.candle_close_buffer_s * 1000)
        self.status.deep_fetches = res.deep_fetches
        self.status.evaluated = len(res.signals)
        self.status.last_scan_ms = now_ms()

        series_5m = {}
        for sig in res.signals:
            self.journal.record(
                sig, ENTERED if sig.passed else REJECTED_STRATEGY,
                rank=sig.features.get("rank"))

        if not entries_allowed or self.status.halted:
            return 0

        taken = 0
        for sig in res.entries:
            if taken >= self.cfg.risk.max_new_entries_per_cycle:
                break
            if self._enter(sig, ctx, markets, series_5m):
                taken += 1
        return taken

    def _enter(self, sig, ctx: MarketContext, markets: dict,
               series_5m: dict) -> bool:
        c, a = self.cfg, self.cfg.aggressive
        sym = sig.symbol
        rank = sig.features.get("rank")

        def refuse(reason: str) -> bool:
            self.risk.log_rejection(sym, reason, rank=rank)
            self.journal.record(sig, REJECTED_RISK, reason, rank=rank)
            return False

        meta = markets.get(sym)
        if meta is None:
            return refuse("exchange metadata unavailable")
        if self.repo.get_position(self.name, sym) is not None:
            return refuse("position already open in symbol")

        open_positions = self.account.positions()
        n_open = len(open_positions)
        if n_open >= a.max_open_positions:
            return refuse(f"all {a.max_open_positions} position slots are in use")

        marks = self.marks(series_5m)
        equity = self.account.equity(marks)
        exposure = self.account.exposure(marks)

        # ---- confidence ---------------------------------------------------
        confidence = conf.to_confidence(sig.score, a.confidence_is_identity)
        bucket, mult = conf.bucket_for(confidence, a.confidence_buckets)
        if not conf.is_tradable(confidence, a.min_confidence, a.confidence_buckets):
            return refuse(f"confidence {confidence:.1f} below the "
                          f"{a.min_confidence:.0f} floor (bucket {bucket})")

        # ---- the quote gate: NEW ENTRIES FAIL CLOSED ----------------------
        quote = self._safe_quote(sym)
        qc = self.broker.validate_entry_quote(
            quote, ref_price=sig.ref_price,
            max_age_s=c.execution.max_quote_age_s,
            max_future_skew_s=c.execution.max_quote_future_skew_s,
            max_deviation_pct=c.execution.max_quote_deviation_pct,
            require_venue_timestamp=c.execution.require_venue_quote_timestamp)
        if not qc.ok:
            return refuse(f"entry blocked: {qc.reason}")

        d = sig.direction
        est_fill = self.broker.expected_entry_price(sig.ref_price, quote, meta)
        if d < 0 and quote is not None and quote.bid > 0:
            # A short opens by SELLING, so it crosses to the bid, not the ask.
            est_fill = self.broker.exit_fill(
                sym, 1.0, sig.ref_price, 1, quote, meta).fill_price

        # ---- the ladder proposes, risk disposes ---------------------------
        plan = L.plan(
            free_cash=self.account.cash(), equity=equity, n_open=n_open,
            confidence=confidence, multiplier=mult, entry_price=est_fill,
            stop_price=sig.stop_price, direction=d, exposure=exposure,
            daily_buffer_remaining=self.daily_buffer(equity),
            ceilings_pct=a.ladder_ceilings_pct, max_loss_pct=a.max_loss_pct,
            daily_buffer_fraction=a.daily_buffer_fraction, leverage=a.leverage,
            max_exposure_pct=a.max_exposure_pct,
            min_notional=meta.min_cost or 0.0,
            max_positions=a.max_open_positions)
        self.status.binding[plan.binding_constraint] = (
            self.status.binding.get(plan.binding_constraint, 0) + 1)
        if not plan.tradable:
            return refuse(plan.reason or "ladder allows nothing")

        from .execution.paper_broker import round_amount
        qty = round_amount(plan.notional / est_fill, meta.amount_precision,
                           meta.amount_step)
        if qty <= 0:
            return refuse("size rounds to zero")
        if meta.min_amount and qty < meta.min_amount:
            return refuse(f"below exchange min amount ({qty} < {meta.min_amount})")

        fill = self.broker.entry_fill(sym, qty, sig.ref_price, d, quote, meta,
                                      ts_ms=now_ms())

        # ---- revalidate against the SIMULATED FILL ------------------------
        stop_pct = abs(fill.fill_price - sig.stop_price) / fill.fill_price
        loss_at_stop = fill.notional * stop_pct
        if loss_at_stop > plan.max_loss_cash * (
                1.0 + c.execution.risk_overshoot_tolerance_pct / 100.0):
            return refuse(
                f"loss at the simulated fill ${loss_at_stop:,.2f} exceeds the "
                f"cap ${plan.max_loss_cash:,.2f}")
        ok_risk, risk_reason, _ = self.broker.revalidate_risk(
            qty=qty, fill_price=fill.fill_price, stop_price=sig.stop_price,
            equity=equity,
            risk_pct=loss_at_stop / equity * 100.0 if equity else 0.0,
            tolerance_pct=c.execution.risk_overshoot_tolerance_pct,
            direction=d)
        if not ok_risk:
            return refuse(risk_reason)

        journal = self._entry_journal(sig, plan, bucket, confidence, qty, fill,
                                      stop_pct, equity, qc, ctx)
        pos = self.account.open_position(
            symbol=sym, strategy=self.name, strategy_version=sig.version,
            qty=qty, ref_price=sig.ref_price, fill=fill,
            initial_stop=sig.stop_price, risk_amount=loss_at_stop,
            candle_id=sig.candle_id, signal_score=sig.score, journal=journal,
            side=sig.side, margin_held=fill.notional / max(1.0, a.leverage))
        if pos is None:
            return refuse("duplicate or insufficient cash")

        self.status.entries += 1
        self.journal.record(sig, ENTERED, "", rank=rank, extra=journal)
        self.notifier.send(
            fmt.entry(
                symbol=sym, side=sig.side, entry_price=fill.fill_price, qty=qty,
                position_value=fill.notional,
                pct_of_account=fill.notional / equity * 100.0 if equity else 0.0,
                stop=sig.stop_price, dollar_risk=loss_at_stop,
                pct_risk=loss_at_stop / equity * 100.0 if equity else 0.0,
                score=sig.score, htf_regime=f"slot {plan.slot} x{mult:.2f}",
                btc_regime=ctx.btc_regime, breadth=ctx.breadth_pct,
                reasons=[f"setup {sig.score:.1f} -> confidence {confidence:.1f} "
                         f"(bucket {bucket})",
                         f"ladder slot {plan.slot}: ceiling "
                         f"${plan.ceiling_cash:,.0f} x {mult:.2f}",
                         f"limited by {plan.binding_constraint}"],
                equity=self.account.equity(self.marks(series_5m))),
            dedupe_key=f"entryB:{sig.candle_id}", kind="entry")
        return True

    def _entry_journal(self, sig, plan, bucket, confidence, qty, fill,
                       stop_pct, equity, qc, ctx) -> dict:
        """Everything Stage 3 asked to be recorded, on every entry."""
        f = sig.features
        return {
            "symbol": sig.symbol, "strategy": self.name,
            "strategy_version": sig.version, "candle_id": sig.candle_id,
            "side": sig.side, "rank": f.get("rank"),
            "setup_score": sig.score,
            "confidence": confidence,
            "conf_bucket": bucket,
            "conf_multiplier": plan.multiplier,
            "ladder_slot": plan.slot,
            "ladder_ceiling_cash": plan.ceiling_cash,
            "target_notional": plan.target_notional,
            "final_notional": fill.notional,
            "binding_constraint": plan.binding_constraint,
            "limits": plan.detail,
            "max_loss_cash": plan.max_loss_cash,
            "stop_price": sig.stop_price,
            "stop_distance_pct": stop_pct * 100.0,
            "expected_loss_cash": fill.notional * stop_pct,
            "expected_loss_pct": (fill.notional * stop_pct / equity * 100.0
                                  if equity else 0.0),
            "leverage": self.cfg.aggressive.leverage,
            "borrow_bps_per_day": (self.cfg.aggressive.short_borrow_bps_per_day
                                   if sig.side == SHORT else 0.0),
            "qty": qty, "entry_fill_price": fill.fill_price,
            "entry_ref_price": sig.ref_price,
            "account_equity": equity,
            "free_cash_before": self.account.cash(),
            "spread_bps": qc.spread_bps, "quote_age_s": qc.age_s,
            "quote_ts_source": qc.ts_source,
            "btc_regime": ctx.btc_regime, "breadth_pct": ctx.breadth_pct,
            "score_components": sig.components,
            "momentum": f.get("momentum"), "momentum_atr": f.get("momentum_atr"),
            "atr_pct": f.get("atr_pct"), "atr_expansion": f.get("atr_expansion"),
            "rel_volume": f.get("rel_volume"),
            "rel_strength": f.get("rel_strength"),
        }

    # ============================================================== safety
    def check_safety(self, series_5m: dict) -> bool:
        """This strategy's own circuit breakers. Halting B never halts A."""
        acct = self.account.state
        equity = self.equity(series_5m)
        cs = self.risk.check_circuit_breakers(
            equity, float(acct["peak_equity"]),
            float(acct["daily_start_equity"]))
        if cs.halted and not self.status.halted:
            self.status.halted, self.status.halt_reason = True, cs.reason
            self.repo.set_halt(self.name, True, cs.reason)
            log_event("app", "ERROR", "Strategy B halted",
                      strategy=self.name, reason=cs.reason)
        elif not cs.halted and self.status.halted:
            self.status.halted, self.status.halt_reason = False, ""
            self.repo.set_halt(self.name, False, "")
        return not self.status.halted
