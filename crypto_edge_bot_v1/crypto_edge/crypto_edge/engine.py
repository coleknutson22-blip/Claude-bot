"""The trading engine.

Cycle order is deliberate and must not be rearranged casually:

    1. safety gates (clock, staleness, DB integrity, circuit breakers)
    2. universe refresh (scheduled)
    3. market data fetch -- unclosed candles discarded immediately
    4. market context (BTC regime, breadth, news, blocks)
    5. POSITION MANAGEMENT (stops and exits) -- always before entries, so a
       stop is never skipped because the entry logic consumed the cycle
    6. signal evaluation across the universe
    7. ranking, then risk gate, then entries
    8. bookkeeping: snapshots, daily roll, heartbeat, counterfactuals

The engine is single-threaded by design. Concurrency between stop processing
and entry processing is the classic source of double-fills, so we simply do
not have it; ordering is enforced by this function, and duplicate protection
is additionally enforced in the database.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .data.broad_universe import (BroadUniverseService,
                                  CoinGeckoUniverseProvider,
                                  StaticBroadUniverseProvider)
from .data.feed import DataUnavailable
from .data.market_age import MarketAgeService
from .data.universe import UniverseBuilder
from .execution.paper_broker import PaperBroker
from .indicators import atr, last_valid, log_returns, roc
from .intel.derivatives import DerivativesEngine
from .intel.events import TokenEventCalendar
from .intel.news import NewsEngine
from .logging_setup import log_event
from .models import Series
from .notify import formatters as fmt
from .notify.telegram import TelegramNotifier
from .performance import PerformanceCalculator
from .portfolio.account import PaperAccount
from .portfolio.risk import RiskManager
from .portfolio.stops import update_stop
from .research.counterfactual import CounterfactualTracker
from .research.journal import (ENTERED, RANKED_OUT, REJECTED_RISK,
                               REJECTED_STRATEGY, ResearchJournal)
from .storage.repo import Repo
from .strategy.base import MarketContext
from .strategy.regime import UNKNOWN, btc_regime, classify_environment, market_breadth
from .strategy.trend_breakout import TrendBreakoutStrategy
from .timeutils import (DAY_MS, candle_id, iso, now_ms, tf_ms, utc_date)


@dataclass
class EngineStatus:
    started_ms: int = 0
    cycles: int = 0
    signals_evaluated: int = 0
    entries: int = 0
    exits: int = 0
    last_data_ms: int = 0
    last_heartbeat_ms: int = 0
    last_daily_report_date: str = ""
    last_universe_refresh_ms: int = 0
    last_snapshot_ms: int = 0
    universe: list[str] = field(default_factory=list)
    broad_universe_ok: bool = False
    broad_universe_reason: str = "not refreshed yet"
    broad_universe_source: str = ""
    broad_universe_n: int = 0
    broad_universe_stale: bool = False
    btc_regime: str = UNKNOWN
    btc_regime_score: float = 50.0
    breadth: float = 50.0
    environment: str = "unknown"
    halted: bool = False
    halt_reason: str = ""
    data_errors: int = 0
    last_cycle_seconds: float = 0.0
    slowest_cycle_seconds: float = 0.0
    consecutive_overruns: int = 0
    next_cycle_ms: int = 0


class TradingEngine:
    def __init__(self, cfg: Config, repo: Repo, feed, notifier: TelegramNotifier,
                 broad_provider=None) -> None:
        self.cfg = cfg
        self.repo = repo
        self.feed = feed
        self.notifier = notifier
        self.broker = PaperBroker(
            cfg.execution.taker_fee_bps, cfg.execution.slippage_bps,
            cfg.execution.stop_slippage_bps, cfg.execution.use_book_spread,
            cfg.execution.max_spread_bps_entry)
        self.account = PaperAccount(repo, self.broker,
                                    cfg.starting_equity_for(cfg.strategy.name),
                                    cfg.strategy.name)
        self.risk = RiskManager(cfg.risk)
        self.strategy = TrendBreakoutStrategy(cfg.strategy)
        self.journal = ResearchJournal(repo)
        self.perf = PerformanceCalculator(repo, cfg.strategy.name)
        # Strategy B rides along on the SAME data: same feed, same candle
        # cache, same universe, same context. It owns its own ledger, entries
        # and exits, so nothing here touches Strategy A's execution path.
        self.aggressive = None
        if getattr(cfg, "aggressive", None) and cfg.aggressive.enabled:
            from .aggressive_runtime import AggressiveRuntime
            self.aggressive = AggressiveRuntime(cfg, repo, feed, notifier,
                                                self.broker, self.journal)
        self.universe_builder = UniverseBuilder(cfg.universe)
        self.market_age = MarketAgeService(
            repo, probe_timeframe=cfg.universe.age_probe_timeframe,
            probe_bars=cfg.universe.age_probe_bars,
            cache_hours=cfg.universe.age_cache_hours)
        self.broad_universe = BroadUniverseService(
            repo, broad_provider or _build_broad_provider(cfg),
            limit=cfg.universe.broad_limit,
            min_assets=cfg.universe.broad_min_assets,
            refresh_hours=cfg.universe.broad_refresh_hours,
            max_cache_age_hours=cfg.universe.broad_max_cache_age_hours,
            collision_scan_limit=cfg.universe.broad_collision_scan_limit,
            symbol_overrides=cfg.universe.broad_symbol_overrides)
        self.news = NewsEngine(repo, [], cfg.intel.news_block_severity,
                               cfg.intel.news_block_hours)
        self.derivatives = DerivativesEngine(repo, [])
        self.calendar = TokenEventCalendar(repo, cfg.intel.token_event_block_hours)
        self.counterfactual = CounterfactualTracker(
            repo, cfg.engine.counterfactual_horizons_h)
        self.status = EngineStatus(started_ms=now_ms())
        self._markets: dict = {}
        # Candles keyed by TIMEFRAME then symbol. A second strategy needs its
        # own timeframes (5m/15m/1h rather than 1h/4h), and the engine fetches
        # the union once so a shared symbol is not downloaded twice.
        self._series: dict[str, dict[str, Series]] = {}
        self._tickers: dict[str, dict] = {}
        # bars asked of the venue per (symbol, timeframe); lets stage 2 separate
        # "the venue has no more history" from "this market is genuinely new"
        self._requested_bars: dict[tuple[str, str], int] = {}
        self._running = False
        repo.record_strategy_version(self.strategy.name, self.strategy.version,
                                     cfg.strategy_fingerprint())

    # ================================================================ helpers
    def check_feed_matches_config(self) -> bool:
        """The feed we are actually pulling prices from must be the venue the
        configuration says. Returns False and logs loudly if they disagree.

        Offline feeds (the fixture feed used by tests and the smoke test) name
        themselves and are exempt -- they are not a venue.
        """
        feed_name = getattr(self.feed, "name", "")
        if not feed_name or feed_name == "fixture":
            return True
        if feed_name != self.cfg.exchange.name:
            log_event("app", "ERROR",
                      "feed/config exchange mismatch -- prices and reports would "
                      "describe different venues",
                      feed=feed_name, config=self.cfg.exchange.name)
            return False
        return True

    @property
    def buffer_ms(self) -> int:
        return self.cfg.safety.candle_close_buffer_s * 1000

    def marks(self) -> dict[str, float]:
        """Current mark price per open position, from the freshest closed candle."""
        out = {}
        for p in self.account.positions():
            s = self._series_1h.get(p.symbol)
            if s is not None and len(s):
                out[p.symbol] = float(s.close[-1])
        return out

    # ============================================================== universe
    def refresh_universe(self, force: bool = False) -> list[str]:
        """Rebuild the tradable universe:

            broad top-N assets (market cap)  ->  intersect with this venue
            ->  static/liquidity/spread filters  ->  cap at max_tradable

        A failure here suppresses NEW ENTRIES only. Open positions are always
        managed (see `fetch_data`), so a third-party ranking API can never
        strand risk we already hold.
        """
        due = (force or self.status.last_universe_refresh_ms == 0 or
               now_ms() - self.status.last_universe_refresh_ms >=
               self.cfg.universe.refresh_minutes * 60_000)
        if not due:
            return self.status.universe
        try:
            self._markets = self.feed.load_markets()
            self._tickers = self.feed.fetch_tickers()
        except DataUnavailable as e:
            log_event("data", "ERROR", "universe refresh failed", error=str(e))
            self.notifier.send_error("universe refresh", str(e))
            return self.status.universe

        broad = self.broad_universe.get(force=force)
        if broad is None:
            # No live ranking and no usable cache: we cannot say what is a
            # legitimate asset, so we do not open anything new. FAIL CLOSED.
            self.status.broad_universe_ok = False
            self.status.broad_universe_reason = (
                self.broad_universe.last_error
                or "no current or cached broad universe available")
            self.status.broad_universe_source = ""
            self.status.broad_universe_n = 0
            self.status.broad_universe_stale = False
            self.status.universe = []
            self.status.last_universe_refresh_ms = now_ms()
            log_event("data", "ERROR",
                      "broad universe unavailable -- new entries suspended",
                      reason=self.status.broad_universe_reason)
            self.notifier.send_error("broad universe",
                                     self.status.broad_universe_reason)
            return self.status.universe

        self.status.broad_universe_ok = True
        self.status.broad_universe_reason = ""
        self.status.broad_universe_source = broad.source
        self.status.broad_universe_n = len(broad)
        self.status.broad_universe_stale = bool(broad.stale)
        if broad.stale:
            # A fresh cache hit is the normal steady state (we refresh every few
            # hours and cycle every few seconds) and is not worth a warning.
            # A STALE one means the provider is failing -- that is.
            log_event("data", "WARNING",
                      "broad universe is stale; provider may be down",
                      **broad.provenance())

        candidates, audit = self.universe_builder.build_candidates(
            self._markets, self._tickers, broad)
        self.repo.add_universe_snapshot(now_ms(), audit)
        self.universe_builder.log_audit(audit, broad)
        # cap the pipeline: the top-N list is for scanning, not for trading
        self.status.universe = candidates[:self.cfg.universe.max_tradable]
        self.status.last_universe_refresh_ms = now_ms()
        return self.status.universe

    def entries_allowed(self) -> tuple[bool, str]:
        """New entries require a universe we can justify. Open positions do not."""
        if not self.cfg.universe.require_broad_universe:
            return True, ""
        if not self.status.broad_universe_ok:
            return False, ("no valid broad asset universe: "
                           + (self.status.broad_universe_reason or "unavailable"))
        return True, ""

    # =========================================================== market data
    def fetch_data(self, symbols: list[str]) -> None:
        """Fetch and immediately discard any unfinished candle.

        Open-position symbols are ALWAYS included, regardless of whether they
        are still in the scanning universe. Without this, a universe provider
        outage (or an asset simply dropping out of the top-N) would leave a
        live position with no candles, hence no stop evaluation -- turning a
        research-data problem into an unmanaged-risk problem.
        """
        c = self.cfg
        held = [p.symbol for p in self.account.positions()]
        needed = list(dict.fromkeys(held + list(symbols) + [c.strategy.btc_symbol]))
        now = now_ms()
        ok = 0
        for sym in needed:
            for tf in self.timeframes():
                store = self._series.setdefault(tf, {})
                # Ask for the history the universe filters actually require, not
                # a fixed page size. Fetching less does not make the filters
                # stricter, it makes them unsatisfiable.
                want = c.required_history_bars(tf)
                try:
                    s = self.feed.fetch_ohlcv(sym, tf, want)
                except DataUnavailable as e:
                    self.status.data_errors += 1
                    log_event("data", "WARNING", "ohlcv fetch failed",
                              symbol=sym, timeframe=tf, error=str(e))
                    store.pop(sym, None)
                    continue
                closed = s.drop_unclosed(now, self.buffer_ms)
                if len(closed) == 0:
                    store.pop(sym, None)
                    continue
                store[sym] = closed
                self._requested_bars[(sym, tf)] = want
                ok += 1
        if ok:
            self.status.last_data_ms = now

    def timeframes(self) -> list[str]:
        """Every timeframe this engine must fetch, in a stable order.

        Today it is one strategy's entry and regime timeframes. It is a list, and
        a set union, so adding a strategy that wants 5m and 15m costs one more
        fetch per symbol rather than a second engine.
        """
        c = self.cfg
        tfs = [c.strategy.entry_timeframe, c.strategy.regime_timeframe]
        if self.aggressive is not None:
            # Only the RANKING timeframe. The 5m and 15m frames are fetched by
            # the scan for the shortlist alone -- pulling them for the whole
            # universe here is exactly the cost the two-phase scan avoids.
            tfs.append(c.aggressive.rank_timeframe)
        return list(dict.fromkeys(tfs))

    def series_for(self, timeframe: str) -> dict[str, Series]:
        return self._series.setdefault(timeframe, {})

    @property
    def _series_1h(self) -> dict[str, Series]:
        """The entry timeframe's candles, whatever that timeframe is named."""
        return self.series_for(self.cfg.strategy.entry_timeframe)

    @property
    def _series_htf(self) -> dict[str, Series]:
        return self.series_for(self.cfg.strategy.regime_timeframe)

    def data_is_stale(self) -> tuple[bool, str]:
        """Refuse to trade on old data. Fail closed."""
        if self.status.last_data_ms == 0:
            return True, "no market data received yet"
        age_s = (now_ms() - self.status.last_data_ms) / 1000.0
        if age_s > self.cfg.safety.max_data_staleness_s:
            return True, f"market data stale by {age_s:.0f}s"
        btc = self._series_1h.get(self.cfg.strategy.btc_symbol)
        if self.cfg.safety.require_btc_data and (btc is None or len(btc) == 0):
            return True, "BTC reference data unavailable"
        if btc is not None and len(btc):
            tf = tf_ms(self.cfg.strategy.entry_timeframe)
            bar_age_s = (now_ms() - (int(btc.open_ms[-1]) + tf)) / 1000.0
            # The most recent CLOSED candle is legitimately up to one full
            # timeframe old, so this tolerance must scale with the timeframe --
            # NOT with max_data_staleness_s, which governs fetch freshness.
            tolerance_s = 2.0 * tf / 1000.0
            if bar_age_s > tolerance_s:
                return True, (f"latest closed {self.cfg.strategy.entry_timeframe} "
                              f"BTC candle is {bar_age_s / 60:.1f} min old "
                              f"(tolerance {tolerance_s / 60:.0f} min)")
        return False, ""

    # ================================================================ context
    def build_context(self) -> MarketContext:
        c = self.cfg
        btc_htf = self._series_htf.get(c.strategy.btc_symbol)
        label, score = btc_regime(btc_htf, c.strategy.regime_ema)
        eligible = {s: self._series_1h[s] for s in self.status.universe
                    if s in self._series_1h}
        breadth = market_breadth(eligible, c.strategy.ema_slow)
        now = now_ms()
        blocked = self.repo.blocked_symbols(now)
        news_ctx = self.news.context_for(list(eligible), now) if c.intel.news_enabled else {}
        deriv_ctx = (self.derivatives.context_for(list(eligible), now)
                     if c.intel.derivatives_enabled else {})

        btc_1h = self._series_1h.get(c.strategy.btc_symbol)
        btc_atr_pct = float("nan")
        if btc_1h is not None and len(btc_1h) > c.strategy.atr_period + 2:
            a = last_valid(atr(btc_1h.high, btc_1h.low, btc_1h.close, c.strategy.atr_period))
            px = float(btc_1h.close[-1])
            if np.isfinite(a) and px > 0:
                btc_atr_pct = a / px * 100.0

        self.status.btc_regime = label
        self.status.btc_regime_score = score
        self.status.breadth = breadth
        self.status.environment = classify_environment(label, breadth, btc_atr_pct)

        return MarketContext(ts_ms=now, btc_regime=label, btc_regime_score=score,
                             breadth_pct=breadth, n_candidates=len(eligible),
                             news_by_symbol=news_ctx, derivatives_by_symbol=deriv_ctx,
                             blocked_symbols=blocked)

    # ==================================================== position management
    def manage_positions(self, ctx: MarketContext) -> None:
        """Stops first, then strategy exits. Runs before any entry logic."""
        c = self.cfg
        for pos in self.account.positions():
            s = self._series_1h.get(pos.symbol)
            if s is None or len(s) == 0 or not s.is_sane():
                log_event("strategy", "WARNING", "no usable data for open position",
                          symbol=pos.symbol)
                continue

            candle = s.last_candle()
            meta = self._markets.get(pos.symbol)

            # --- 1. stop resolution against the completed candle -------------
            fill = self.broker.stop_exit(pos.symbol, pos.qty, pos.current_stop,
                                         candle, meta,
                                         ts_ms=int(s.open_ms[-1]) + tf_ms(c.strategy.entry_timeframe))
            if fill is not None:
                self._close(pos, fill, "stop_loss" if fill.reason == "stop" else "stop_gap")
                continue

            price = float(s.close[-1])

            # --- 2. strategy exits ------------------------------------------
            reason = self.strategy.should_exit(pos, s, ctx)
            if reason:
                # An exit is never blocked for want of a quote -- but a
                # STRUCTURALLY broken one must not price the fill either.
                # Dropping it falls the broker back to the reference price,
                # which is the safe behaviour; refusing to exit is not.
                quote = self._safe_quote(pos.symbol)
                bad = self.broker.quote_structure_error(quote)
                if quote is not None and bad:
                    log_event("data", "WARNING",
                              "ignoring unusable quote for exit; using reference price",
                              symbol=pos.symbol, reason=bad)
                    quote = None
                exit_fill = self.broker.sell(pos.symbol, pos.qty, price, quote, meta,
                                             ts_ms=now_ms(), reason=reason)
                self._close(pos, exit_fill, reason)
                continue

            # --- 3. trail the stop ------------------------------------------
            a = last_valid(atr(s.high, s.low, s.close, c.strategy.atr_period))
            upd = update_stop(pos, price, a if np.isfinite(a) else 0.0, c.strategy)
            if upd.changed:
                prev = pos.current_stop
                self.account.set_stop(pos, upd.new_stop)
                log_event("trades", "INFO", "stop updated", symbol=pos.symbol,
                          previous=prev, new=upd.new_stop, kind=upd.kind)
                if upd.pct_move >= self.cfg.telegram.stop_update_min_pct:
                    self.notifier.send(
                        fmt.stop_update(symbol=pos.symbol, prev_stop=prev,
                                        new_stop=upd.new_stop, price=price,
                                        entry_price=pos.entry_fill_price,
                                        qty=pos.qty, kind=upd.kind),
                        dedupe_key=f"stop:{pos.id}:{round(upd.new_stop, 10)}",
                        kind="stop_update")
        self.account.update_marks(self.marks())

    def _close(self, pos, fill, reason: str) -> None:
        marks = self.marks()
        trade = self.account.close_position(pos, fill, reason, marks)
        if trade is None:
            return
        self.status.exits += 1
        acct = self.account.state
        equity = self.account.equity(self.marks())
        self.notifier.send(
            fmt.exit_trade(
                symbol=trade.symbol, entry_price=trade.entry_fill_price,
                exit_price=trade.exit_fill_price,
                position_value=trade.qty * trade.entry_fill_price,
                held_s=trade.duration_s, gross=trade.gross_pnl, fees=trade.fees,
                slippage=trade.slippage_cost, net=trade.net_pnl,
                return_pct=trade.return_pct,
                account_return_pct=trade.account_return_pct,
                exit_reason=trade.exit_reason, equity=equity,
                total_pnl=float(acct["realized_pnl"])),
            dedupe_key=f"exit:{trade.id}", kind="exit")

    def _safe_quote(self, symbol: str):
        try:
            return self.feed.fetch_quote(symbol)
        except Exception as e:
            log_event("data", "WARNING", "quote fetch failed", symbol=symbol, error=str(e))
            return None

    # ========================================================= signal + entry
    def evaluate_signals(self, ctx: MarketContext) -> list:
        """Evaluate every eligible symbol on its latest CLOSED candle."""
        c = self.cfg
        signals = []
        open_symbols = {p.symbol for p in self.account.positions()}

        btc_1h = self._series_1h.get(c.strategy.btc_symbol)
        btc_roc = float("nan")
        if btc_1h is not None and len(btc_1h) > c.strategy.momentum_periods[-1] + 1:
            btc_roc = last_valid(roc(btc_1h.close, c.strategy.momentum_periods[-1]))

        for sym in self.status.universe:
            s = self._series_1h.get(sym)
            if s is None or len(s) == 0:
                continue
            cid = candle_id(sym, c.strategy.entry_timeframe, int(s.open_ms[-1]))
            if self.repo.is_candle_processed(cid, c.strategy.name):
                continue          # already acted on this bar, even across restarts

            htf = self._series_htf.get(sym)
            ticker = self._tickers.get(sym, {})
            spread = self.universe_builder.spread_bps(ticker)
            meta_extra = {
                "btc_roc": btc_roc,
                "dollar_volume": float(ticker.get("quoteVolume") or 0.0),
                "spread_bps": 0.0 if not np.isfinite(spread) else spread,
                "min_dollar_volume": c.universe.min_dollar_volume_24h,
            }
            # stage-2 universe filter needs candles, so it runs here
            a = last_valid(atr(s.high, s.low, s.close, c.strategy.atr_period))
            px = float(s.close[-1])
            atr_pct = (a / px * 100.0) if (np.isfinite(a) and px > 0) else None
            hist_reject = self.universe_builder.filter_by_history(
                sym, s, atr_pct,
                requested_bars=self._requested_bars.get(
                    (sym, c.strategy.entry_timeframe)))
            if not hist_reject and c.universe.min_market_age_days > 0:
                # Age is a separate question with its own source; see
                # data/market_age.py for why it must not come from `s`.
                verdict = self.market_age.age_of(
                    sym, meta=self._markets.get(sym), feed=self.feed)
                hist_reject = verdict.reason_if_blocked(
                    c.universe.min_market_age_days)

            sig = self.strategy.evaluate(s, htf, ctx, meta_extra)
            self.status.signals_evaluated += 1

            if hist_reject:
                sig.passed = False
                sig.reject_reason = hist_reject

            if sym in open_symbols:
                sig.passed = False
                sig.reject_reason = sig.reject_reason or "position already open"

            cal_ev = self.calendar.blocking_event(sym, ctx.ts_ms)
            if cal_ev and sig.passed:
                sig.passed = False
                sig.reject_reason = f"scheduled {cal_ev['kind']} event within window"

            if not sig.passed:
                self.risk.log_rejection(sym, sig.reject_reason, score=sig.score)
                self.journal.record(sig, REJECTED_STRATEGY)
            signals.append(sig)
        return signals

    def rank_and_enter(self, signals: list, ctx: MarketContext) -> int:
        """Rank all qualifying setups, then spend capital on the best first."""
        c = self.cfg
        qualified = sorted([s for s in signals if s.passed],
                           key=lambda x: x.score, reverse=True)
        entries = 0
        for rank, sig in enumerate(qualified, start=1):
            marks = self.marks()
            equity = self.account.equity(marks)
            exposure = self.account.exposure(marks)
            positions = self.account.positions()
            open_returns = {}
            for p in positions:
                ps = self._series_1h.get(p.symbol)
                if ps is not None and len(ps) > 30:
                    r = log_returns(ps.close)
                    open_returns[p.symbol] = r[-min(len(r), c.risk.correlation_lookback):]

            cand_returns = None
            if sig.returns is not None and len(sig.returns):
                cand_returns = sig.returns[-c.risk.correlation_lookback:]

            decision = self.risk.check_entry(
                symbol=sig.symbol, equity=equity, exposure=exposure,
                n_open=len(positions), open_symbols=[p.symbol for p in positions],
                candidate_returns=cand_returns, open_returns=open_returns,
                entries_this_cycle=entries)
            if not decision.allowed:
                self.risk.log_rejection(sig.symbol, decision.reason,
                                        score=sig.score, rank=rank)
                self.journal.record(sig, REJECTED_RISK, decision.reason, rank)
                continue

            if self._enter(sig, ctx, rank, equity, exposure):
                entries += 1
            if entries >= c.risk.max_new_entries_per_cycle:
                # remaining qualified setups are recorded as ranked out
                for r2, rest in enumerate(qualified[rank:], start=rank + 1):
                    self.journal.record(rest, RANKED_OUT,
                                        "capital allocated to higher-ranked setups", r2)
                break
        return entries

    def _enter(self, sig, ctx: MarketContext, rank: int, equity: float,
               exposure: float) -> bool:
        c = self.cfg
        meta = self._markets.get(sig.symbol)
        if meta is None:
            self.risk.log_rejection(sig.symbol, "exchange metadata unavailable")
            self.journal.record(sig, REJECTED_RISK, "exchange metadata unavailable", rank)
            return False

        # ---- 1. the quote gate: NEW ENTRIES FAIL CLOSED --------------------
        # A missing, invalid, stale, malformed or unavailable quote means we
        # cannot price the trade honestly, so we do not take it. Exits are
        # deliberately exempt (see manage_positions).
        quote = self._safe_quote(sig.symbol)
        qc = self.broker.validate_entry_quote(
            quote, ref_price=sig.ref_price,
            max_age_s=c.execution.max_quote_age_s,
            max_future_skew_s=c.execution.max_quote_future_skew_s,
            max_deviation_pct=c.execution.max_quote_deviation_pct,
            require_venue_timestamp=c.execution.require_venue_quote_timestamp)
        if not qc.ok:
            reason = f"entry blocked: {qc.reason}"
            self.risk.log_rejection(sig.symbol, reason, rank=rank)
            self.journal.record(sig, REJECTED_RISK, reason, rank)
            return False
        spread = qc.spread_bps

        # ---- 2. size against the price we will ACTUALLY fill at ------------
        # The signal's reference price is a closed candle's close. By the time
        # we act, the ask has moved; sizing on the stale reference while filling
        # at the live ask is exactly how a position ends up risking more than
        # the configured budget.
        est_fill = self.broker.expected_entry_price(sig.ref_price, quote, meta)
        sizing = self.broker.size_position(
            equity=equity, cash=self.account.cash(), entry_price=est_fill,
            stop_price=sig.stop_price, risk_pct=c.risk.risk_per_trade_pct,
            max_position_pct=c.risk.max_position_pct, current_exposure=exposure,
            max_exposure_pct=c.risk.max_portfolio_exposure_pct, meta=meta,
            min_stop_distance_pct=c.risk.min_stop_distance_pct)
        if not sizing.ok:
            self.risk.log_rejection(sig.symbol, sizing.reason)
            self.journal.record(sig, REJECTED_RISK, sizing.reason, rank)
            return False

        fill = self.broker.buy(sig.symbol, sizing.qty, sig.ref_price, quote, meta,
                               ts_ms=now_ms())

        # ---- 3. revalidate against the realised simulated fill -------------
        # Belt and braces: recompute risk from the fill itself, so no future
        # change to the fill model can quietly reintroduce over-sizing.
        ok_risk, risk_reason, actual_risk = self.broker.revalidate_risk(
            qty=sizing.qty, fill_price=fill.fill_price, stop_price=sig.stop_price,
            equity=equity, risk_pct=c.risk.risk_per_trade_pct,
            tolerance_pct=c.execution.risk_overshoot_tolerance_pct)
        if not ok_risk:
            self.risk.log_rejection(sig.symbol, risk_reason, rank=rank)
            self.journal.record(sig, REJECTED_RISK, risk_reason, rank)
            return False

        journal = self.journal.entry_journal(sig, {
            "account_equity": equity, "position_value": fill.notional,
            "risk_amount": actual_risk, "rank": rank,
            "environment": self.status.environment,
            "spread_bps": spread,
            "quote_age_s": qc.age_s,
            "quote_ts_source": qc.ts_source,
            "quote_age_verified": qc.age_verified,
            "entry_ref_price": sig.ref_price,
            "expected_fill_price": est_fill,
            "entry_fill_price": fill.fill_price,
            "risk_budget": sizing.risk_budget,
            "risk_at_fill": actual_risk,
            "est_cost_at_stop": sizing.est_cost_at_stop,
            "config_fingerprint_version": self.strategy.version,
            "derivatives": ctx.derivatives_by_symbol.get(sig.symbol),
            "news": ctx.news_by_symbol.get(sig.symbol),
        })

        pos = self.account.open_position(
            symbol=sig.symbol, strategy=sig.strategy, strategy_version=sig.version,
            qty=sizing.qty, ref_price=sig.ref_price, fill=fill,
            initial_stop=sig.stop_price, risk_amount=actual_risk,
            candle_id=sig.candle_id, signal_score=sig.score, journal=journal)
        if pos is None:
            self.journal.record(sig, REJECTED_RISK, "duplicate or insufficient cash", rank)
            return False

        self.status.entries += 1
        self.journal.record(sig, ENTERED, "", rank)
        new_equity = self.account.equity(self.marks())
        reasons = _entry_reasons(sig)
        self.notifier.send(
            fmt.entry(symbol=sig.symbol, side="long", entry_price=fill.fill_price,
                      qty=sizing.qty, position_value=fill.notional,
                      pct_of_account=fill.notional / equity * 100.0 if equity else 0.0,
                      stop=sig.stop_price, dollar_risk=actual_risk,
                      pct_risk=actual_risk / equity * 100.0 if equity else 0.0,
                      score=sig.score, htf_regime=sig.features.get("htf_regime", "?"),
                      btc_regime=ctx.btc_regime, breadth=ctx.breadth_pct,
                      reasons=reasons, equity=new_equity),
            dedupe_key=f"entry:{sig.candle_id}", kind="entry")
        return True

    # =============================================================== breakers
    def check_safety(self) -> bool:
        """Returns True if trading may proceed. Fails closed on every doubt."""
        acct = self.account.state
        if int(acct["halted"]):
            self.status.halted = True
            self.status.halt_reason = acct["halt_reason"]
            return False

        marks = self.marks()
        equity = self.account.equity(marks)
        cs = self.risk.check_circuit_breakers(
            equity, float(acct["peak_equity"]), float(acct["daily_start_equity"]))
        if cs.halted:
            self.repo.set_halt(self.cfg.strategy.name, True, cs.reason)
            self.status.halted = True
            self.status.halt_reason = cs.reason
            kind = ("MAX DRAWDOWN" if "DRAWDOWN" in cs.reason.upper() else "DAILY LOSS")
            log_event("performance", "ERROR", "circuit breaker tripped", reason=cs.reason)
            self.notifier.send(
                fmt.circuit_breaker(kind=kind, reason=cs.reason, equity=equity,
                                    action="New entries stopped; open positions still managed"),
                dedupe_key=f"cb:{kind}:{utc_date(now_ms())}", kind="circuit_breaker")
            return False
        return True

    # ============================================================ bookkeeping
    def bookkeeping(self) -> None:
        c, now = self.cfg, now_ms()
        marks = self.marks()
        equity = self.account.equity(marks)

        if self.account.roll_daily(now, marks):
            # a new day clears the DAILY loss halt, but never the drawdown halt
            acct = self.account.state
            if int(acct["halted"]) and "DAILY LOSS" in acct["halt_reason"].upper():
                self.repo.set_halt(self.cfg.strategy.name, False, "")
                self.status.halted = False
                self.status.halt_reason = ""
                log_event("performance", "INFO", "daily loss halt cleared by rollover")

        if now - self.status.last_snapshot_ms >= c.engine.equity_snapshot_minutes * 60_000:
            self.repo.add_equity_snapshot(
                now, equity, self.account.cash(), self.account.unrealized(marks),
                len(self.account.positions()), self.account.drawdown_pct(marks))
            self.status.last_snapshot_ms = now

        if now - self.status.last_heartbeat_ms >= c.telegram.heartbeat_minutes * 60_000:
            self._heartbeat(equity)
            self.status.last_heartbeat_ms = now

        today = utc_date(now)
        from .timeutils import to_dt
        if (self.status.last_daily_report_date != today
                and to_dt(now).hour >= c.telegram.daily_report_hour_utc
                and self.status.last_daily_report_date != ""):
            self._daily_report(today)
            self.status.last_daily_report_date = today
        elif self.status.last_daily_report_date == "":
            self.status.last_daily_report_date = today

        try:
            self.counterfactual.evaluate_pending(lambda sym: self._series_1h.get(sym))
        except Exception as e:
            log_event("app", "WARNING", "counterfactual evaluation failed", error=str(e))

        self.repo.prune_processed_candles(now - 30 * DAY_MS)
        # Only DELIVERED rows are prunable; PENDING and FAILED are kept so an
        # undelivered critical message can never be silently discarded.
        self.repo.prune_telegram_outbox(
            now - c.telegram.outbox_retention_days * DAY_MS)

    def _heartbeat(self, equity: float) -> None:
        acct = self.account.state
        day = self.repo.get_daily(utc_date(now_ms()))
        today_pnl = equity - float(acct["daily_start_equity"])
        self.notifier.send(
            fmt.heartbeat(
                uptime_s=(now_ms() - self.status.started_ms) / 1000.0, equity=equity,
                today_pnl=today_pnl, total_pnl=equity - float(acct["starting_equity"]),
                open_positions=len(self.account.positions()),
                btc_regime=self.status.btc_regime, breadth=self.status.breadth,
                signals_evaluated=self.status.signals_evaluated,
                last_data_ms=self.status.last_data_ms or now_ms(),
                halted=self.status.halted,
                cycle_s=self.status.last_cycle_seconds,
                poll_s=float(self.cfg.engine.poll_seconds),
                overruns=self.status.consecutive_overruns),
            kind="heartbeat")

    def _daily_report(self, date: str) -> None:
        marks = self.marks()
        rep = self.perf.report(marks).as_dict()
        acct = self.account.state
        top = [{"symbol": o["symbol"], "score": o["score"]}
               for o in sorted(self.repo.get_observations(since_ms=now_ms() - DAY_MS),
                               key=lambda r: (r["score"] or 0), reverse=True)[:5]]
        events = []
        for sym, reason in self.repo.blocked_symbols(now_ms()).items():
            events.append(f"WARNING — {sym} entries blocked: {reason[:80]}")
        self.notifier.send(
            fmt.daily_report(
                date=date, uptime_s=(now_ms() - self.status.started_ms) / 1000.0,
                rep=rep, top_setups=top, btc_regime=self.status.btc_regime,
                breadth=self.status.breadth,
                strategy_status="HALTED: " + self.status.halt_reason
                if self.status.halted else "ACTIVE",
                daily_loss_remaining=self.risk.daily_loss_remaining(
                    self.account.equity(marks), float(acct["daily_start_equity"])),
                events=events),
            dedupe_key=f"daily:{date}", kind="daily_report")

    # ================================================================== cycle
    def cycle(self) -> None:
        self.status.cycles += 1
        # Replay anything the notifier wrote but never delivered -- including
        # messages left PENDING by a process that died mid-send.
        self.notifier.flush_pending()
        universe = self.refresh_universe()
        self.fetch_data(universe)

        stale, why = self.data_is_stale()
        if stale:
            log_event("data", "ERROR", "trading suspended: data not trustworthy",
                      reason=why)
            self.notifier.send(
                fmt.circuit_breaker(kind="STALE DATA", reason=why,
                                    equity=self.account.equity(self.marks()),
                                    action="No signals evaluated this cycle"),
                dedupe_key=f"stale:{utc_date(now_ms())}:{int(now_ms() / 3_600_000)}",
                kind="circuit_breaker")
            self.bookkeeping()
            return

        ctx = self.build_context()
        self.manage_positions(ctx)          # ALWAYS before entries

        # Circuit breakers are evaluated FIRST and unconditionally: a drawdown
        # halt must still trip and persist even on a cycle where we already
        # know no entries are possible for some other reason.
        safe = self.check_safety()
        universe_ok, universe_why = self.entries_allowed()

        if not safe:
            log_event("strategy", "WARNING", "entries suppressed",
                      reason=self.status.halt_reason)
        elif not universe_ok:
            log_event("strategy", "ERROR", "entries suspended", reason=universe_why)
            self.notifier.send(
                fmt.circuit_breaker(
                    kind="UNIVERSE UNAVAILABLE", reason=universe_why,
                    equity=self.account.equity(self.marks()),
                    action="No new entries; open positions still managed"),
                dedupe_key=f"universe:{utc_date(now_ms())}:"
                           f"{int(now_ms() / 3_600_000)}",
                kind="circuit_breaker")
        else:
            signals = self.evaluate_signals(ctx)
            self.rank_and_enter(signals, ctx)
        self._run_aggressive(ctx, entries_allowed=bool(safe and universe_ok))
        self.bookkeeping()

    def _run_aggressive(self, ctx: MarketContext, *, entries_allowed: bool) -> None:
        """One Strategy B pass. Never raises into Strategy A's cycle.

        B is an experiment running beside a strategy with real recorded
        history. A fault in the experiment must not stop the incumbent from
        managing its own positions, so the whole pass is contained: it is
        logged and the cycle continues.
        """
        rt = self.aggressive
        if rt is None:
            return
        a = self.cfg.aggressive
        try:
            rank_series = dict(self.series_for(a.rank_timeframe))
            held = {p.symbol for p in rt.account.positions()}
            frames_by_symbol = {
                sym: {tf: self.series_for(tf).get(sym) for tf in a.timeframes}
                for sym in held}
            # A held symbol's fast frames are not in the engine's store, so
            # fetch them here: an open position must always be manageable.
            for sym in held:
                for tf in a.timeframes:
                    if frames_by_symbol[sym].get(tf) is None:
                        try:
                            raw = self.feed.fetch_ohlcv(
                                sym, tf, self.cfg.required_history_bars(tf))
                            frames_by_symbol[sym][tf] = raw.drop_unclosed(
                                now_ms(), self.buffer_ms)
                        except DataUnavailable as e:
                            log_event("data", "WARNING",
                                      "no data for open Strategy B position",
                                      symbol=sym, timeframe=tf, error=str(e))
            rt.manage(ctx, frames_by_symbol, self._markets)
            series_5m = {s: fr.get("5m") for s, fr in frames_by_symbol.items()}
            if not rt.check_safety(series_5m):
                return
            meta_by = {}
            for sym in rank_series:
                t = (self._tickers or {}).get(sym) or {}
                qv, _ = self.universe_builder.quote_volume(t)
                meta_by[sym] = {"dollar_volume": qv,
                                "spread_bps": self.universe_builder.spread_bps(t)}
            rt.scan_and_enter(ctx, rank_series=rank_series,
                              btc_1h=rank_series.get(self.cfg.strategy.btc_symbol),
                              meta_by_symbol=meta_by, markets=self._markets,
                              entries_allowed=entries_allowed)
        except Exception as e:
            log_event("app", "ERROR", "Strategy B pass failed",
                      strategy=a.name, error=str(e))

    def pause_after_cycle(self, elapsed: float) -> float:
        """Seconds to wait before starting the next cycle. NEVER ZERO.

        Cycles cannot overlap -- `run` is single-threaded and calls `cycle()`
        to completion before looking at the clock -- so there is no queue to
        drain and no possibility of two cycles touching the database at once.
        What CAN go wrong is subtler: the old code slept
        `max(0, poll_seconds - elapsed)`, which for a cycle slower than the poll
        interval is exactly zero. An 11-symbol Kraken cycle measured 74.6s
        against a 30s interval, so the bot ran flat out with no gap between
        requests, no CPU idle, and nothing anywhere saying it was behind. That
        is how a paper bot earns a rate-limit ban.

        A floor on the pause converts "as fast as possible" into a slower but
        honest cadence of max(poll_seconds, elapsed + min_pause_seconds), and
        the overrun is logged with the time the next cycle will start.
        """
        poll = float(self.cfg.engine.poll_seconds)
        floor = max(0.0, float(getattr(self.cfg.engine, "min_pause_seconds", 0.0)))
        if elapsed > poll:
            self.status.consecutive_overruns += 1
            log_event("app", "WARNING", "cycle slower than poll interval",
                      cycle_seconds=round(elapsed, 1), poll_seconds=poll,
                      pausing_seconds=round(floor, 1),
                      consecutive=self.status.consecutive_overruns)
            return floor
        self.status.consecutive_overruns = 0
        return max(floor, poll - elapsed)

    def run(self, max_cycles: int | None = None, sleep=time.sleep) -> None:
        self._running = True
        n = 0
        while self._running:
            start = time.time()
            try:
                self.cycle()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log_event("app", "ERROR", "cycle failed", error=str(e))
                self.notifier.send_error("engine cycle", str(e))
            n += 1
            elapsed = time.time() - start
            self.status.last_cycle_seconds = elapsed
            self.status.slowest_cycle_seconds = max(
                self.status.slowest_cycle_seconds, elapsed)
            if max_cycles is not None and n >= max_cycles:
                self.status.next_cycle_ms = 0
                break
            pause = self.pause_after_cycle(elapsed)
            self.status.next_cycle_ms = now_ms() + int(pause * 1000)
            sleep(pause)

    def stop(self) -> None:
        self._running = False

    def announce_start(self) -> None:
        self.check_feed_matches_config()
        marks = self.marks()
        self.notifier.send(
            # cfg.exchange_label(), not feed.name: the configuration is the
            # single source of truth for which venue we are on, and every other
            # surface (selfcheck, status, verify-live) reports the same string.
            # A feed whose own name disagrees is a wiring bug, reported below.
            fmt.bot_start(mode=self.cfg.safety.mode,
                          exchange=self.cfg.exchange_label(),
                          equity=self.account.equity(marks), cash=self.account.cash(),
                          open_positions=len(self.account.positions()),
                          strategy=self.strategy.name, version=self.strategy.version,
                          universe_size=len(self.status.universe),
                          telegram_ok=self.notifier.enabled),
            kind="bot_start")


def _build_broad_provider(cfg: Config):
    """Construct the configured broad-universe provider, or None.

    `CoinGeckoUniverseProvider` imports `requests` inside `fetch()`, so
    constructing it here costs nothing and needs no HTTP dependency present.
    """
    src = cfg.universe.broad_source
    if src == "coingecko":
        return CoinGeckoUniverseProvider()
    if src == "static":
        return StaticBroadUniverseProvider(cfg.universe.broad_static_assets)
    return None


def _entry_reasons(sig) -> list[str]:
    f, out = sig.features, []
    if f.get("donchian_high"):
        out.append(f"broke {f['donchian_high']:.6g} Donchian high")
    if f.get("adx"):
        out.append(f"ADX {f['adx']:.0f}")
    if f.get("rel_volume"):
        out.append(f"rel vol {f['rel_volume']:.1f}x")
    mom = f.get("momentum") or {}
    if mom:
        k = max(mom)
        v = mom[k]
        if v is not None and np.isfinite(v):
            out.append(f"{k}h momentum {v:+.1f}%")
    return out
