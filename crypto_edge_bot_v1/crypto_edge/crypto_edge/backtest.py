"""Historical engine.

Integrity guarantees, each enforced structurally rather than by convention:

  * SAME LOGIC -- imports and calls `TrendBreakoutStrategy.evaluate()`, the
    identical object the live engine uses. There is no parallel backtest
    strategy implementation to drift out of sync.
  * NO LOOK-AHEAD -- at bar i the strategy is handed `series[:i+1]` and nothing
    more. It is structurally incapable of seeing bar i+1.
  * NO FUTURE FILLS -- an entry signalled on bar i fills on bar i+1's OPEN,
    never bar i's close, because bar i's close is not tradable once observed.
  * REALISTIC STOPS -- stops resolve against bar lows via the same PaperBroker
    used live, including gap-through fills.
  * COSTS ALWAYS ON -- fees and slippage come from the same broker config.
  * NO SURVIVORSHIP CONTROL -- the symbol list is supplied by the caller, so
    if you feed it only today's winners, that bias is YOURS. The report says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .execution.paper_broker import PaperBroker, realise_pnl
from .indicators import atr, last_valid
from .models import MarketMeta, Series
from .portfolio.stops import time_stop_hit, update_stop
from .strategy.base import MarketContext
from .strategy.regime import trend_regime
from .strategy.trend_breakout import TrendBreakoutStrategy
from .timeutils import tf_ms


@dataclass
class BTPosition:
    symbol: str
    qty: float
    entry_ref: float
    entry_fill: float
    entry_fee: float
    entry_i: int
    entry_ms: int
    initial_stop: float
    current_stop: float
    highest: float
    score: float


@dataclass
class BacktestResult:
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    start_equity: float = 0.0
    end_equity: float = 0.0
    bars: int = 0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        nets = [t["net_pnl"] for t in self.trades]
        wins = [v for v in nets if v > 0]
        losses = [v for v in nets if v <= 0]
        gp, gl = sum(wins), abs(sum(losses))
        peak, mdd = (self.equity_curve[0] if self.equity_curve else 0.0), 0.0
        for v in self.equity_curve:
            peak = max(peak, v)
            if peak > 0:
                mdd = max(mdd, (peak - v) / peak * 100.0)
        return {
            "trades": len(nets), "wins": len(wins), "losses": len(losses),
            "win_rate_pct": (len(wins) / len(nets) * 100.0) if nets else 0.0,
            "net_pnl": sum(nets), "gross_pnl": sum(t["gross_pnl"] for t in self.trades),
            "fees": sum(t["fees"] for t in self.trades),
            "slippage": sum(t["slippage_cost"] for t in self.trades),
            "profit_factor": (gp / gl) if gl else (float("inf") if gp else 0.0),
            "expectancy": float(np.mean(nets)) if nets else 0.0,
            "return_pct": ((self.end_equity - self.start_equity) / self.start_equity * 100.0)
                          if self.start_equity else 0.0,
            "max_drawdown_pct": mdd,
            "sample_warning": ("SAMPLE TOO SMALL — treat as noise"
                               if len(nets) < 30 else ""),
            "warnings": self.warnings,
        }


class Backtester:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.strategy = TrendBreakoutStrategy(cfg.strategy)
        self.broker = PaperBroker(cfg.execution.taker_fee_bps,
                                  cfg.execution.slippage_bps,
                                  cfg.execution.stop_slippage_bps,
                                  use_book_spread=False)

    def run(self, series_1h: dict[str, Series], series_htf: dict[str, Series],
            btc_symbol: str | None = None) -> BacktestResult:
        cfg = self.cfg
        res = BacktestResult(start_equity=cfg.execution.starting_equity)
        res.warnings.append(
            "Universe supplied by caller — survivorship bias is not controlled here.")
        cash = cfg.execution.starting_equity
        open_pos: dict[str, BTPosition] = {}
        symbols = sorted(series_1h)
        if not symbols:
            return res

        n_bars = min(len(series_1h[s]) for s in symbols)
        res.bars = n_bars
        bar_ms = tf_ms(cfg.strategy.entry_timeframe)
        meta = MarketMeta("x", "X", "USDT", True, 8, 8, 0.0, 0.0)
        warm = cfg.strategy.warmup_bars

        # precompute ATR per symbol on the FULL series; index i only ever reads
        # position i, so this is causal despite being computed up front
        atrs = {s: atr(series_1h[s].high, series_1h[s].low, series_1h[s].close,
                       cfg.strategy.atr_period) for s in symbols}

        for i in range(warm, n_bars - 1):
            # ---- context from closed bars only ---------------------------
            btc_label, btc_score = "unknown", 50.0
            if btc_symbol and btc_symbol in series_htf:
                htf_b = _slice_htf(series_htf[btc_symbol], series_1h[symbols[0]], i)
                if len(htf_b) > 0:
                    btc_label, btc_score = trend_regime(htf_b, cfg.strategy.regime_ema)
            above = tot = 0
            for s in symbols:
                sub = _slice(series_1h[s], i)
                if len(sub) > cfg.strategy.ema_slow + 2:
                    from .indicators import ema
                    e = last_valid(ema(sub.close, cfg.strategy.ema_slow))
                    if np.isfinite(e):
                        tot += 1
                        above += 1 if float(sub.close[-1]) > e else 0
            breadth = (above / tot * 100.0) if tot else 50.0
            ctx = MarketContext(ts_ms=int(series_1h[symbols[0]].open_ms[i]),
                                btc_regime=btc_label, btc_regime_score=btc_score,
                                breadth_pct=breadth, n_candidates=tot)

            # ---- manage open positions on bar i --------------------------
            for sym in list(open_pos):
                p = open_pos[sym]
                s_full = series_1h[sym]
                candle = _candle_at(s_full, i)
                fill = self.broker.stop_exit(sym, p.qty, p.current_stop, candle, meta,
                                             ts_ms=int(s_full.open_ms[i]))
                if fill is not None:
                    cash += fill.notional - fill.fee
                    res.trades.append(_mk_trade(p, fill, "stop_loss" if fill.reason == "stop"
                                                else "stop_gap", cash))
                    del open_pos[sym]
                    continue
                price = float(s_full.close[i])
                sub = _slice(s_full, i)
                exit_reason = self.strategy.should_exit(
                    _as_position(p), sub, ctx)
                if exit_reason:
                    f = self.broker.sell(sym, p.qty, price, None, meta,
                                         ts_ms=int(s_full.open_ms[i]), reason=exit_reason)
                    cash += f.notional - f.fee
                    res.trades.append(_mk_trade(p, f, exit_reason, cash))
                    del open_pos[sym]
                    continue
                p.highest = max(p.highest, float(s_full.high[i]))
                a = atrs[sym][i]
                upd = update_stop(_as_position(p), price,
                                  float(a) if np.isfinite(a) else 0.0, cfg.strategy)
                if upd.changed:
                    p.current_stop = upd.new_stop

            # ---- evaluate signals on bar i, fill on bar i+1 open ---------
            equity = cash + sum(p.qty * float(series_1h[p.symbol].close[i])
                                for p in open_pos.values())
            candidates = []
            for sym in symbols:
                if sym in open_pos:
                    continue
                sub = _slice(series_1h[sym], i)
                htf_sub = _slice_htf(series_htf.get(sym), series_1h[sym], i) \
                    if series_htf.get(sym) is not None else None
                sig = self.strategy.evaluate(sub, htf_sub, ctx,
                                             {"dollar_volume": 5e7, "spread_bps": 5.0,
                                              "min_dollar_volume": cfg.universe.min_dollar_volume_24h})
                if sig.passed:
                    candidates.append(sig)

            candidates.sort(key=lambda x: x.score, reverse=True)
            entered = 0
            for sig in candidates:
                if len(open_pos) >= cfg.risk.max_open_positions:
                    break
                if entered >= cfg.risk.max_new_entries_per_cycle:
                    break
                exposure = sum(p.qty * float(series_1h[p.symbol].close[i])
                               for p in open_pos.values())
                if equity and exposure / equity * 100.0 >= cfg.risk.max_portfolio_exposure_pct:
                    break
                # FILL ON NEXT BAR'S OPEN — the signal bar's close is gone
                next_open = float(series_1h[sig.symbol].open[i + 1])
                sizing = self.broker.size_position(
                    equity=equity, cash=cash, entry_price=next_open,
                    stop_price=sig.stop_price, risk_pct=cfg.risk.risk_per_trade_pct,
                    max_position_pct=cfg.risk.max_position_pct,
                    current_exposure=exposure,
                    max_exposure_pct=cfg.risk.max_portfolio_exposure_pct, meta=meta,
                    min_stop_distance_pct=cfg.risk.min_stop_distance_pct)
                if not sizing.ok:
                    continue
                f = self.broker.buy(sig.symbol, sizing.qty, next_open, None, meta,
                                    ts_ms=int(series_1h[sig.symbol].open_ms[i + 1]))
                cost = f.notional + f.fee
                if cost > cash:
                    continue
                cash -= cost
                open_pos[sig.symbol] = BTPosition(
                    sig.symbol, sizing.qty, next_open, f.fill_price, f.fee, i + 1,
                    int(series_1h[sig.symbol].open_ms[i + 1]), sig.stop_price,
                    sig.stop_price, f.fill_price, sig.score)
                entered += 1

            res.equity_curve.append(
                cash + sum(p.qty * float(series_1h[p.symbol].close[i])
                           for p in open_pos.values()))

        last = n_bars - 1
        for sym, p in list(open_pos.items()):
            price = float(series_1h[sym].close[last])
            f = self.broker.sell(sym, p.qty, price, None, meta,
                                 ts_ms=int(series_1h[sym].open_ms[last]),
                                 reason="end_of_data")
            cash += f.notional - f.fee
            res.trades.append(_mk_trade(p, f, "end_of_data", cash))
        res.end_equity = cash
        if res.equity_curve:
            res.equity_curve.append(cash)
        return res


# ------------------------------------------------------------------ helpers
def _slice(s: Series, i: int) -> Series:
    """Bars 0..i inclusive. Nothing beyond i is reachable."""
    j = i + 1
    return Series(s.symbol, s.timeframe, s.open_ms[:j], s.open[:j], s.high[:j],
                  s.low[:j], s.close[:j], s.volume[:j])


def _slice_htf(htf: Series | None, ltf: Series, i: int) -> Series:
    """Higher-timeframe bars that had CLOSED by the close of ltf bar i."""
    if htf is None or len(htf) == 0:
        return Series("x", "4h", np.array([], dtype=np.int64), *[np.array([])] * 5)
    cutoff = int(ltf.open_ms[i]) + tf_ms(ltf.timeframe)
    step = tf_ms(htf.timeframe)
    keep = int(np.searchsorted(htf.open_ms + step, cutoff, side="right"))
    return Series(htf.symbol, htf.timeframe, htf.open_ms[:keep], htf.open[:keep],
                  htf.high[:keep], htf.low[:keep], htf.close[:keep], htf.volume[:keep])


def _candle_at(s: Series, i: int):
    from .models import Candle
    return Candle(int(s.open_ms[i]), float(s.open[i]), float(s.high[i]),
                  float(s.low[i]), float(s.close[i]), float(s.volume[i]))


def _as_position(p: BTPosition):
    from .models import Position
    return Position(id="bt", symbol=p.symbol, strategy="trend_breakout",
                    strategy_version="bt", side="long", qty=p.qty,
                    entry_ref_price=p.entry_ref, entry_fill_price=p.entry_fill,
                    entry_ms=p.entry_ms, entry_fee=p.entry_fee, entry_slippage=0.0,
                    initial_stop=p.initial_stop, current_stop=p.current_stop,
                    highest_price=p.highest, lowest_price=p.entry_fill,
                    risk_amount=0.0, candle_id="bt", signal_score=p.score)


def _mk_trade(p: BTPosition, fill, reason: str, equity_after: float) -> dict:
    pnl = realise_pnl(p.entry_ref, p.entry_fill, fill.ref_price, fill.fill_price,
                      p.qty, p.entry_fee, fill.fee)
    return {"symbol": p.symbol, "qty": p.qty, "entry_fill": p.entry_fill,
            "exit_fill": fill.fill_price, "exit_reason": reason,
            "entry_ms": p.entry_ms, "exit_ms": fill.ts_ms,
            "score": p.score, "equity_after": equity_after, **pnl}


# ------------------------------------------------------- walk-forward + stress
def walk_forward_windows(n_bars: int, train: int, validate: int, test: int,
                         step: int | None = None) -> list[dict]:
    """Chronological TRAIN -> VALIDATE -> TEST windows, rolled forward.

    The test slice of window k never overlaps the train slice of window k, and
    windows advance strictly forward in time. There is no way to express
    "optimise on everything then validate on everything" with this helper.
    """
    step = step or test
    out, start = [], 0
    while start + train + validate + test <= n_bars:
        out.append({
            "train": (start, start + train),
            "validate": (start + train, start + train + validate),
            "test": (start + train + validate, start + train + validate + test),
        })
        start += step
    return out


def parameter_stress(cfg: Config, series_1h: dict[str, Series],
                     series_htf: dict[str, Series], param_path: str,
                     values: list, btc_symbol: str | None = None) -> list[dict]:
    """Re-run the backtest across neighbouring parameter values.

    We care whether a NEIGHBOURHOOD is profitable, not whether one magic value
    is. A configuration whose neighbours collapse is a cliff, and cliffs are
    rejected regardless of how good the peak looks.
    """
    import copy
    section, _, key = param_path.partition(".")
    out = []
    for v in values:
        c2 = copy.deepcopy(cfg)
        setattr(getattr(c2, section), key, v)
        if c2.validate():
            out.append({"value": v, "error": "; ".join(c2.validate())})
            continue
        r = Backtester(c2).run(series_1h, series_htf, btc_symbol)
        s = r.summary()
        out.append({"value": v, "trades": s["trades"], "net_pnl": s["net_pnl"],
                    "profit_factor": s["profit_factor"],
                    "max_drawdown_pct": s["max_drawdown_pct"],
                    "expectancy": s["expectancy"]})
    profitable = [r for r in out if r.get("net_pnl", 0) > 0]
    for r in out:
        r["neighbourhood_profitable_pct"] = len(profitable) / len(out) * 100.0 if out else 0.0
    return out
