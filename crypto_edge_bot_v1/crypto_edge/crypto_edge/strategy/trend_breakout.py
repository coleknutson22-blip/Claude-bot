"""Strategy A: higher-timeframe trend + momentum + Donchian breakout (long only).

Contract with the rest of the system:
  * `evaluate()` receives ONLY closed candles. It has no access to the current
    bar, the account, or the clock. Given identical inputs it returns an
    identical Signal -- which is what makes backtest and live provably equal.
  * It returns a Signal even when rejecting, carrying the reason and the full
    feature vector, so the research database can learn from rejections.
"""
from __future__ import annotations

import numpy as np

from ..config import StrategyCfg
from ..indicators import (adx, atr, donchian, ema, last_valid, log_returns,
                          realized_vol, rel_volume, roc)
from ..models import Position, Series
from ..portfolio.stops import initial_stop, time_stop_hit
from ..timeutils import candle_id, tf_ms
from . import scoring
from .base import MarketContext, Signal
from .regime import BEAR, UNKNOWN, trend_regime


class TrendBreakoutStrategy:
    name = "trend_breakout"

    def __init__(self, cfg: StrategyCfg) -> None:
        self.cfg = cfg
        self.version = cfg.version

    # ------------------------------------------------------------ features
    def compute_features(self, s: Series, htf: Series | None,
                         ctx: MarketContext) -> dict:
        c = self.cfg
        close, high, low, vol = s.close, s.high, s.low, s.volume
        a = atr(high, low, close, c.atr_period)
        adx_v, pdi, mdi = adx(high, low, close, c.adx_period)
        from ..indicators import rsi as _rsi
        rsi_v = _rsi(close, c.rsi_period)
        dch, dcl = donchian(high, low, c.donchian_lookback, exclude_current=True)
        rv = rel_volume(vol, 24)
        e_fast = ema(close, c.ema_fast)
        e_slow = ema(close, c.ema_slow)
        e_trend = ema(close, c.ema_trend)
        rocs = {p: last_valid(roc(close, p)) for p in c.momentum_periods}
        rvol_pct = last_valid(realized_vol(close, 24))

        price = float(close[-1])
        atr_v = last_valid(a)
        atr_pct = (atr_v / price * 100.0) if (price > 0 and np.isfinite(atr_v)) else float("nan")

        htf_label, htf_score = (UNKNOWN, 50.0)
        if htf is not None and len(htf) > 0:
            htf_label, htf_score = trend_regime(htf, self.cfg.regime_ema)

        return {
            "price": price,
            "atr": atr_v,
            "atr_pct": atr_pct,
            "adx": last_valid(adx_v),
            "plus_di": last_valid(pdi),
            "minus_di": last_valid(mdi),
            "rsi": last_valid(rsi_v),
            "donchian_high": last_valid(dch),
            "donchian_low": last_valid(dcl),
            "rel_volume": last_valid(rv),
            "ema_fast": last_valid(e_fast),
            "ema_slow": last_valid(e_slow),
            "ema_trend": last_valid(e_trend),
            "realized_vol_pct": rvol_pct,
            "momentum": rocs,
            "htf_regime": htf_label,
            "htf_score": htf_score,
            "btc_regime": ctx.btc_regime,
            "btc_regime_score": ctx.btc_regime_score,
            "breadth_pct": ctx.breadth_pct,
            "bar_open_ms": int(s.open_ms[-1]),
        }

    # ------------------------------------------------------------ evaluate
    def evaluate(self, s: Series, htf: Series | None, ctx: MarketContext,
                 meta: dict | None = None) -> Signal:
        c = self.cfg
        cid = candle_id(s.symbol, c.entry_timeframe, int(s.open_ms[-1])) if len(s) else ""
        meta = meta or {}

        def reject(reason: str, f: dict | None = None, score: float = 0.0) -> Signal:
            return Signal(symbol=s.symbol, strategy=self.name, version=self.version,
                          candle_id=cid, ts_ms=int(s.open_ms[-1]) if len(s) else ctx.ts_ms,
                          ref_price=float(f.get("price", 0.0)) if f else 0.0,
                          stop_price=0.0, atr=float(f.get("atr", 0.0)) if f else 0.0,
                          score=score, passed=False, reject_reason=reason,
                          features=f or {})

        # ---- data sufficiency: fail closed --------------------------------
        if len(s) < c.warmup_bars:
            return reject(f"insufficient history ({len(s)} < {c.warmup_bars} bars)")
        if not s.is_sane():
            return reject("candle data failed sanity check")

        f = self.compute_features(s, htf, ctx)
        needed = ("atr", "adx", "rsi", "donchian_high", "rel_volume",
                  "ema_fast", "ema_slow", "ema_trend")
        missing = [k for k in needed if not np.isfinite(f.get(k, float("nan")))]
        if missing:
            return reject(f"indicator not ready: {','.join(missing)}", f)
        if f["atr"] <= 0 or f["price"] <= 0:
            return reject("invalid ATR or price", f)

        # ---- hard filters, cheapest and most decisive first ---------------
        if s.symbol in ctx.blocked_symbols:
            return reject(f"no-trade list: {ctx.blocked_symbols[s.symbol]}", f)

        if ctx.btc_regime == BEAR:
            return reject("BTC regime bearish", f)
        if ctx.btc_regime == UNKNOWN:
            return reject("BTC regime unavailable", f)

        if f["htf_regime"] in (BEAR, UNKNOWN):
            return reject(f"higher-timeframe regime {f['htf_regime']}", f)

        if not (f["ema_fast"] > f["ema_slow"] > f["ema_trend"]):
            return reject("EMA structure not aligned", f)
        if f["price"] <= f["ema_fast"]:
            return reject("price below fast EMA", f)

        if f["price"] <= f["donchian_high"]:
            return reject("no Donchian breakout on closed candle", f)

        if f["adx"] < c.min_adx:
            return reject(f"ADX {f['adx']:.1f} < {c.min_adx}", f)
        if f["rsi"] > c.max_rsi:
            return reject(f"RSI {f['rsi']:.1f} exhausted (> {c.max_rsi})", f)
        if f["rel_volume"] < c.min_rel_volume:
            return reject(f"relative volume {f['rel_volume']:.2f} < {c.min_rel_volume}", f)

        extension_atr = (f["price"] - f["donchian_high"]) / f["atr"]
        if extension_atr > c.max_extension_atr:
            return reject(f"over-extended {extension_atr:.2f} ATR above trigger", f)

        # ---- score --------------------------------------------------------
        btc_roc = float(meta.get("btc_roc", float("nan")))
        comps = {
            "htf_trend": scoring.score_htf_trend(f["htf_score"]),
            "breakout": scoring.score_breakout(f["price"], f["donchian_high"], f["atr"]),
            "adx": scoring.score_adx(f["adx"]),
            "momentum": scoring.score_momentum(list(f["momentum"].values())),
            "rel_strength": scoring.score_rel_strength(
                f["momentum"].get(c.momentum_periods[-1], float("nan")), btc_roc),
            "rel_volume": scoring.score_rel_volume(f["rel_volume"]),
            "extension": scoring.score_extension(f["price"], f["donchian_high"],
                                                 f["atr"], c.max_extension_atr),
            "liquidity": scoring.score_liquidity(
                float(meta.get("dollar_volume", 0.0)),
                float(meta.get("spread_bps", 0.0)),
                float(meta.get("min_dollar_volume", 1e6))),
            "market_ctx": scoring.score_market_context(ctx.btc_regime_score,
                                                       ctx.breadth_pct),
        }
        score = scoring.combine(comps)
        news = ctx.news_by_symbol.get(s.symbol)
        score, news_note = scoring.apply_news_penalty(score, news)
        if news_note:
            f["news_note"] = news_note

        stop = initial_stop(f["price"], f["atr"], c.stop_atr_mult)
        if stop <= 0 or stop >= f["price"]:
            return reject("computed stop invalid", f, score)

        f["extension_atr"] = extension_atr
        f["score_components"] = comps
        f["stop_price"] = stop

        if score < c.min_score:
            sig = reject(f"score {score:.1f} < threshold {c.min_score}", f, score)
            sig.components = comps
            return sig

        rets = log_returns(s.close)
        return Signal(symbol=s.symbol, strategy=self.name, version=self.version,
                      candle_id=cid, ts_ms=int(s.open_ms[-1]), ref_price=f["price"],
                      stop_price=stop, atr=f["atr"], score=score, passed=True,
                      components=comps, features=f,
                      returns=rets[-min(len(rets), 336):])

    # -------------------------------------------------------------- exits
    def should_exit(self, pos: Position, s: Series, ctx: MarketContext) -> str | None:
        """Non-stop exits, evaluated on closed candles only. Stop exits are
        handled by the execution engine against the candle's low."""
        c = self.cfg
        if len(s) < c.ema_slow + 5 or not s.is_sane():
            return None
        price = float(s.close[-1])
        e_fast = last_valid(ema(s.close, c.ema_fast))
        e_slow = last_valid(ema(s.close, c.ema_slow))

        # trend invalidation: the structural premise of the trade is gone
        if np.isfinite(e_fast) and np.isfinite(e_slow) and e_fast < e_slow and price < e_slow:
            return "trend_invalidation"

        bar = tf_ms(c.entry_timeframe)
        bar_close_ms = int(s.open_ms[-1]) + bar
        if time_stop_hit(pos, bar_close_ms, price, c, bar):
            return "time_stop"
        return None
