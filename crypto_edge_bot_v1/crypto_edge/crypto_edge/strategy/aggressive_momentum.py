"""Strategy B: fast multi-timeframe momentum, LONG or SHORT.

HOW THIS DIFFERS FROM STRATEGY A, AND WHY
-----------------------------------------
`trend_breakout` asks one binary question -- has price closed beyond a 48-bar
Donchian channel with trend, ADX and volume all confirming? Every condition is a
veto, so a market that is moving decisively but has not printed a textbook
breakout produces nothing. That is the correct design for a slow trend
strategy and the reason it trades rarely.

This strategy asks a graded question instead: is there enough AGREEMENT across
fast timeframes to justify a directional trade right now? Momentum on several
windows, constructive 15m structure, a 1h structure that is not actively
hostile, a market regime that is not against it, and a move that has not already
finished. Individual conditions contribute to a score rather than each holding a
veto, which is what makes it willing to trade materially more often without
lowering the bar on the things that actually protect the account -- data
sanity, liquidity, exhaustion and regime hostility remain hard gates.

WHAT IT DOES NOT DO
-------------------
Stage 2 is signal generation. This module chooses a side and scores a setup. It
never sizes a position, never touches an account, never consults a model, and
has no access to the clock beyond the timestamps on the closed candles it is
given. Sizing, the capital ladder and confidence gating are Stage 3.

`setup_score` IS NOT A PROBABILITY. It is a weighted sum of hand-chosen factors,
on the same footing as Strategy A's score: 70 means "better on the factors we
believe matter than a 60", not "wins 70% of the time". Calibrating it against
realised outcomes is what the recorded components exist for.
"""
from __future__ import annotations

import numpy as np

from ..config import AggressiveCfg
from ..indicators import log_returns
from ..models import Position, Series
from ..timeutils import candle_id
from . import features as feat
from .base import LONG, NO_TRADE, SHORT, MarketContext, Signal
from .regime import BEAR, BULL, UNKNOWN

NAN = float("nan")

# component -> (weight, why it earns its place)
WEIGHTS: dict[str, float] = {
    "momentum": 0.24,
    "structure": 0.20,
    "trend_quality": 0.14,
    "rel_strength": 0.14,
    "participation": 0.10,
    "volatility": 0.08,
    "room": 0.06,
    "market_ctx": 0.04,
}

JUSTIFICATIONS = {
    "momentum": "Agreement across 30m/1h/2h/3h/6h in ATR units; persistence "
                "across horizons is the core premise.",
    "structure": "EMA 9/21/50 stack on 5m and 15m; the trade needs the fast "
                 "frames on its side to be timed at all.",
    "trend_quality": "R-squared of the log-price fit; an orderly move is more "
                     "tradable than an equally large spike.",
    "rel_strength": "Divergence from BTC over 1h and 6h; leaders keep leading "
                    "and laggards keep lagging more often than not.",
    "participation": "Relative volume; a move without volume is a thin print.",
    "volatility": "ATR expansion; a move starting from a volatility squeeze "
                  "has more room than one at the end of an expansion.",
    "room": "Distance still available to the swing level in the trade's "
            "favour; late entries have worse reward-to-risk.",
    "market_ctx": "BTC regime and breadth as a tilt -- the hard vetoes are "
                  "handled separately, so this is deliberately small.",
}

# Momentum windows that vote on direction. 24h is deliberately excluded: this
# is an intraday strategy, and a day-old move should not outvote the last hour.
VOTING_WINDOWS = ("30m", "1h", "2h", "3h", "6h")


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(min(hi, max(lo, x)))


def _scale(value: float, lo: float, hi: float) -> float:
    if not np.isfinite(value) or hi <= lo:
        return 0.0
    return _clamp((value - lo) / (hi - lo) * 100.0)


def momentum_votes(momentum_atr: dict[str, float], direction: int,
                   min_atr: float = 0.0) -> int:
    """How many voting windows point the way the trade wants to go.

    A window must move at least `min_atr` of the asset's own ATR to count. Sign
    alone is not evidence: in a flat market every window still has one, three of
    five agree by chance roughly a third of the time, and the strategy would
    read pure chop as a directional setup.
    """
    n = 0
    for w in VOTING_WINDOWS:
        v = momentum_atr.get(w, NAN)
        if np.isfinite(v) and v * direction >= max(min_atr, 1e-12):
            n += 1
    return n


def _signed_mean(values: dict[str, float], keys, direction: int) -> float:
    vals = [values.get(k, NAN) * direction for k in keys
            if np.isfinite(values.get(k, NAN))]
    return float(sum(vals) / len(vals)) if vals else NAN


class AggressiveMomentumStrategy:
    """Deterministic. Same closed candles in, same Signal out, every time."""

    name = "aggressive_momentum_v2"

    def __init__(self, cfg: AggressiveCfg) -> None:
        self.cfg = cfg
        self.version = cfg.version

    # ------------------------------------------------------------ scoring
    def score_components(self, f: dict, direction: int) -> dict[str, float]:
        """Every component in [0, 100], measured FROM THE TRADE'S POINT OF VIEW.

        Multiplying by direction is what makes the short path a genuine mirror
        rather than a second implementation: a -3% move scores for a short
        exactly as a +3% move scores for a long.
        """
        d = direction
        m_atr = f.get("momentum_atr", {})
        mom = _signed_mean(m_atr, VOTING_WINDOWS, d)
        struct5 = f.get("ema_struct_5m", NAN)
        struct15 = f.get("ema_struct_15m", NAN)
        structs = [v * d for v in (struct5, struct15) if np.isfinite(v)]
        rs = f.get("rel_strength", {})
        rel = _signed_mean(rs, ("1h", "6h"), d)
        r2s = [v for v in (f.get("trend_r2_15m", NAN), f.get("trend_r2_1h", NAN))
               if np.isfinite(v)]
        # Room to the level the trade is heading TOWARDS.
        room = (f.get("dist_to_swing_high_atr", NAN) if d > 0
                else f.get("dist_to_swing_low_atr", NAN))
        breadth = f.get("breadth_pct", 50.0)
        ctx_val = breadth if d > 0 else 100.0 - breadth

        return {
            "momentum": _scale(mom, 0.0, 2.5),
            "structure": _scale(sum(structs) / len(structs) if structs else NAN,
                                -1.0, 1.0),
            "trend_quality": _scale(sum(r2s) / len(r2s) if r2s else NAN, 0.05, 0.7),
            "rel_strength": _scale(rel, -2.0, 6.0),
            "participation": _scale(f.get("rel_volume", NAN), 0.8, 2.5),
            "volatility": _scale(f.get("atr_expansion", NAN), 0.7, 2.0),
            "room": _scale(room, 0.2, 3.0),
            "market_ctx": _scale(ctx_val, 20.0, 80.0),
        }

    @staticmethod
    def combine(components: dict[str, float]) -> float:
        total = sum(WEIGHTS.values())
        return _clamp(sum(components.get(k, 0.0) * w
                          for k, w in WEIGHTS.items()) / total)

    # ------------------------------------------------------------ direction
    def _blockers(self, f: dict, side: str) -> list[str]:
        """Every condition standing between this symbol and a trade on `side`.

        Returned as a list rather than a first-failure string so a rejection can
        say "momentum agreement 2/5 < 3; 1h structure -1.00 hostile" instead of
        naming whichever check happened to be written first.
        """
        c = self.cfg
        d = 1 if side == LONG else -1
        m_atr = f.get("momentum_atr", {})
        struct15 = f.get("ema_struct_15m", NAN)
        struct1h = f.get("ema_struct_1h", NAN)
        out: list[str] = []

        votes = momentum_votes(m_atr, d, c.min_vote_atr)
        if votes < c.min_momentum_agree:
            out.append(f"momentum agreement {votes}/{len(VOTING_WINDOWS)} "
                       f"< {c.min_momentum_agree}")
        if not np.isfinite(struct15):
            out.append("15m structure unavailable")
        elif struct15 * d < c.min_ema_struct_15m:
            want = "bullish" if d > 0 else "bearish"
            out.append(f"15m structure {struct15:+.2f} not {want} enough "
                       f"(need {c.min_ema_struct_15m * d:+.2f})")
        if np.isfinite(struct1h) and struct1h * d <= c.max_hostile_ema_1h:
            out.append(f"1h structure {struct1h:+.2f} hostile to a {side}")
        return out

    def choose_side(self, f: dict, ctx: MarketContext) -> tuple[str, str]:
        """Pick a side, or explain why neither is available.

        Returns (side, reason). The reason is recorded on every NO_TRADE so the
        research database can show WHICH condition is doing the filtering rather
        than only that something did.
        """
        c = self.cfg
        m_atr = f.get("momentum_atr", {})
        struct15 = f.get("ema_struct_15m", NAN)
        struct1h = f.get("ema_struct_1h", NAN)

        if not np.isfinite(struct15):
            return NO_TRADE, "15m EMA structure unavailable"

        blockers = {LONG: self._blockers(f, LONG), SHORT: self._blockers(f, SHORT)}
        long_ok, short_ok = not blockers[LONG], not blockers[SHORT]

        if long_ok and short_ok:            # cannot happen, but never guess
            return NO_TRADE, "contradictory long and short conditions"
        if not long_ok and not short_ok:
            # Report the side that came CLOSEST, and name the conditions that
            # actually blocked it. Reporting one fixed condition made a 1h-
            # hostility rejection read as a 15m structure problem, which sent
            # calibration after the wrong threshold.
            near = min((LONG, SHORT), key=lambda sd: (len(blockers[sd]), sd))
            return NO_TRADE, f"no {near}: " + "; ".join(blockers[near])

        side = LONG if long_ok else SHORT
        d = 1 if side == LONG else -1

        # --- regime vetoes -------------------------------------------------
        if side == LONG and c.veto_longs_in_btc_bear and ctx.btc_regime == BEAR:
            return NO_TRADE, "BTC regime bearish -- long vetoed"
        if side == SHORT and c.veto_shorts_in_btc_bull and ctx.btc_regime == BULL:
            return NO_TRADE, "BTC regime bullish -- short vetoed"
        if ctx.btc_regime == UNKNOWN:
            return NO_TRADE, "BTC regime unavailable"
        breadth = f.get("breadth_pct", 50.0)
        if side == LONG and breadth < c.min_breadth_for_long:
            return NO_TRADE, f"breadth {breadth:.0f}% too weak for a long"
        if side == SHORT and breadth > c.max_breadth_for_short:
            return NO_TRADE, f"breadth {breadth:.0f}% too strong for a short"

        # --- exhaustion ----------------------------------------------------
        move_1h = f.get("momentum_atr", {}).get("1h", NAN)
        if np.isfinite(move_1h) and move_1h * d > c.max_move_atr_1h:
            return NO_TRADE, (f"already moved {abs(move_1h):.1f} ATR in 1h "
                              f"(> {c.max_move_atr_1h})")
        # Past the level it was heading for, by more than the allowance.
        behind = (f.get("dist_to_swing_high_atr", NAN) if d > 0
                  else f.get("dist_to_swing_low_atr", NAN))
        if np.isfinite(behind) and behind < -c.max_extension_atr:
            return NO_TRADE, (f"over-extended {abs(behind):.1f} ATR beyond the "
                              f"swing level")
        return side, ""

    # ------------------------------------------------------------ evaluate
    def evaluate_frames(self, frames: dict[str, Series], ctx: MarketContext,
                        btc_frames: dict[str, Series] | None = None,
                        meta: dict | None = None) -> Signal:
        c = self.cfg
        meta = meta or {}
        anchor = frames.get("15m") or frames.get("5m") or frames.get("1h")
        symbol = anchor.symbol if anchor is not None else str(meta.get("symbol", "?"))
        ts = int(anchor.open_ms[-1]) if anchor is not None and len(anchor) else ctx.ts_ms
        cid = candle_id(symbol, "15m", ts) if anchor is not None else ""

        def out(side: str, score: float, passed: bool, reason: str,
                f: dict | None = None, comps: dict | None = None,
                stop: float = 0.0, returns=None) -> Signal:
            sig = Signal(
                symbol=symbol, strategy=self.name, version=self.version,
                candle_id=cid, ts_ms=ts,
                ref_price=float(f.get("price", 0.0)) if f else 0.0,
                stop_price=stop, atr=float(f.get("atr", 0.0)) if f else 0.0,
                score=score, passed=passed, reject_reason=reason, side=side,
                components=comps or {}, features=f or {}, returns=returns)
            return sig

        # ---- data sufficiency: fail closed --------------------------------
        need = feat.required_bars()
        for tf in c.timeframes:
            s = frames.get(tf)
            if s is None or len(s) == 0:
                return out(NO_TRADE, 0.0, False, f"no {tf} candles")
            if len(s) < need.get(tf, 0):
                return out(NO_TRADE, 0.0, False,
                           f"insufficient {tf} history ({len(s)} < {need[tf]} bars)")
            if not s.is_sane():
                return out(NO_TRADE, 0.0, False, f"{tf} candles failed sanity check")

        f = feat.compute(frames, btc=btc_frames,
                         breadth_pct=ctx.breadth_pct, btc_regime=ctx.btc_regime,
                         btc_regime_score=ctx.btc_regime_score, meta=meta,
                         rel_volume_bars=c.rel_volume_bars)

        required = ("price", "atr", "atr_pct", "ema_struct_15m", "rel_volume")
        missing = [k for k in required if not np.isfinite(f.get(k, NAN))]
        if missing:
            return out(NO_TRADE, 0.0, False,
                       f"indicator not ready: {','.join(missing)}", f)
        if f["price"] <= 0 or f["atr"] <= 0:
            return out(NO_TRADE, 0.0, False, "invalid ATR or price", f)
        spread = f.get("frame_price_spread_pct", 0.0)
        if spread > c.max_frame_price_spread_pct:
            return out(NO_TRADE, 0.0, False,
                       f"timeframes disagree on price by {spread:.1f}% "
                       f"(> {c.max_frame_price_spread_pct}%) -- not one market", f)

        if symbol in ctx.blocked_symbols:
            return out(NO_TRADE, 0.0, False,
                       f"no-trade list: {ctx.blocked_symbols[symbol]}", f)
        if f["atr_pct"] < c.min_atr_pct:
            return out(NO_TRADE, 0.0, False,
                       f"ATR {f['atr_pct']:.2f}% < {c.min_atr_pct}%", f)
        if np.isfinite(f["rel_volume"]) and f["rel_volume"] < c.min_rel_volume:
            return out(NO_TRADE, 0.0, False,
                       f"relative volume {f['rel_volume']:.2f} < {c.min_rel_volume}", f)

        # ---- direction ----------------------------------------------------
        side, why = self.choose_side(f, ctx)
        if side == NO_TRADE:
            return out(NO_TRADE, 0.0, False, why, f)

        d = 1 if side == LONG else -1
        comps = self.score_components(f, d)
        score = self.combine(comps)
        f["score_components"] = comps
        f["side"] = side
        f["momentum_votes"] = momentum_votes(f.get("momentum_atr", {}), d,
                                             c.min_vote_atr)

        r2 = max([v for v in (f.get("trend_r2_15m", NAN), f.get("trend_r2_1h", NAN))
                  if np.isfinite(v)] or [NAN])
        if np.isfinite(r2) and r2 < c.min_trend_r2:
            return out(side, score, False,
                       f"trend quality R2 {r2:.2f} < {c.min_trend_r2}", f, comps)

        # Stop sits on the losing side: below a long, ABOVE a short.
        stop = f["price"] - d * c.stop_atr_mult * f["atr"]
        f["stop_price"] = stop
        if stop <= 0 or (d > 0 and stop >= f["price"]) or (d < 0 and stop <= f["price"]):
            return out(side, score, False, "computed stop invalid", f, comps)

        if score < c.min_setup_score:
            # Still a directional READING, just not a tradable one. Recorded
            # with its side so the research database can later answer whether
            # the threshold is in the right place.
            return out(side, score, False,
                       f"setup score {score:.1f} < {c.min_setup_score}", f, comps,
                       stop=stop)

        anchor15 = frames["15m"]
        rets = log_returns(anchor15.close)
        return out(side, score, True, "", f, comps, stop=stop,
                   returns=rets[-min(len(rets), 336):])

    # --- Strategy protocol adapter ---------------------------------------
    def evaluate(self, s: Series, htf: Series | None, ctx: MarketContext,
                 meta: dict | None = None) -> Signal:
        """Protocol-shaped entry point: (entry series, higher timeframe).

        Maps onto the frame dict so this strategy satisfies the same interface
        as Strategy A, for the backtester and anything else that speaks it.
        """
        frames = {"15m": s, "1h": htf} if htf is not None else {"15m": s}
        frames.setdefault("5m", s)
        return self.evaluate_frames(frames, ctx, meta=meta)

    # -------------------------------------------------------------- exits
    def should_exit(self, pos: Position, s: Series, ctx: MarketContext) -> str | None:
        """Momentum invalidation only. Stops, targets and the time stop are
        Stage 3; the protective stop is the engine's job against the candle."""
        if len(s) < 55 or not s.is_sane():
            return None
        struct = feat.ema_structure(s.close)
        if not np.isfinite(struct):
            return None
        if struct * pos.direction <= -0.5:
            return "momentum_invalidation"
        return None
