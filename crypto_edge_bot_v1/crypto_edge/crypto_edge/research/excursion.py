"""Path-based forward excursion for every signal -- taken AND rejected.

WHY THIS EXISTS
---------------
The counterfactual table already stores what a rejected signal was worth at
fixed horizons, but each row is a SINGLE POINT: the close at signal time plus
N hours. A +0.3% reading at 4h is perfectly consistent with a move that went
+2.5% and gave it all back, and telling those two apart is the entire question
behind "is a 2R target leaving money on the table".

So this module walks the 5m path forward bar by bar and records what the price
actually DID: how far it ran in the trade's favour, how far against, when, and
-- for each candidate target -- whether that level was reached BEFORE the
initial stop. That last part is what a point-in-time return can never answer.

CAUSALITY
---------
Bars are folded in strictly forward. A bar only ever enters the walk if its
OPEN time is at or after the signal, and only if it has CLOSED. The recorder
reads market data the trading path has already seen and writes to a table no
trading code reads -- `research/` is downstream of execution by construction,
and nothing here is wired back into a sizing or entry decision.

THE SAME-BAR PROBLEM, AND WHY WE REFUSE TO GUESS
------------------------------------------------
A 5m candle gives open/high/low/close. If the stop AND a target both fall
inside [low, high] of one bar, the candle does not say which came first. Both
orderings are consistent with the same OHLC.

The convenient assumptions -- "stop first, be conservative" or "target first,
be optimistic" -- are both wrong in the same way: they invent an ordering the
data does not contain, and the invented ordering then propagates into exactly
the win-rate comparison this table was built to settle. A policy comparison
built on a guess measures the guess.

So such a bar is recorded as AMBIGUOUS_SAME_BAR and excluded from both sides of
any policy comparison. The share of ambiguous outcomes is itself reported: if
it is large, the answer is finer data, not a bolder assumption.

DIRECTION
---------
Everything is expressed as a signed move in the trade's favour:

    favourable = (price - ref) * direction / ref * 100

so a short that falls 2% scores +2.0, and one set of thresholds serves both
sides. The extreme in the trade's favour is the candle HIGH for a long and the
candle LOW for a short; the adverse extreme is the other one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Per-threshold outcome. Strings because they are written to the database and
# read back by research queries.
NOT_TOUCHED = "not_touched"
BEFORE_STOP = "before_stop"
AFTER_STOP = "after_stop"
AMBIGUOUS = "AMBIGUOUS_SAME_BAR"

# Path lifecycle.
OPEN, COMPLETE = "open", "complete"

# The fixed percentage targets under test, and the R multiples they are being
# compared against. Both are recorded for every signal so the comparison is a
# query rather than a re-run.
PCT_TARGETS = (1.0, 1.5, 2.0, 2.5, 3.0)
R_TARGETS = (1.0, 1.5, 2.0)

# Forward checkpoints, in minutes. Strategy B is intraday, so the resolution is
# deliberately front-loaded: the first eight hours carry the signal, and 24h is
# there to catch the slow reversals.
HORIZONS_MIN = (15, 30, 60, 120, 240, 480, 1440)

MIN_MS = 60_000


def r_target_pct(stop_distance_pct: float, r: float) -> float:
    """What an R multiple costs in percentage terms. 2R = 2 x the stop."""
    return stop_distance_pct * r


@dataclass
class Excursion:
    """One signal's forward path. Serialisable; resumes from its own fields."""
    observation_id: str
    symbol: str
    strategy: str
    side: str
    direction: int
    ref_price: float
    stop_price: float
    stop_distance_pct: float
    signal_ms: int

    # --- accumulated state (every field below is resumable) --------------
    status: str = OPEN
    bars: int = 0
    last_bar_ms: int = 0        # open_ms of the last bar folded in
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    mfe_ms: int = 0             # when the best excursion was set
    mae_ms: int = 0
    stop_touched: int = 0
    stop_touched_ms: int = 0
    # threshold label -> status. Labels are stable strings ("pct_2.0", "r_2.0").
    touches: dict = field(default_factory=dict)
    # minutes-since-signal -> favourable % at that checkpoint
    horizons: dict = field(default_factory=dict)
    # venue rounding rules, so a replay can reproduce the live fill price
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- setup
    @property
    def targets(self) -> dict[str, float]:
        """Every threshold under test, as a favourable percentage move."""
        out = {f"pct_{p}": p for p in PCT_TARGETS}
        for r in R_TARGETS:
            out[f"r_{r}"] = r_target_pct(self.stop_distance_pct, r)
        return out

    def favourable_pct(self, price: float) -> float:
        if self.ref_price <= 0:
            return 0.0
        return (price - self.ref_price) * self.direction / self.ref_price * 100.0

    def ensure_targets(self) -> None:
        for label in self.targets:
            self.touches.setdefault(label, NOT_TOUCHED)

    # ------------------------------------------------------------- the walk
    def apply_bar(self, open_ms: int, high: float, low: float,
                  close: float) -> bool:
        """Fold ONE closed 5m candle in. Returns True if anything changed.

        Idempotent with respect to bars already seen: a bar at or before
        `last_bar_ms` is ignored, so replaying the same series after a restart
        cannot double-count a touch or inflate an excursion.
        """
        if self.status != OPEN:
            return False
        if open_ms <= self.last_bar_ms or open_ms < self.signal_ms:
            return False
        if not all(math.isfinite(v) for v in (high, low, close)):
            return False

        self.ensure_targets()
        d = self.direction
        # The favourable extreme is the high for a long and the LOW for a
        # short. Scoring a short off the high would report every winning short
        # as its worst moment.
        fav_price = high if d > 0 else low
        adv_price = low if d > 0 else high
        fav = self.favourable_pct(fav_price)
        adv = self.favourable_pct(adv_price)

        if fav > self.mfe_pct:
            self.mfe_pct, self.mfe_ms = fav, open_ms
        if adv < self.mae_pct:
            self.mae_pct, self.mae_ms = adv, open_ms

        # Did this bar's range contain the stop? A touch counts at the level,
        # not past it: a stop is an order resting at a price.
        stop_hit = (low <= self.stop_price) if d > 0 else (high >= self.stop_price)
        hit_now = [label for label, need in self.targets.items()
                   if self.touches.get(label) == NOT_TOUCHED and fav >= need]

        if stop_hit and not self.stop_touched:
            self.stop_touched, self.stop_touched_ms = 1, open_ms

        for label in hit_now:
            if not stop_hit:
                # Clean: the level was reached and the stop was nowhere in
                # this bar, so the ordering is certain.
                self.touches[label] = BEFORE_STOP
            elif self.stop_touched_ms < open_ms:
                # The stop was already gone on an EARLIER bar, so this touch
                # is unambiguously after it.
                self.touches[label] = AFTER_STOP
            else:
                self.touches[label] = AMBIGUOUS

        # Levels never reached, on a bar where the stop went, are settled as
        # "after the stop" -- the trade was over before they could happen.
        if stop_hit:
            for label in [l for l, s in self.touches.items() if s == NOT_TOUCHED]:
                self.touches[label] = AFTER_STOP

        self.bars += 1
        self.last_bar_ms = open_ms

        elapsed = (open_ms - self.signal_ms) / MIN_MS
        for h in HORIZONS_MIN:
            if str(h) not in self.horizons and elapsed >= h:
                self.horizons[str(h)] = self.favourable_pct(close)

        # The path ENDS at the stop. Excursions measured past it would describe
        # a trade nobody was still in, and would quietly inflate MFE for the
        # signals that were stopped out -- the ones whose MFE matters most.
        if stop_hit:
            self.status = COMPLETE
        return True

    def maybe_complete(self, now_ms: int) -> bool:
        """A path is done once the stop is gone or the longest horizon passes.

        It is NOT completed early just because every target was reached: MFE is
        still moving, and "how far did it actually run" is one of the questions.
        """
        if self.status != OPEN:
            return False
        longest = max(HORIZONS_MIN) * MIN_MS
        if self.stop_touched or (now_ms - self.signal_ms) >= longest:
            self.status = COMPLETE
            return True
        return False

    # ----------------------------------------------------------- reporting
    def reached(self, label: str) -> bool:
        """Did this level happen BEFORE the stop, unambiguously?"""
        return self.touches.get(label) == BEFORE_STOP

    def ambiguous(self, label: str) -> bool:
        return self.touches.get(label) == AMBIGUOUS

    def target_pct_for(self, label: str) -> float:
        return self.targets.get(label, float("nan"))

    def minutes_to_mfe(self) -> float | None:
        return (self.mfe_ms - self.signal_ms) / MIN_MS if self.mfe_ms else None

    def minutes_to_mae(self) -> float | None:
        return (self.mae_ms - self.signal_ms) / MIN_MS if self.mae_ms else None

    def minutes_to_stop(self) -> float | None:
        return ((self.stop_touched_ms - self.signal_ms) / MIN_MS
                if self.stop_touched_ms else None)


def build(*, observation_id: str, symbol: str, strategy: str, side: str,
          direction: int, ref_price: float, stop_price: float,
          signal_ms: int, meta: dict | None = None) -> Excursion:
    """Start a path from a signal. No market data needed yet.

    The stop distance is derived from the two prices rather than passed in, so
    it can never disagree with the stop the path is actually measured against.
    """
    ref = float(ref_price or 0.0)
    dist = abs(ref - float(stop_price)) / ref * 100.0 if ref > 0 else 0.0
    ex = Excursion(
        observation_id=observation_id, symbol=symbol, strategy=strategy,
        side=side or "long", direction=1 if direction >= 0 else -1,
        ref_price=ref, stop_price=float(stop_price), stop_distance_pct=dist,
        signal_ms=int(signal_ms), meta=dict(meta or {}))
    ex.ensure_targets()
    return ex


def walk(ex: Excursion, series, now_ms: int, tape_ctx=None) -> tuple[bool, list]:
    """Fold every newly closed bar of `series` into `ex`.

    Returns `(changed, tape_bars)`. The tape rows are the bars this call
    actually consumed -- the caller persists them, so a path's aggregates and
    its tape advance together or not at all.
    """
    changed, tape = False, []
    if series is not None and len(series):
        for i in range(len(series)):
            open_ms = int(series.open_ms[i])
            if open_ms <= ex.last_bar_ms or open_ms < ex.signal_ms:
                continue
            if tape_ctx is not None:
                # Built BEFORE apply_bar, because apply_bar may complete the
                # path and the bar that closed it still belongs on the tape.
                tape.append(tape_ctx.bar_at(i, ex.signal_ms))
            changed |= ex.apply_bar(open_ms, float(series.high[i]),
                                    float(series.low[i]), float(series.close[i]))
            if ex.status != OPEN:
                break
    changed |= ex.maybe_complete(now_ms)
    return changed, tape
