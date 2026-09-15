"""Drives forward-excursion paths from the live cycle. Research only.

WHERE THIS SITS
---------------
Downstream of everything. It is called AFTER the strategy has produced its
signals and after the runtime has decided what to do about them, it reads the
same 5m candles the scan already fetched, and it writes to `excursions` -- a
table no execution path reads. Nothing computed here can reach a sizing,
entry or exit decision, which is what keeps a module that deliberately looks
at "what happened next" from becoming a look-ahead bug.

TWO RULES THAT MAKE A RESTART SAFE
----------------------------------
1. Every path stores the open time of the last bar it folded in, and resumes
   strictly after it. Replaying an overlapping series is a no-op.
2. Paths are keyed on `observation_id`, one row each, written with INSERT OR
   REPLACE. A signal cannot acquire a second path, and a crash between two
   writes loses at most the bars since the previous cycle -- which the next
   cycle re-reads anyway.
"""
from __future__ import annotations

from ..logging_setup import log_event
from ..timeutils import now_ms
from . import excursion as ex


class ExcursionRecorder:
    def __init__(self, repo, strategy: str, max_open: int = 500) -> None:
        self.repo = repo
        self.strategy = strategy
        self.max_open = max_open

    # ------------------------------------------------------------- opening
    def start_from_signal(self, observation_id: str, sig) -> bool:
        """Begin a path for one journalled signal. Idempotent.

        Called for EVERY signal that carries a stop -- entered and rejected
        alike. The rejected ones are the population this whole exercise is
        about: "would it have reached +2%" is only an open question for the
        trades we did not take.

        A signal REJECTED BEFORE ITS STOP WAS COMPUTED (no data, the ATR floor,
        the relative-volume floor) gets no path, because "before the stop" has
        no meaning without one. That coverage boundary is reported rather than
        hidden -- see `coverage()`.
        """
        if not observation_id:
            return False
        ref, stop = float(sig.ref_price or 0.0), float(sig.stop_price or 0.0)
        if ref <= 0 or stop <= 0:
            return False
        if self.repo.get_excursion(observation_id) is not None:
            return False
        path = ex.build(observation_id=observation_id, symbol=sig.symbol,
                        strategy=self.strategy, side=sig.side,
                        direction=sig.direction, ref_price=ref,
                        stop_price=stop, signal_ms=int(sig.ts_ms))
        if path.stop_distance_pct <= 0:
            return False
        self.repo.upsert_excursion(path)
        return True

    # ------------------------------------------------------------- walking
    def advance(self, series_by_symbol: dict, now: int | None = None) -> int:
        """Fold newly closed 5m bars into every open path. Returns rows changed.

        `series_by_symbol` holds ALREADY-CLOSED candles -- the caller drops the
        forming bar before this is reached, exactly as the strategy does.
        """
        now = now if now is not None else now_ms()
        changed = 0
        for path in self.repo.open_excursions(self.strategy, self.max_open):
            s = series_by_symbol.get(path.symbol)
            try:
                if ex.walk(path, s, now):
                    self.repo.upsert_excursion(path)
                    changed += 1
            except Exception as e:      # one bad path must not stop the rest
                log_event("performance", "WARNING", "excursion walk failed",
                          observation_id=path.observation_id, error=str(e))
        return changed

    def counts(self) -> dict:
        return self.repo.excursion_counts()

    def coverage(self) -> dict:
        """How many journalled signals have a path, and how many cannot.

        A silent coverage gap would look exactly like a strategy that never
        produced those signals, so the shortfall is counted explicitly.
        """
        obs = self.repo.get_observations(strategy=self.strategy)
        tracked = {e.observation_id for e in
                   self.repo.get_excursions(self.strategy)}
        no_stop = [o for o in obs if o["id"] not in tracked]
        return {"observations": len(obs), "tracked": len(tracked),
                "untracked_no_stop": len(no_stop),
                "reasons": _top_reasons(no_stop)}


def _top_reasons(rows: list[dict], limit: int = 6) -> dict:
    """Why untracked signals had no stop, most common first."""
    from .forward_test import normalise_reason
    counts: dict[str, int] = {}
    for o in rows:
        k = normalise_reason(o.get("reject_reason") or "")
        counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:limit])
