"""Rejected-signal counterfactual tracking.

HARD RULE: nothing in this module can open a position. It only computes what
WOULD have happened, stores it flagged `hypothetical=1` in a separate table,
and never contributes to paper-account performance.
"""
from __future__ import annotations

import numpy as np

from ..logging_setup import log_event
from ..models import Series
from ..storage.repo import Repo
from ..timeutils import now_ms


class CounterfactualTracker:
    def __init__(self, repo: Repo, horizons_h: list[int]) -> None:
        self.repo = repo
        self.horizons_h = horizons_h

    def evaluate_pending(self, series_lookup) -> int:
        """`series_lookup(symbol) -> Series | None` supplies recent 1h candles.

        For each rejected observation whose horizon has elapsed, find the close
        at (signal time + horizon) and record the hypothetical return.
        """
        pending = self.repo.pending_counterfactuals(self.horizons_h, now_ms())
        done = 0
        for row in pending:
            s: Series | None = series_lookup(row["symbol"])
            if s is None or len(s) == 0:
                continue
            target_ms = int(row["ts_ms"]) + int(row["horizon_h"]) * 3_600_000
            idx = int(np.searchsorted(s.open_ms, target_ms, side="right")) - 1
            if idx < 0 or idx >= len(s):
                continue
            if int(s.open_ms[idx]) < target_ms - 4 * 3_600_000:
                continue    # too far from the horizon to be meaningful
            entry_ref = float(row["price"])
            price_at = float(s.close[idx])
            ret = (price_at / entry_ref - 1.0) * 100.0 if entry_ref > 0 else None
            self.repo.add_counterfactual(row["id"], int(row["horizon_h"]),
                                         entry_ref, price_at, ret, now_ms())
            done += 1
        if done:
            log_event("performance", "INFO",
                      "counterfactuals evaluated (HYPOTHETICAL / NOT EXECUTED)",
                      count=done)
        return done

    def filter_report(self, min_sample: int = 20) -> list[dict]:
        """Per-rejection-reason hypothetical outcomes. Suppressed below
        `min_sample` so we never read signal into noise."""
        rows = self.repo.conn.execute(
            """SELECT o.reject_reason AS reason, c.horizon_h AS h,
                      COUNT(*) AS n, AVG(c.return_pct) AS avg_ret
               FROM counterfactuals c JOIN observations o
                 ON o.id = c.observation_id
               WHERE c.return_pct IS NOT NULL AND o.reject_reason != ''
               GROUP BY o.reject_reason, c.horizon_h
               ORDER BY n DESC""").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["sufficient_sample"] = d["n"] >= min_sample
            d["label"] = "HYPOTHETICAL / NOT EXECUTED"
            out.append(d)
        return out
