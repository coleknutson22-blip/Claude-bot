"""Reads the research journal to answer the questions a forward test asks.

WHAT THIS IS FOR
----------------
Every gate the strategy applies is a hypothesis: that setups below the line are
worse than setups above it. A live run generates the evidence to test each one,
but only if the evidence is actually assembled -- a rejection count alone says
how often a filter fired, never whether it was RIGHT to fire.

So each report here pairs a gate with the counterfactual outcome of what it
rejected. "Relative volume rejected 240 setups" is an operations metric.
"Relative volume rejected 240 setups whose average 24h return was +1.9%" is a
finding.

WHAT IT IS NOT
--------------
It does not tune anything, and nothing here feeds back into the strategy. It
reads two tables and prints. Every threshold stays exactly where it was set
until a person looks at these numbers and decides to move it -- and no number
below is worth acting on until its sample is large enough, which is why every
row carries its own `n` and a `sufficient_sample` flag rather than a single
threshold applied globally and forgotten.

A NOTE ON WHAT THE COUNTERFACTUALS MEASURE
------------------------------------------
The stored return is the raw price move from the signal price over a fixed
horizon, in the SIGNAL'S OWN DIRECTION. It is not a trade: no stop, no target,
no fees, no slippage, no financing. A rejected setup showing +2% did not
necessarily survive to collect it -- the same move could have hit a stop first.
Treat these as a ranking signal between filters, never as forgone P&L.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# Which journal feature each gate reads, and which way the gate points. The
# reject reasons are matched as substrings because the message carries the
# measured value too ("relative volume 0.71 < 0.90").
GATES = {
    "rel_volume": {
        "feature": "rel_volume", "match": "relative volume",
        "config": "aggressive.min_rel_volume",
        "question": "is the 0.9 relative-volume floor too strict?",
    },
    "atr_pct": {
        "feature": "atr_pct", "match": "atr",
        "config": "aggressive.min_atr_pct",
        "question": "is the 0.25% ATR floor too strict?",
    },
    "setup_score": {
        "feature": None, "match": "setup score",
        "config": "aggressive.min_setup_score",
        "question": "is a minimum setup score of 50 too strict?",
    },
    "ema_struct_15m": {
        "feature": "ema_struct_15m", "match": "15m structure",
        "config": "aggressive.min_ema_struct_15m",
        "question": "is the 15m structure requirement too strict?",
    },
}


@dataclass
class Row:
    bucket: str
    n: int = 0
    wins: int = 0
    net: float = 0.0
    avg: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.wins / self.n * 100.0 if self.n else 0.0


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def normalise_reason(reason: str) -> str:
    """Collapse a rejection message to the RULE it names, dropping the numbers.

    Reasons carry their measurement -- "relative volume 0.71 < 0.90" -- which
    is exactly right in a log and useless as a group key, because every
    rejection becomes its own bucket. Stripping digits character by character
    leaves debris ("setup score . < ."), so each numeric run is replaced whole
    and the leftover comparison punctuation goes with it.

    Timeframe tokens are held back first: the 15 in "15m structure" names the
    rule, not a reading of it, and dropping it turns the gate into "m
    structure" -- which no longer matches the config key it refers to.
    """
    text = (reason or "").split("(")[0].strip().lower()
    # The placeholder must itself contain no digits, or the strip below eats
    # the very thing it is protecting.
    held: list[str] = []

    def _hold(m: re.Match) -> str:
        held.append(m.group(0))
        return f"\x00{'z' * len(held)}\x00"

    text = re.sub(r"\b\d+[mhd]\b", _hold, text)
    text = re.sub(r"-?\d+(?:\.\d+)?%?", "", text)
    for i, original in enumerate(held, 1):
        text = text.replace(f"\x00{'z' * i}\x00", original)
    text = re.sub(r"[<>=+\-/]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" .,:;")
    return (text or "unspecified")[:60]


def _bucket(v, width: float) -> str | None:
    f = _num(v)
    if f is None:
        return None
    lo = math.floor(f / width) * width
    return f"{lo:g}-{lo + width:g}"


class ForwardTestReport:
    """One strategy's journal, sliced the ways a forward test needs."""

    def __init__(self, repo, strategy: str, min_sample: int = 20) -> None:
        self.repo = repo
        self.strategy = strategy
        self.min_sample = min_sample
        self.obs = repo.get_observations(strategy=strategy)
        self.trades = repo.get_trades(strategy)
        self._cf = None

    # ------------------------------------------------------------ helpers
    def counterfactuals(self) -> dict[str, list[dict]]:
        """Hypothetical returns keyed by observation id."""
        if self._cf is None:
            out: dict[str, list[dict]] = {}
            for r in self.repo.conn.execute(
                    """SELECT observation_id, horizon_h, return_pct
                       FROM counterfactuals WHERE return_pct IS NOT NULL"""):
                out.setdefault(r["observation_id"], []).append(dict(r))
            self._cf = out
        return self._cf

    def _signed_returns(self, rows: list[dict],
                        horizon_h: int | None) -> list[float]:
        """Counterfactual moves, signed by the SIGNAL'S OWN DIRECTION.

        A rejected SHORT was right when price fell, so its raw -3% is a +3%
        result for that signal. Averaging the raw moves instead would make a
        directional filter look wrong precisely when it was working.
        """
        cf, out = self.counterfactuals(), []
        for o in rows:
            sign = -1.0 if (o.get("side") or "long") == "short" else 1.0
            for c in cf.get(o["id"], []):
                if horizon_h is None or c["horizon_h"] == horizon_h:
                    out.append(float(c["return_pct"]) * sign)
        return out

    # ------------------------------------------------------------ sections
    def decisions(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for o in self.obs:
            counts[o["decision"]] = counts.get(o["decision"], 0) + 1
        return counts

    def by_side(self) -> dict[str, dict]:
        """Evaluated / entered / rejected split by direction.

        A strategy that is nominally long-and-short but takes 40 longs and 2
        shorts has a finding in it, and this is where it shows up first.
        """
        out = {}
        for side in ("long", "short"):
            rows = [o for o in self.obs if (o["side"] or "long") == side]
            entered = [o for o in rows if o["decision"] == "ENTERED"]
            trades = [t for t in self.trades if t["side"] == side]
            out[side] = {
                "evaluated": len(rows), "entered": len(entered),
                "rejected": len(rows) - len(entered),
                "closed_trades": len(trades),
                "net_pnl": sum(t["net_pnl"] for t in trades),
                "financing": sum(t["financing"] for t in trades),
            }
        return out

    def rejection_counts(self) -> list[Row]:
        """Every distinct rejection reason, most frequent first."""
        groups: dict[str, list[dict]] = {}
        for o in self.obs:
            if o["decision"] == "ENTERED" or not o["reject_reason"]:
                continue
            key = normalise_reason(o["reject_reason"])
            groups.setdefault(key, []).append(o)
        rows = []
        for k, obs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            rets = self._signed_returns(obs, None)
            rows.append(Row(bucket=k, n=len(obs),
                            avg=sum(rets) / len(rets) if rets else 0.0,
                            extra={"with_outcome": len(rets)}))
        return rows

    def score_buckets(self, width: float = 10.0) -> list[Row]:
        """Setup score vs what happened next -- for taken AND rejected setups.

        This is the only view that can say whether the score means anything at
        all: if the 40-50 bucket's rejected setups moved as well as the 80-90
        bucket's, the score is not measuring what it claims to.
        """
        groups: dict[str, list[dict]] = {}
        for o in self.obs:
            b = _bucket(o["score"], width)
            if b:
                groups.setdefault(b, []).append(o)
        rows = []
        for b, obs in sorted(groups.items(), key=lambda kv: _num(kv[0].split("-")[0])):
            rets = self._signed_returns(obs, None)
            entered = sum(1 for o in obs if o["decision"] == "ENTERED")
            rows.append(Row(
                bucket=b, n=len(obs),
                avg=sum(rets) / len(rets) if rets else 0.0,
                wins=sum(1 for r in rets if r > 0),
                extra={"entered": entered, "with_outcome": len(rets)}))
        return rows

    def gate_sensitivity(self) -> list[dict]:
        """For each gate: what it rejected, and how those setups then moved.

        The `near_miss` column is the part worth reading -- setups that failed
        the gate by a little. If those behave like the ones that passed, the
        line is in the wrong place. If they behave like the ones that failed
        badly, it is doing its job.
        """
        out = []
        for name, spec in GATES.items():
            hit = [o for o in self.obs
                   if o["reject_reason"]
                   and spec["match"] in o["reject_reason"].lower()]
            rets = self._signed_returns(hit, None)
            vals = []
            if spec["feature"]:
                vals = [v for v in
                        (_num(o["features"].get(spec["feature"])) for o in hit)
                        if v is not None]
            else:
                vals = [v for v in (_num(o["score"]) for o in hit)
                        if v is not None]
            out.append({
                "gate": name,
                "config_key": spec["config"],
                "question": spec["question"],
                "rejected": len(hit),
                "with_outcome": len(rets),
                "avg_return_pct": sum(rets) / len(rets) if rets else 0.0,
                "win_rate_pct": (sum(1 for r in rets if r > 0) / len(rets) * 100.0
                                 if rets else 0.0),
                "measured_min": min(vals) if vals else None,
                "measured_median": (sorted(vals)[len(vals) // 2] if vals else None),
                "measured_max": max(vals) if vals else None,
                "sufficient_sample": len(rets) >= self.min_sample,
            })
        return sorted(out, key=lambda r: -r["rejected"])

    def confidence_buckets(self) -> list[Row]:
        """Realised results per confidence bucket -- the calibration question.

        Phase 1 maps setup score to confidence with the identity and sizes off
        the result. That is a HYPOTHESIS. These rows are what eventually
        confirms or kills it: whether 60-69 loses money, and whether 90+
        actually outperforms.
        """
        groups: dict[str, list[dict]] = {}
        for t in self.trades:
            b = t["journal"].get("conf_bucket")
            if b:
                groups.setdefault(str(b), []).append(t)
        rows = []
        for b, ts in sorted(groups.items()):
            net = sum(t["net_pnl"] for t in ts)
            rows.append(Row(bucket=b, n=len(ts),
                            wins=sum(1 for t in ts if t["net_pnl"] > 0),
                            net=net, avg=net / len(ts),
                            extra={"avg_notional": sum(
                                t["journal"].get("final_notional", 0.0) or 0.0
                                for t in ts) / len(ts)}))
        return rows

    def sufficient(self, n: int) -> bool:
        return n >= self.min_sample
