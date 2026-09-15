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


# ===================================================== forward excursion paths
class ExcursionReport:
    """Answers the take-profit question from recorded paths, or says it cannot.

    Every rate below is computed over UNAMBIGUOUS outcomes only. A path whose
    stop and target fell in the same 5m candle is counted in `ambiguous` and
    excluded from both the numerator and the denominator, because a comparison
    that resolved those by assumption would be measuring the assumption.
    """

    def __init__(self, repo, strategy: str, cfg=None, min_sample: int = 30) -> None:
        from . import excursion as X
        self.X = X
        self.repo = repo
        self.strategy = strategy
        self.min_sample = min_sample
        self.paths = repo.get_excursions(strategy)
        self.obs = {o["id"]: o for o in repo.get_observations(strategy=strategy)}
        # The crossover where a 2R target and a fixed +2% coincide. Derived from
        # config so it tracks the shipped parameters rather than a stale note.
        self.target_r = getattr(cfg, "target_r", 2.0) if cfg else 2.0
        self.stop_mult = getattr(cfg, "stop_atr_mult", 1.8) if cfg else 1.8
        self.crossover_atr = 2.0 / (self.target_r * self.stop_mult)

    # ------------------------------------------------------------- helpers
    def atr_pct(self, path) -> float | None:
        """Recorded ATR% if the journal has it; else implied by the stop."""
        o = self.obs.get(path.observation_id)
        if o:
            v = _num((o.get("features") or {}).get("atr_pct"))
            if v is not None:
                return v
        return (path.stop_distance_pct / self.stop_mult
                if self.stop_mult else None)

    def score(self, path) -> float | None:
        o = self.obs.get(path.observation_id)
        return _num(o.get("score")) if o else None

    def regime(self, path) -> str:
        o = self.obs.get(path.observation_id)
        return str((o.get("features") or {}).get("btc_regime") or "unknown") \
            if o else "unknown"

    def complete(self) -> list:
        """Only finished paths. An open path has not had its chance yet, and
        counting it as a miss would bias every rate downward."""
        return [p for p in self.paths if p.status == self.X.COMPLETE]

    # -------------------------------------------------------- 3. hit rates
    def hit_rates(self, paths=None) -> list[dict]:
        rows = []
        paths = self.complete() if paths is None else paths
        for label in [f"pct_{p}" for p in self.X.PCT_TARGETS] + \
                     [f"r_{r}" for r in self.X.R_TARGETS]:
            hit = sum(1 for p in paths if p.reached(label))
            amb = sum(1 for p in paths if p.ambiguous(label))
            decided = len(paths) - amb
            rows.append({
                "target": label, "n": len(paths), "decided": decided,
                "hit": hit, "ambiguous": amb,
                "hit_rate_pct": hit / decided * 100.0 if decided else 0.0,
                "sufficient_sample": decided >= self.min_sample,
            })
        return rows

    # ------------------------------ 1 + 2. the two crossover questions
    def stalled_between_2pct_and_2r(self) -> dict:
        """Q1: above the crossover, how often did +2% land but 2R never?

        This is the money-on-the-table case: with a demanding 2R the trade gave
        back a move a fixed +2% would have banked.
        """
        hi = [p for p in self.complete()
              if (self.atr_pct(p) or 0) > self.crossover_atr]
        amb = [p for p in hi if p.ambiguous("pct_2.0") or p.ambiguous("r_2.0")]
        decided = [p for p in hi if p not in amb]
        stalled = [p for p in decided
                   if p.reached("pct_2.0") and not p.reached("r_2.0")]
        both = [p for p in decided if p.reached("r_2.0")]
        return {
            "question": "reached +2.0% but never 2R, before the stop",
            "atr_filter": f"atr_pct > {self.crossover_atr:.4f}%",
            "n": len(hi), "decided": len(decided), "ambiguous": len(amb),
            "stalled": len(stalled), "reached_2r": len(both),
            "stalled_pct": len(stalled) / len(decided) * 100.0 if decided else 0.0,
            "sufficient_sample": len(decided) >= self.min_sample,
        }

    def two_r_cheaper_than_2pct(self) -> dict:
        """Q2: below the crossover, how often does 2R fire where +2% would not?

        Here the current rule is the LESS demanding of the two, and a fixed
        +2% would have been the one leaving trades unclosed.
        """
        lo = [p for p in self.complete()
              if (self.atr_pct(p) or 0) < self.crossover_atr]
        amb = [p for p in lo if p.ambiguous("pct_2.0") or p.ambiguous("r_2.0")]
        decided = [p for p in lo if p not in amb]
        better = [p for p in decided
                  if p.reached("r_2.0") and not p.reached("pct_2.0")]
        return {
            "question": "2R reached where a fixed +2% was never reached",
            "atr_filter": f"atr_pct < {self.crossover_atr:.4f}%",
            "n": len(lo), "decided": len(decided), "ambiguous": len(amb),
            "two_r_only": len(better),
            "two_r_only_pct": len(better) / len(decided) * 100.0 if decided else 0.0,
            "sufficient_sample": len(decided) >= self.min_sample,
        }

    # ----------------------------------------------------- 4. breakdowns
    def _group(self, keyfn) -> dict[str, list]:
        out: dict[str, list] = {}
        for p in self.complete():
            try:
                k = keyfn(p)
            except Exception:
                k = None
            if k is not None:
                out.setdefault(str(k), []).append(p)
        return out

    def breakdowns(self) -> dict[str, dict]:
        conf_edges = [(0, 60), (60, 70), (70, 80), (80, 90), (90, 95), (95, 1e9)]

        def conf_bucket(p):
            s = self.score(p)
            if s is None:
                return None
            for lo, hi in conf_edges:
                if lo <= s < hi:
                    return f"{lo:.0f}-{hi:.0f}" if hi < 1e9 else f"{lo:.0f}+"
            return None

        return {
            "side": self._group(lambda p: p.side),
            "score_bucket": self._group(lambda p: _bucket(self.score(p), 10)),
            "confidence_bucket": self._group(conf_bucket),
            "atr_bucket": self._group(lambda p: _bucket(self.atr_pct(p), 0.25)),
            "btc_regime": self._group(self.regime),
        }

    # -------------------------------------------------------- excursions
    def excursion_stats(self, paths=None) -> dict:
        import statistics as st
        paths = self.complete() if paths is None else paths
        if not paths:
            return {"n": 0}
        mfe = [p.mfe_pct for p in paths]
        mae = [p.mae_pct for p in paths]
        t_mfe = [v for v in (p.minutes_to_mfe() for p in paths) if v is not None]
        t_mae = [v for v in (p.minutes_to_mae() for p in paths) if v is not None]
        stopped = [p for p in paths if p.stop_touched]
        t_stop = [v for v in (p.minutes_to_stop() for p in stopped) if v is not None]
        return {
            "n": len(paths),
            "mfe_mean": st.mean(mfe), "mfe_median": st.median(mfe),
            "mfe_max": max(mfe),
            "mae_mean": st.mean(mae), "mae_median": st.median(mae),
            "mae_min": min(mae),
            "minutes_to_mfe_median": st.median(t_mfe) if t_mfe else None,
            "minutes_to_mae_median": st.median(t_mae) if t_mae else None,
            "stopped": len(stopped),
            "stopped_pct": len(stopped) / len(paths) * 100.0,
            "minutes_to_stop_median": st.median(t_stop) if t_stop else None,
            "sufficient_sample": len(paths) >= self.min_sample,
        }

    def status_counts(self) -> dict:
        return self.repo.excursion_counts()

    def ambiguity_share(self) -> dict:
        """How much of the record the OHLC resolution cannot settle.

        Reported prominently: if this is large, the honest fix is finer data,
        not a bolder assumption about intrabar ordering.
        """
        done = self.complete()
        if not done:
            return {"paths": 0, "any_ambiguous": 0, "share_pct": 0.0}
        amb = sum(1 for p in done
                  if any(v == self.X.AMBIGUOUS for v in p.touches.values()))
        return {"paths": len(done), "any_ambiguous": amb,
                "share_pct": amb / len(done) * 100.0}
