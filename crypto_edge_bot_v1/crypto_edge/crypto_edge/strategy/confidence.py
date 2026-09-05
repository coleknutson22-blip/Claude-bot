"""Setup score -> confidence -> allocation multiplier.

TWO FIELDS, ON PURPOSE
----------------------
`setup_score` is what the strategy computed: a weighted sum of hand-chosen
factors. `confidence` is what the sizing logic consumes. In Phase 1 the
transform between them is the IDENTITY, so the two numbers are equal -- but
they are kept as separate fields because they mean different things and will
eventually diverge.

WHY THAT MATTERS MORE THAN IT LOOKS
-----------------------------------
Sizing at 80% because a number reads 82 implies that number is a probability.
It is not. `strategy/scoring.py` says the same thing about Strategy A's score in
its own docstring: "a score of 70 does not mean a 70% win rate; it means better
than a 60 on the factors we believe matter". Nothing has yet shown that an 85
wins more often than a 65 -- that is the experiment the research journal exists
to run, and it needs a few hundred trades per bucket before it can answer.

So Phase 1 ships the mechanism and records everything needed to fit the real
mapping later. When that data exists, `confidence_is_identity` goes false and a
calibrated transform replaces the identity here. Sizing does not change; only
this file does. Until then the buckets are a HYPOTHESIS being tested, not a
measurement being applied, and the journal records `setup_score`, `confidence`
and `conf_bucket` separately so the difference stays visible.
"""
from __future__ import annotations

NO_TRADE_BUCKET = "<60"


def to_confidence(setup_score: float, identity: bool = True) -> float:
    """Map a raw setup score onto the confidence scale.

    Phase 1 is deliberately the identity. It is a named function rather than an
    inlined `confidence = score` so the calibration has one place to land, and
    so every caller already speaks in terms of confidence rather than score.
    """
    if identity:
        return float(setup_score)
    raise NotImplementedError(
        "no calibrated confidence transform has been fitted yet; "
        "set confidence_is_identity=True until one exists")


def bucket_for(confidence: float, buckets: list[dict]) -> tuple[str, float]:
    """The bucket a confidence falls in, and its allocation multiplier.

    Returns `(NO_TRADE_BUCKET, 0.0)` below the lowest bucket. Below the floor is
    NO TRADE -- not a very small trade. A setup we do not believe in is not
    improved by being small; it just loses less slowly, and it still consumes a
    position slot the ladder could give to something better.
    """
    if not buckets:
        return NO_TRADE_BUCKET, 0.0
    ordered = sorted(buckets, key=lambda b: float(b["min"]))
    chosen: dict | None = None
    for b in ordered:
        if confidence >= float(b["min"]):
            chosen = b
        else:
            break
    if chosen is None:
        return NO_TRADE_BUCKET, 0.0

    lo = float(chosen["min"])
    higher = [float(b["min"]) for b in ordered if float(b["min"]) > lo]
    if higher:
        # Labelled by the range it covers, so a research query can group on it.
        label = f"{lo:.0f}-{min(higher) - 1:.0f}"
    else:
        label = f"{lo:.0f}+"
    return label, float(chosen["mult"])


def multiplier(confidence: float, buckets: list[dict]) -> float:
    return bucket_for(confidence, buckets)[1]


def is_tradable(confidence: float, min_confidence: float,
                buckets: list[dict]) -> bool:
    return confidence >= min_confidence and multiplier(confidence, buckets) > 0.0
