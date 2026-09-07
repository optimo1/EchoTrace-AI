"""Classification and risk-tier mapping for EchoTrace AI.

Takes audio slices, runs them through Jabberjay, and returns a scored
timeline of intervals with risk tiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import Jabberjay
import numpy as np

from slicer import Slice


class Tier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


TIER_THRESHOLDS: dict[Tier, tuple[float, float]] = {
    Tier.LOW: (0.0, 0.40),
    Tier.MEDIUM: (0.40, 0.70),
    Tier.HIGH: (0.70, 1.0),
}


@dataclass(frozen=True)
class Interval:
    """A scored time interval from the evidence map."""

    start_sec: float
    end_sec: float
    score: float
    tier: Tier


def classify(score: float) -> Tier:
    """Map a spoof probability *score* ∈ [0, 1] to a risk tier."""
    if score < 0.40:
        return Tier.LOW
    if score < 0.70:
        return Tier.MEDIUM
    return Tier.HIGH


def analyse_slices(
    slices: Sequence[Slice],
    *,
    model: str = "Spectra0",
) -> list[Interval]:
    """Run detection on each slice and return scored intervals.

    Parameters
    ----------
    slices:
        Slices produced by :func:`slicer.slice_audio`.
    model:
        Jabberjay model name.

    Returns
    -------
    list[Interval]
        One interval per slice, ordered by start time.
    """
    jj = Jabberjay.Jabberjay()
    intervals: list[Interval] = []

    for s in slices:
        result = jj.detect(str(s.path), model=model)
        score = result.confidence
        tier = classify(score)
        intervals.append(
            Interval(start_sec=s.start_sec, end_sec=s.end_sec, score=score, tier=tier)
        )

    return intervals


def aggregate_score(intervals: Sequence[Interval]) -> float:
    """Compute the global spoof score as the top-20% mean of interval scores.

    If there are fewer than 5 intervals, the maximum score is used instead.
    """
    if not intervals:
        return 0.0

    scores = np.array([i.score for i in intervals])

    if len(scores) < 5:
        return float(np.max(scores))

    k = max(1, int(np.ceil(len(scores) * 0.2)))
    top_k = np.partition(scores, -k)[-k:]
    return float(np.mean(top_k))


def global_verdict(score: float) -> str:
    """Map a global spoof score to a human-readable verdict."""
    if score < 0.40:
        return "Human Voice (Bonafide)"
    if score < 0.70:
        return "Uncertain / Suspicious"
    return "AI-Generated Voice (Spoof)"
