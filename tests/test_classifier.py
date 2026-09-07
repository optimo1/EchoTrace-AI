"""Tests for classifier.py – risk tiers, aggregation, and verdict logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from classifier import Interval, Tier, aggregate_score, classify, global_verdict


def test_classify_low():
    assert classify(0.0) == Tier.LOW
    assert classify(0.39) == Tier.LOW


def test_classify_medium():
    assert classify(0.40) == Tier.MEDIUM
    assert classify(0.55) == Tier.MEDIUM
    assert classify(0.69) == Tier.MEDIUM


def test_classify_high():
    assert classify(0.70) == Tier.HIGH
    assert classify(1.0) == Tier.HIGH


def test_aggregate_empty():
    assert aggregate_score([]) == 0.0


def test_aggregate_few_intervals():
    intervals = [
        Interval(start_sec=0.0, end_sec=3.0, score=0.3, tier=Tier.LOW),
        Interval(start_sec=3.0, end_sec=6.0, score=0.8, tier=Tier.HIGH),
    ]
    assert abs(aggregate_score(intervals) - 0.8) < 1e-6  # max of 2


def test_aggregate_top20_percent():
    # 10 intervals, top 20% = top 2
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    intervals = [
        Interval(start_sec=i * 3.0, end_sec=(i + 1) * 3.0, score=s, tier=Tier.LOW)
        for i, s in enumerate(scores)
    ]
    agg = aggregate_score(intervals)
    # top-2 = [0.9, 1.0], mean = 0.95
    assert abs(agg - 0.95) < 1e-6


def test_global_verdict():
    assert global_verdict(0.1) == "Human Voice (Bonafide)"
    assert global_verdict(0.5) == "Uncertain / Suspicious"
    assert global_verdict(0.9) == "AI-Generated Voice (Spoof)"


if __name__ == "__main__":
    test_classify_low()
    test_classify_medium()
    test_classify_high()
    test_aggregate_empty()
    test_aggregate_few_intervals()
    test_aggregate_top20_percent()
    test_global_verdict()
    print("ALL TESTS PASSED")
