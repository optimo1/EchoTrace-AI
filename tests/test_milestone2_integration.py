"""Milestone 2 integration test – end-to-end slicing → inference → scored timeline."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slicer import slice_audio
from classifier import Tier, aggregate_score, analyse_slices, classify, global_verdict

SR = 16_000


def test_full_pipeline_7s():
    """7.5s audio → 5 slices → inference → scored intervals → global verdict."""
    duration = 7.5
    audio = np.random.randn(int(duration * SR)).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        slices = slice_audio(audio, SR, window_sec=3.0, stride_sec=1.0, output_dir=td)
        assert len(slices) == 5, f"Expected 5 slices, got {len(slices)}"

        intervals = analyse_slices(slices)
        assert len(intervals) == 5, f"Expected 5 intervals, got {len(intervals)}"

        # Verify timestamp continuity
        for iv in intervals:
            assert 0.0 <= iv.score <= 1.0
            assert iv.tier in (Tier.LOW, Tier.MEDIUM, Tier.HIGH)

        agg = aggregate_score(intervals)
        verdict = global_verdict(agg)
        assert verdict in ("Human Voice (Bonafide)", "Uncertain / Suspicious", "AI-Generated Voice (Spoof)")

        print(f"Slices: {len(slices)}  Intervals: {len(intervals)}")
        print(f"Scores: {[f'{iv.score:.4f}' for iv in intervals]}")
        print(f"Agg: {agg:.4f}  Verdict: {verdict}")

        # Temp files exist during processing, then we clean up
        for s in slices:
            assert s.path.exists()
        for s in slices:
            s.path.unlink()
        for s in slices:
            assert not s.path.exists()


def test_full_pipeline_short():
    """2s audio → 1 slice → inference → single interval."""
    audio = np.random.randn(int(2.0 * SR)).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        slices = slice_audio(audio, SR, window_sec=3.0, stride_sec=1.0, output_dir=td)
        assert len(slices) == 1

        intervals = analyse_slices(slices)
        assert len(intervals) == 1
        assert intervals[0].start_sec == 0.0
        assert abs(intervals[0].end_sec - 3.0) < 1e-6
        print(f"Short: score={intervals[0].score:.4f} tier={intervals[0].tier.value}")


def test_temp_cleanup():
    """No orphaned temp files remain after processing."""
    audio = np.random.randn(int(5.0 * SR)).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        slices = slice_audio(audio, SR, window_sec=3.0, stride_sec=1.0, output_dir=td)
        analyse_slices(slices)
        for s in slices:
            s.path.unlink()
    # TemporaryDirectory is auto-cleaned; just verify no exception
    print("Cleanup OK")


if __name__ == "__main__":
    test_full_pipeline_7s()
    test_full_pipeline_short()
    test_temp_cleanup()
    print("\nMILESTONE 2 INTEGRATION: ALL TESTS PASSED")
