"""Tests for slicer.py – verifies window logic and timestamp correctness."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slicer import Slice, slice_audio

SR = 16_000
WIN = 3.0
STRIDE = 1.0


def _make_dummy(duration_sec: float) -> tuple[np.ndarray, int]:
    n = int(duration_sec * SR)
    return np.random.randn(n).astype(np.float32), SR


def test_short_audio_below_one_window():
    """Audio < window_sec should produce exactly 1 slice of duration == window_sec."""
    audio, sr = _make_dummy(2.0)
    with tempfile.TemporaryDirectory() as td:
        slices = slice_audio(audio, sr, window_sec=WIN, stride_sec=STRIDE, output_dir=td)
        assert len(slices) == 1, f"Expected 1 slice, got {len(slices)}"
        s = slices[0]
        assert abs(s.start_sec - 0.0) < 1e-6
        assert abs(s.end_sec - WIN) < 1e-6
        data, out_sr = sf.read(str(s.path))
        assert len(data) == int(WIN * SR)
        assert out_sr == SR


def test_audio_exactly_one_window():
    """3s audio (== window) → 1 slice."""
    audio, sr = _make_dummy(3.0)
    with tempfile.TemporaryDirectory() as td:
        slices = slice_audio(audio, sr, window_sec=WIN, stride_sec=STRIDE, output_dir=td)
        assert len(slices) == 1
        assert slices[0].start_sec == 0.0
        assert abs(slices[0].end_sec - 3.0) < 1e-6


def test_longer_audio_overlap():
    """7.5s audio → 5 slices at 0.0, 1.0, 2.0, 3.0, 4.0 s."""
    audio, sr = _make_dummy(7.5)
    with tempfile.TemporaryDirectory() as td:
        slices = slice_audio(audio, sr, window_sec=WIN, stride_sec=STRIDE, output_dir=td)
        assert len(slices) == 5, f"Expected 5 slices, got {len(slices)}"
        expected_starts = [0.0, 1.0, 2.0, 3.0, 4.0]
        for s, exp_start in zip(slices, expected_starts):
            assert abs(s.start_sec - exp_start) < 1e-6
            assert abs(s.end_sec - (exp_start + WIN)) < 1e-6
        for a, b in zip(slices, slices[1:]):
            assert abs(a.end_sec - b.start_sec - (WIN - STRIDE)) < 1e-6


def test_slice_files_are_wav():
    audio, sr = _make_dummy(4.0)
    with tempfile.TemporaryDirectory() as td:
        slices = slice_audio(audio, sr, window_sec=WIN, stride_sec=STRIDE, output_dir=td)
        for s in slices:
            assert s.path.exists()
            assert s.path.suffix == ".wav"


def test_cleanup():
    """Slice temp files are regular files that can be deleted."""
    audio, sr = _make_dummy(3.0)
    with tempfile.TemporaryDirectory() as td:
        slices = slice_audio(audio, sr, window_sec=WIN, stride_sec=STRIDE, output_dir=td)
        for s in slices:
            assert s.path.exists()
        for s in slices:
            s.path.unlink()
        for s in slices:
            assert not s.path.exists()


if __name__ == "__main__":
    test_short_audio_below_one_window()
    test_audio_exactly_one_window()
    test_longer_audio_overlap()
    test_slice_files_are_wav()
    test_cleanup()
    print("ALL TESTS PASSED")
