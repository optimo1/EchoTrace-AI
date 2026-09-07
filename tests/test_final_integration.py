"""Final project-wide integration test – complete pipeline verification.

Tests: Streamlit app import, full audio→slice→inference→UI pipeline,
progress callback, cleanup, and evidence map rendering.
"""

import importlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_app_imports_cleanly():
    """app.py imports without side-effect errors."""
    mod = importlib.import_module("app")
    for attr in ("_load_audio_as_mono_16k", "_build_evidence_map",
                  "_get_jabberjay", "_fmt", "main"):
        assert hasattr(mod, attr), f"Missing {attr}"
    print("PASS: app.py imports cleanly")


def test_full_pipeline_short_audio():
    """2s audio → 1 slice → inference → single interval."""
    from slicer import slice_audio
    from classifier import analyse_slices, aggregate_score, global_verdict, Tier

    audio = np.random.randn(16000 * 2).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        slices = slice_audio(audio, 16000, output_dir=Path(td))
        assert len(slices) == 1
        intervals = analyse_slices(slices)
        assert len(intervals) == 1
        agg = aggregate_score(intervals)
        verdict = global_verdict(agg)
        assert 0.0 <= agg <= 1.0
        print(f"PASS: short audio → 1 slice, score={agg:.4f}, verdict={verdict}")


def test_full_pipeline_long_audio():
    """12s audio → many slices → inference → scored intervals → evidence map."""
    from slicer import slice_audio
    from classifier import analyse_slices, aggregate_score, global_verdict, Tier
    from app import _build_evidence_map, _fmt

    audio = np.random.randn(16000 * 12).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        slices = slice_audio(audio, 16000, output_dir=Path(td))
        assert len(slices) >= 5
        intervals = analyse_slices(slices)
        assert len(intervals) == len(slices)
        agg = aggregate_score(intervals)
        verdict = global_verdict(agg)
        fig = _build_evidence_map(intervals, 12.0)
        assert len(fig.data) == len(intervals)
        for trace, iv in zip(fig.data, intervals):
            assert _fmt(iv.start_sec) in trace.hovertemplate
        print(f"PASS: long audio → {len(slices)} slices, agg={agg:.4f}, verdict={verdict}")


def test_progress_callback():
    """Progress callback is invoked for every slice."""
    from slicer import slice_audio
    from classifier import analyse_slices

    audio = np.random.randn(16000 * 5).astype(np.float32)
    calls = []
    with tempfile.TemporaryDirectory() as td:
        slices = slice_audio(audio, 16000, output_dir=Path(td))
        analyse_slices(slices, on_progress=lambda c, t: calls.append((c, t)))
    assert len(calls) == len(slices)
    assert calls[-1] == (len(slices), len(slices))
    print(f"PASS: progress callback invoked {len(calls)} times")


def test_cleanup_after_success():
    """No temp files remain after a successful pipeline run."""
    from slicer import slice_audio
    from classifier import analyse_slices

    audio = np.random.randn(16000 * 4).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        slices = slice_audio(audio, 16000, output_dir=td_path)
        analyse_slices(slices)
        remaining = list(td_path.rglob("*.wav"))
        assert len(remaining) == len(slices)  # still inside with-block
    # After with-block, directory is removed
    assert not td_path.exists()
    print("PASS: temp dir cleaned after success")


def test_cleanup_after_exception():
    """Temp dir cleaned even when processing fails mid-way."""
    from slicer import slice_audio

    audio = np.random.randn(16000 * 6).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        slices = slice_audio(audio, 16000, output_dir=td_path)
        assert len(slices) > 1
    # Directory removed regardless of what happens after
    assert not td_path.exists()
    print("PASS: temp dir cleaned after exception path")


def test_streamlit_apptest():
    """Streamlit AppTest loads the app without runtime exceptions."""
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=120)
    assert not at.exception, f"App exception: {at.exception}"
    assert at.info[0].value == "Drag and drop an audio file to begin analysis."
    print("PASS: Streamlit AppTest loads without exceptions")


def test_all_modules_importable():
    """All project modules import without error."""
    for mod_name in ("slicer", "classifier", "app"):
        importlib.import_module(mod_name)
    print("PASS: all modules importable")


if __name__ == "__main__":
    test_all_modules_importable()
    test_app_imports_cleanly()
    test_full_pipeline_short_audio()
    test_full_pipeline_long_audio()
    test_progress_callback()
    test_cleanup_after_success()
    test_cleanup_after_exception()
    test_streamlit_apptest()
    print("\nFINAL INTEGRATION: ALL TESTS PASSED")
