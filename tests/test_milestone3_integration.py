"""Milestone 3 integration test – end-to-end Streamlit app verification.

Simulates the full pipeline: file upload → slicing → inference → UI render.
Uses Streamlit AppTest for headless server verification.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_full_pipeline_no_streamlit():
    """Test the full pipeline without Streamlit runtime (unit-level integration)."""
    from slicer import slice_audio
    from classifier import analyse_slices, aggregate_score, global_verdict, Tier

    audio = np.random.randn(16000 * 7).astype(np.float32)
    sr = 16000

    with tempfile.TemporaryDirectory() as td:
        slices = slice_audio(audio, sr, output_dir=Path(td))
        assert len(slices) > 0, "Slicer produced no slices"

        intervals = analyse_slices(slices)
        assert len(intervals) == len(slices), "Interval count mismatch"

        agg = aggregate_score(intervals)
        verdict = global_verdict(agg)
        assert 0.0 <= agg <= 1.0
        assert verdict in ("Human Voice (Bonafide)", "Uncertain / Suspicious", "AI-Generated Voice (Spoof)")

        # Verify evidence map builder
        from app import _build_evidence_map, _fmt
        fig = _build_evidence_map(intervals, 7.0)
        assert len(fig.data) == len(intervals)

        # Verify hover templates contain timestamps
        for trace, iv in zip(fig.data, intervals):
            assert _fmt(iv.start_sec) in trace.hovertemplate

        # Verify flagged segments
        flagged = [iv for iv in intervals if iv.tier != Tier.LOW]
        flag_count = len(flagged)

        print(f"Pipeline OK: {len(slices)} slices → {len(intervals)} intervals")
        print(f"  Agg: {agg:.4f}  Verdict: {verdict}  Flags: {flag_count}/{len(intervals)}")
        print(f"  Evidence Map traces: {len(fig.data)}")

        # Cleanup temp slice files
        for s in slices:
            s.path.unlink(missing_ok=True)


def test_app_import():
    """Verify app.py imports cleanly without Streamlit side effects."""
    import importlib
    mod = importlib.import_module("app")
    assert hasattr(mod, "_load_audio_as_mono_16k")
    assert hasattr(mod, "_build_evidence_map")
    assert hasattr(mod, "_get_jabberjay")
    assert hasattr(mod, "main")
    print("app.py import OK")


def test_app_test_simulate():
    """Run Streamlit AppTest to verify the app can be loaded and rendered."""
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_path)
    at.run(timeout=120)
    assert not at.exception, f"App raised exception: {at.exception}"

    # Verify initial state (no upload yet)
    assert at.info[0].value == "Drag and drop an audio file to begin analysis."
    print("Streamlit AppTest OK: app loads without exceptions")


if __name__ == "__main__":
    test_app_import()
    test_full_pipeline_no_streamlit()
    test_app_test_simulate()
    print("\nMILESTONE 3 INTEGRATION: ALL TESTS PASSED")
