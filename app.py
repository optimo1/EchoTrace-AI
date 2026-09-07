"""EchoTrace AI – Streamlit web interface.

Upload an audio file to detect synthetic speech artifacts and view
an interactive Evidence Map of flagged temporal segments.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydub import AudioSegment

import Jabberjay
from slicer import Slice, slice_audio
from classifier import Interval, Tier, analyse_slices, aggregate_score, global_verdict

SUPPORTED_FORMATS = ("wav", "mp3", "m4a", "ogg")
TARGET_SAMPLE_RATE = 16_000

TIER_COLORS = {
    Tier.LOW: "#2ecc71",
    Tier.MEDIUM: "#f1c40f",
    Tier.HIGH: "#e74c3c",
}


@st.cache_resource
def _get_jabberjay() -> Jabberjay.Jabberjay:
    """Return a cached Jabberjay instance (loads weights once per server lifetime)."""
    return Jabberjay.Jabberjay()


def _load_audio_as_mono_16k(uploaded_file) -> tuple[np.ndarray, int]:
    """Read an uploaded file into a 1-D float32 numpy array at 16 kHz mono."""
    suffix = Path(uploaded_file.name).suffix.lstrip(".").lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: .{suffix}")

    raw = uploaded_file.read()
    with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)

    try:
        segment = AudioSegment.from_file(str(tmp_path))
        segment = segment.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(1)
        samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
        samples /= np.iinfo(segment.array_type).max
    finally:
        tmp_path.unlink(missing_ok=True)

    return samples, TARGET_SAMPLE_RATE


def _build_evidence_map(intervals: list[Interval], total_duration: float) -> go.Figure:
    """Create a Plotly timeline showing each interval as a coloured bar."""
    fig = go.Figure()

    for iv in intervals:
        fig.add_trace(go.Bar(
            x=[iv.end_sec - iv.start_sec],
            y=["Evidence Map"],
            base=iv.start_sec,
            orientation="h",
            marker_color=TIER_COLORS[iv.tier],
            hovertemplate=(
                f"<b>{_fmt(iv.start_sec)} – {_fmt(iv.end_sec)}</b><br>"
                f"Confidence: {iv.score:.1%}<br>"
                f"Tier: {iv.tier.value.capitalize()}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        xaxis=dict(
            title="Time (seconds)",
            range=[0, total_duration],
            dtick=1.0,
        ),
        yaxis=dict(visible=False),
        height=120,
        margin=dict(l=0, r=0, t=10, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="overlay",
    )
    return fig


def _fmt(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def main() -> None:
    st.set_page_config(page_title="EchoTrace AI", layout="wide")
    st.title("EchoTrace AI")
    st.caption("Synthetic speech & deepfake detection with explainable Evidence Map")

    uploaded = st.file_uploader(
        "Upload audio",
        type=SUPPORTED_FORMATS,
        help="Accepted formats: WAV, MP3, M4A, OGG",
    )

    if uploaded is None:
        st.info("Drag and drop an audio file to begin analysis.")
        return

    # ---- Load & display audio ----
    with st.spinner("Loading audio…"):
        audio, sr = _load_audio_as_mono_16k(uploaded)

    duration_sec = len(audio) / sr
    st.audio(uploaded, format=f"audio/{Path(uploaded.name).suffix.lstrip('.')}")

    st.markdown(f"**Duration:** {duration_sec:.1f}s &nbsp;|&nbsp; **Sample rate:** {sr} Hz")

    # ---- Slice & analyse ----
    with tempfile.TemporaryDirectory() as td:
        slices = slice_audio(audio, sr, output_dir=Path(td))

        progress_bar = st.progress(0, text="Analysing chunks…")
        status_box = st.empty()

        def _update_progress(current: int, total: int) -> None:
            pct = current / total
            progress_bar.progress(pct, text=f"Analysing chunk {current}/{total}")
            status_box.caption(f"Processed {current} of {total} slices")

        intervals = analyse_slices(slices, jj=_get_jabberjay(), on_progress=_update_progress)

    progress_bar.empty()
    status_box.empty()

    agg = aggregate_score(intervals)
    verdict = global_verdict(agg)

    st.success(f"**Verdict:** {verdict}  —  Synthetic probability: {agg * 100:.1f}%")

    # ---- Evidence Map ----
    st.subheader("Evidence Map")
    fig = _build_evidence_map(intervals, duration_sec)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Summary Metrics ----
    flag_count = sum(1 for iv in intervals if iv.tier != Tier.LOW)
    verdict_color = {
        "Human Voice (Bonafide)": "green",
        "Uncertain / Suspicious": "orange",
        "AI-Generated Voice (Spoof)": "red",
    }.get(verdict, "gray")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Verdict", verdict)
    col2.metric("Confidence", f"{agg * 100:.1f}%")
    col3.metric("Duration", f"{duration_sec:.1f}s")
    col4.metric("Flagged Segments", f"{flag_count} / {len(intervals)}")

    # ---- Flagged Intervals Table ----
    if intervals:
        flagged = sorted(
            [iv for iv in intervals if iv.tier != Tier.LOW],
            key=lambda iv: iv.score,
            reverse=True,
        )
        if flagged:
            with st.expander("Flagged Segments (sorted by confidence)", expanded=False):
                df = pd.DataFrame([
                    {
                        "Start": _fmt(iv.start_sec),
                        "End": _fmt(iv.end_sec),
                        "Confidence": f"{iv.score:.1%}",
                        "Tier": iv.tier.value.capitalize(),
                    }
                    for iv in flagged
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
