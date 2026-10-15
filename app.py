"""EchoTrace AI – Streamlit web interface.

Upload an audio file to detect synthetic speech artifacts and view
an interactive Evidence Map of flagged temporal segments.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st

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
        audio, orig_sr = sf.read(tmp_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    except sf.LibsndfileError:
        audio, orig_sr = librosa.load(str(tmp_path), sr=None, mono=True)

    if orig_sr != TARGET_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=TARGET_SAMPLE_RATE)

    return audio.astype(np.float32), TARGET_SAMPLE_RATE


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


def _clear_audio() -> None:
    """Drop the loaded audio (upload or recording) so the analysis is removed."""
    for key in ("audio_upload", "audio_record"):
        st.session_state.pop(key, None)


def _fmt(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def main() -> None:
    st.set_page_config(page_title="EchoTrace AI", layout="wide")

    logo_path = Path(__file__).parent / "public" / "eco-icon.png"
    col_logo, col_title = st.columns([0.1, 0.9])
    with col_logo:
        if logo_path.exists():
            st.image(str(logo_path), width=80)
    with col_title:
        st.markdown("# EchoTrace AI")
        st.caption("Synthetic speech & deepfake detection with explainable Evidence Map")

    source = st.radio(
        "Input source",
        ("Upload a file", "Record from microphone"),
        horizontal=True,
    )

    if source == "Upload a file":
        media = st.file_uploader(
            "Upload audio",
            type=SUPPORTED_FORMATS,
            help="Accepted formats: WAV, MP3, M4A, OGG",
            key="audio_upload",
        )
        media_format = f"audio/{Path(media.name).suffix.lstrip('.')}" if media else "audio/wav"
        empty_msg = "Drag and drop an audio file to begin analysis."
    else:
        media = st.audio_input(
            "Record audio",
            help="Click the microphone, speak, then stop. Recording is captured as WAV.",
            key="audio_record",
        )
        media_format = "audio/wav"
        empty_msg = "Record audio with your microphone to begin analysis."

    if media is None:
        st.info(empty_msg)
        return

    # ---- Load & display audio ----
    with st.spinner("Loading audio…"):
        audio, sr = _load_audio_as_mono_16k(media)

    duration_sec = len(audio) / sr
    col_player, col_remove = st.columns([0.85, 0.15])
    with col_player:
        st.audio(media, format=media_format)
    with col_remove:
        st.button(
            "Remove audio",
            on_click=_clear_audio,
            use_container_width=True,
            help="Clear this audio and remove its analysis results",
        )

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
