"""Sliding-window audio slicer for EchoTrace AI.

Produces fixed-size chunks from an audio signal, writing each to a
temporary WAV file that the caller must close (or use as a context
manager via :func:`slice_audio`).
"""

from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class Slice:
    """Metadata for a single audio slice."""

    path: Path
    start_sec: float
    end_sec: float
    sample_rate: int


def slice_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    window_sec: float = 3.0,
    stride_sec: float = 1.0,
    output_dir: Path | str | None = None,
) -> list[Slice]:
    """Yield fixed-size overlapping slices from *audio*.

    Parameters
    ----------
    audio:
        1-D float32 audio samples.
    sample_rate:
        Samples per second.
    window_sec:
        Duration of each slice in seconds (default 3.0).
    stride_sec:
        Hop between successive slices in seconds (default 1.0).
    output_dir:
        Directory for temporary WAV files.  If *None* a system temp
        directory is used.

    Returns
    -------
    list[Slice]
        Ordered slices with their time boundaries.
    """

    window_len = int(window_sec * sample_rate)
    stride_len = int(stride_sec * sample_rate)

    # Total number of samples
    n_samples = len(audio)

    # If audio is shorter than one window, pad with zeros so we get
    # exactly one slice covering [0, window_sec].
    if n_samples < window_len:
        padded = np.zeros(window_len, dtype=audio.dtype)
        padded[:n_samples] = audio
        audio = padded
        n_samples = len(audio)

    # Determine where the last valid window can start
    max_start = n_samples - window_len

    starts = list(range(0, max_start + 1, stride_len))

    out_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
    out_dir.mkdir(parents=True, exist_ok=True)

    slices: list[Slice] = []
    for i, start in enumerate(starts):
        end = start + window_len
        chunk = audio[start:end]

        start_sec = start / sample_rate
        end_sec = end / sample_rate

        fname = out_dir / f"slice_{i:04d}.wav"
        sf.write(str(fname), chunk, sample_rate, subtype="PCM_16")
        slices.append(Slice(path=fname, start_sec=start_sec, end_sec=end_sec, sample_rate=sample_rate))

    return slices
