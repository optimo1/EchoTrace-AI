"""Validate baseline Jabberjay Spectra0 inference on a dummy audio file."""

import tempfile
from pathlib import Path

import Jabberjay
import numpy as np
import soundfile as sf


def main() -> None:
    # Generate 3s of silence at 16 kHz as a minimal test signal
    sample_rate = 16_000
    duration_s = 3.0
    samples = np.zeros(int(sample_rate * duration_s), dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, samples, sample_rate)
        tmp_path = Path(tmp.name)

    try:
        jj = Jabberjay.Jabberjay()
        result = jj.detect(str(tmp_path), model="Spectra0")

        # Validate result structure
        assert hasattr(result, "label"), "Missing 'label' attribute"
        assert hasattr(result, "is_bonafide"), "Missing 'is_bonafide' attribute"
        assert hasattr(result, "confidence"), "Missing 'confidence' attribute"
        assert isinstance(result.confidence, float), f"confidence is {type(result.confidence)}, expected float"
        assert 0.0 <= result.confidence <= 1.0, f"confidence {result.confidence} out of [0, 1]"
        assert result.label.lower() in ("bonafide", "spoof"), f"Unexpected label: {result.label}"

        print(f"PASS  label={result.label}  confidence={result.confidence:.4f}")
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
