# EchoTrace AI: Technical Implementation Specification & Development Plan

## 1. Project Overview
**EchoTrace AI** is a web-based synthetic speech and voice deepfake detection system designed to eliminate "black-box" ambiguity. Rather than simply returning a single global probability, the application conducts sliding-window temporal analysis across an audio track to generate an interactive **Evidence Map**, identifying the exact temporal segments exhibiting synthetic speech artifacts.

---

## 2. Architecture & Processing Flow

```text
User Uploads Audio (.wav / .mp3)
                │
                ▼
      [ Audio Preprocessor ]
   (Resample 16kHz Mono, Normalize)
                │
                ▼
     [ Sliding-Window Slicer ]
(3.0s window, 1.0s stride, 2.0s overlap)
                │
                ▼
    [ Jabberjay Detection Engine ]
      (Pre-trained Spectra0 model)
                │
                ▼
       [ Evidence Map Logic ]
(Map chunk scores to 🟢 / 🟡 / 🔴 tiers)
                │
                ▼
     [ Streamlit Web Application ]
 (Audio Player + Verdict + Interactive Plotly Timeline)
```

---

## 3. Technology Stack & Dependencies

- **Runtime Environment:** Python 3.11+
- **Inference Engine:** `jabberjay` (Spectra0 model backbone for fast, low-compute CPU inference)
- **Audio Processing:** `librosa`, `soundfile`, `pydub`, `numpy`
- **Visualization & UI:** `streamlit`, `plotly`

### Dependencies (`requirements.txt`)
```text
jabberjay>=0.1.0
streamlit>=1.30.0
librosa>=0.10.0
soundfile>=0.12.1
pydub>=0.25.1
plotly>=5.18.0
numpy>=1.24.0
```

---

## 4. Detailed Component Specifications

### 4.1 Input Ingestion & Preprocessing
1. Support common audio formats: `.wav`, `.mp3`, `.m4a`, `.ogg`.
2. Convert audio to mono at a standardized 16 kHz sample rate.
3. Cache model loading in memory via `@st.cache_resource` to ensure instant inference without reloading weights per action.

### 4.2 Sliding-Window Slicing Engine (`slicer.py`)
1. **Window Size:** 3.0 seconds.
2. **Stride Step:** 1.0 second (providing a 2.0-second overlap between consecutive windows).
3. **Short Audio Handling:** Audio tracks under 3.0 seconds are evaluated as a single slice without windowing.
4. Export each slice into a temporary in-memory WAV buffer or temporary directory via `tempfile`.

### 4.3 Classification & Evidence Map Mapping
1. Evaluate each chunk using:
   ```python
   result = jj.detect(chunk_wav_path, model="Spectra0")
   ```
2. Extract the synthetic voice confidence score ($P_{spoof} \in [0.0, 1.0]$).
3. Categorize segments into risk levels:
   - **🟢 Low Suspicion:** $P_{spoof} < 0.40$
   - **🟡 Medium Suspicion:** $0.40 \le P_{spoof} < 0.70$
   - **🔴 High Suspicion:** $P_{spoof} \ge 0.70$
4. Aggregate intervals: `[start_time, end_time, suspicion_score, tier_label]`.

### 4.4 Global Metric Aggregation
- **Overall Track Score:** Top-20% mean of suspicious windows (or maximum window probability) to capture localized deepfake insertions without being diluted by clean segments.
- **Global Verdict:**
  - `Human Voice (Bonafide)`: Global $P_{spoof} < 0.40$
  - `Uncertain / Suspicious`: $0.40 \le P_{spoof} < 0.70$
  - `AI-Generated Voice (Spoof)`: Global $P_{spoof} \ge 0.70$

---

## 5. UI/UX Design (Streamlit + Plotly)

### 5.1 Header & Media Player
- Application title and purpose description.
- File upload widget with drag-and-drop.
- Native HTML5 audio player (`st.audio`) for playback.

### 5.2 Summary Metrics
- Grid showing:
  - **Verdict Badge:** Human (Green), Uncertain (Yellow), AI-Generated (Red).
  - **Confidence Metric:** Percentage indicating overall synthetic probability.
  - **Track Stats:** Total duration, number of examined slices, and count of flagged intervals.

### 5.3 Interactive Evidence Map
- **Interactive Plotly Timeline:**
  - Horizontal bar / segment view mapping seconds along the horizontal axis.
  - Color blocks: Green (`#2ecc71`), Yellow (`#f1c40f`), Red (`#e74c3c`).
  - Hover tooltips showing exact timestamps (e.g., `00:04 - 00:07`) and confidence scores.

### 5.4 Detailed Flagged Intervals Table
- Collapsible data table summarizing all analyzed chunks, sorted by highest suspicion score.

---

## 6. Implementation Milestones

- [ ] **Milestone 1: Environment Setup**
  - Verify Python 3.11 environment.
  - Install dependencies via `pip install -r requirements.txt`.
  - Validate baseline Jabberjay model inference on a test audio file.

- [ ] **Milestone 2: Slicing Pipeline Implementation**
  - Develop sliding-window generator with configurable window size and step.
  - Implement segment probability extraction and risk tier mapping.

- [ ] **Milestone 3: Streamlit Interface Assembly**
  - Create file uploader and audio player.
  - Implement caching for model initialization (`@st.cache_resource`).
  - Integrate Plotly interactive Evidence Map.
  - Display summary metrics and interval breakdown table.

- [ ] **Milestone 4: Performance & Refinements**
  - Add progress tracking bar during multi-chunk evaluation.
  - Ensure temporary files are cleaned up reliably after processing.
