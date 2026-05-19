# PSCDL 2026: Methodology Report

**Team:** The Solo Warrior (Aditya Tayal)  
**Institute:** IIT Mandi  
**Contact:** b23243@students.iitmandi.ac.in

---

## 1. Approach Overview

We propose a classical computer vision pipeline combining **background subtraction**, **CIE-Lab colour-space change detection**, and **per-pixel temporal persistence tracking** to distinguish genuinely persistent scene changes from transient motion.

The pipeline requires no training and is fully parameter-driven via `p` (persistence threshold) and `c` (cooldown period).

---

## 2. Pipeline Description

### Step 1 — Background Estimation
A robust pixel-wise **median background** is computed from frames sampled uniformly across the first 25% of the video (capped at 90 s). Sparse sampling (one frame every ~3 s) ensures moving people are averaged out of the background model.

### Step 2 — Per-Frame Foreground Detection
For each second of video, up to 5 evenly-spaced frames are sampled. For each frame:
- **Global brightness normalisation**: the frame is scaled so its mean brightness matches the background, suppressing cloud/sun illumination shifts.
- **CIE-Lab difference**: both frame and background are converted to Lab colour space; per-channel absolute differences are computed.
- **Weighted score**: `0.4·ΔL + 0.8·Δa + 0.8·Δb` — the L channel contributes less to reduce lighting artefacts; chromatic channels dominate.
- **Gaussian blur** (5×5) suppresses salt-and-pepper noise.
- **Thresholding** at **46.0** (calibrated — see Section 4).
- **Majority vote** across sampled frames: a pixel is foreground if detected in >50% of samples within that second.

### Step 3 — Morphological Cleanup
- MORPH_OPEN (5×5 ellipse kernel) removes speckle noise.
- MORPH_CLOSE (13×13 ellipse kernel) fills small holes.
- Connected components smaller than **400 px** are discarded as noise.

### Step 4 — Temporal Persistence Tracking
Per-pixel counters are maintained:
- `consecutive_fg`: number of consecutive seconds a pixel has been foreground.
- `streak_start`: second at which the current streak began.

A pixel is flagged as a **persistent change** if:
- `consecutive_fg >= p` (present for at least `p` consecutive seconds), AND
- `(current_sec − streak_start) < c` (not yet assimilated into background).

Both counters reset immediately when a pixel returns to background, so transient objects never accumulate enough streak to be flagged.

### Step 5 — Output
For each second `t` (1-indexed), `mask_XXXX.png` is written — a grayscale image where **255 = persistent change, 0 = background**.

---

## 3. Implementation

```
generate_mask(p: int, c: int, video_path: str) -> None
```

- **Language:** Python 3.8+
- **Dependencies:** OpenCV ≥ 4.8.0, NumPy ≥ 1.24.0
- **Test parameters:** p = 60 s, c = 90 s
- The implementation is fully generic; p and c are runtime arguments.

---

## 4. Threshold Calibration

The detection threshold of **46.0** was calibrated across three publicly available datasets:

| Dataset | Pairs | Best Threshold | Peak F1 |
|---------|-------|---------------|---------|
| PSCD 1024×224 | 200 | 46 | 0.136 |
| VL-CMU-CD (train) | 80 | 65 | 0.192 |
| PSCD (full-res) | 200 | 40 | — |

Threshold 46 coincides with the PSCD peak. Since the temporal persistence filter (requiring ≥ p consecutive foreground seconds) acts as a strong precision filter, recall is prioritised at the per-frame level, and threshold 46 was chosen.

---

## 5. Datasets Used

- **PSCD Dataset** (public): https://sakuradaken.net/pscd/term_of_use.html — used for calibration
- **VL-CMU-CD Dataset** (public): https://huggingface.co/datasets/Flourish/VL-CMU-CD — used for calibration
- No custom dataset was created.

---

## 6. Execution Instructions

```bash
# Install dependencies
pip install -r requirements.txt

# Run on a single video
python solution.py <video_path> -p 60 -c 90 -o <output_dir>

# Run on all test videos (produces test_1/, test_2/, ... folders)
python run_pipeline.py --p 60 --c 90 \
    --video_dir PSCDL2026_Test/test_videos \
    --output_dir submission
```
