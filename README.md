# PSCDL 2026: Persistent Scene Change Detection and Localization

Solo entry to the **PSCDL Challenge 2026** run by Vehant Research Labs (Vehant Technologies).
Team name on the form: *The Solo Warrior*. Aditya Tayal, B.Tech CSE, IIT Mandi.

## The task

Fixed-camera surveillance video. Emit one binary mask per second of video. A region counts as a
persistent change only after it has stayed in place for **p** seconds and stops counting **c** seconds
after it first appeared, when it is treated as assimilated into the background. The final test set used
`p = 60` and `c = 90`. Submissions are scored on pixel F1 against ground-truth masks.

Mandatory interface:

```python
generate_mask(p: int, c: int, video_path: str) -> None
# writes output_masks/mask_0001.png, mask_0002.png, ... one per second
```

## Two pipelines and a hybrid

Nothing here is trained. Everything is OpenCV and NumPy on a CPU.

**Aggressive pipeline** (`solution.py`, 201 lines). Median background from the clean opening segment,
sampled about every 3 seconds so moving people average out. Per second, up to 5 frames are
brightness-normalised, converted to CIE-Lab and scored as `0.4·|ΔL| + 0.8·|Δa| + 0.8·|Δb|`, blurred,
thresholded at 46 and majority-voted. Morphology (open 5×5, close 13×13) and components under 400 px
are dropped. Two counters per pixel do the actual work: `consecutive_fg` (seconds in a row as
foreground) and `streak_start` (when the streak began). A pixel is emitted only while
`consecutive_fg >= p` and `(sec - streak_start) < c`. Both counters reset the moment a pixel returns to
background.

**Strict pipeline** (`strict_pipeline/`). Same skeleton with an adaptive threshold
(`median + 6·MAD`, clamped to 22..58), two detectors (chroma-strict for coloured objects, luma-sensitive
for dark ones), first-seen timestamps with a 2-second miss tolerance and shape-constrained component
filtering. This is the generic `generate_mask` that was submitted for blind evaluation.

**Hybrid** (`strict_pipeline/src/build_hybrid_submission.py`). On the five final test videos the two
pipelines disagreed by a wide margin (non-empty masks 305/287/349/130/251 against 42/127/175/9/262).
The shipped masks keep the strict pipeline's structure and borrow recall from the aggressive output
only at the start of each detected run. Details and the full table in `docs/hybrid_comparison.md`.

## Threshold calibration

Five annotated sample videos are not enough to tune a threshold, so it was swept on public
change-detection sets with `calibrate_threshold.py`:

| dataset | pairs | peak threshold | peak F1 |
|---|---|---|---|
| PSCD 1024×224 | 200 | 46 | 0.136 |
| VL-CMU-CD (train) | 80 | 65 | 0.192 |
| PSCD_pers, full pass | 3,078 | 46 | confirms |

46 was chosen because the persistence counters already act as a precision filter, so a lower
per-frame threshold buys recall, which is the side this pipeline needed. Both datasets are image
pairs without p and c semantics, which is the main caveat on the whole calibration.

## What was submitted

`submission/masks/` holds the 2,039 masks uploaded on 14 June 2026 (`test_1` to `test_5`, 449 + 420 +
480 + 300 + 390 seconds, 768 non-empty), plus the report PDF. Result: not published at the time of
writing.

## Repository layout

```
solution.py                 aggressive pipeline, generate_mask(p, c, video_path)
run_pipeline.py             batch runner: one output folder per test video
calibrate_threshold.py      threshold sweep on PSCD_pers-style before/after pairs
requirements.txt
strict_pipeline/            adaptive-threshold pipeline, runners, validator, hybrid builder
docs/                       methodology report, technical report (md + pdf), hybrid comparison
submission/                 uploaded report and the 2,039 final masks
```

## Run

```bash
pip install -r requirements.txt

# one video
python solution.py <video.mp4> -p 60 -c 90 -o output_masks/

# every video in a folder, one output folder per video
python run_pipeline.py --p 60 --c 90 --video_dir <videos/> --output_dir output/

# threshold sweep on a PSCD_pers-style dataset (t0/, t1/, label_t1_integ/)
python calibrate_threshold.py --data_dir <PSCD_pers/>
```

## Not in this repository

The Vehant sample and test videos and the challenge specification are distributed to registered
participants only and are not included. PSCD and VL-CMU-CD are public and are downloaded separately.
