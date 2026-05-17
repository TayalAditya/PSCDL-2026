# Strict pipeline (the generic `generate_mask` that was submitted for blind evaluation)

This package implements the required function:

```python
generate_mask(p: int, c: int, video_path: str)
```

By default, the function writes masks to `output_masks/`. The helper runner sets
`PSCDL_OUTPUT_DIR` so each video is written to a folder whose name exactly
matches the input filename stem, for example `test_1.mp4` to `output/test_1/`.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run One Video

From the packaged submission folder:

```bash
python solution.py path/to/test_video.mp4 --p 60 --c 90 --output-dir output_masks
```

For programmatic use:

```python
from solution import generate_mask
generate_mask(p=60, c=90, video_path="path/to/test_video.mp4")
```

This creates:

```text
output_masks/mask_0001.png
output_masks/mask_0002.png
...
```

## Run Final Test Set

From the `strict_pipeline` folder, with the test videos placed in `input/PSCDL2026_Test/test_videos/`:

```bash
python src/run_final.py
python src/validate_submission.py
python src/build_submission.py
```

`build_hybrid_submission.py` merges these masks with the aggressive pipeline's output
(see `docs/hybrid_comparison.md` at the repository root). The masks that were actually
uploaded are in `submission/masks/` at the repository root.

## Method Summary

The pipeline is designed for fixed-camera videos with an initial clean scene.
It builds a median baseline from the clean starting segment, computes robust
foreground candidates using illumination-normalized Lab color differencing,
combines strict chroma-aware and luma-sensitive detectors, removes noise with
morphology and connected-component filtering, and keeps pixel-level first-seen
timestamps. A pixel is emitted only after it has remained changed for at least
`p` seconds and before `c` seconds have elapsed since its first detection.
