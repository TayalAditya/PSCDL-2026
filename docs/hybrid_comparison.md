# Submission comparison, 18 May 2026

Two pipelines were run on the five final test videos:

- **aggressive**: `solution.py` at the repository root. Fixed threshold 46, single detector.
- **strict**: `strict_pipeline/src/solution.py`. Adaptive threshold (median + 6·MAD), chroma-strict and luma detectors, first-seen timestamps with a 2-second miss tolerance.

Non-empty masks per video (out of 449 / 420 / 480 / 300 / 390 seconds):

| video | aggressive | strict | hybrid v3 | shipped (14 Jun) |
|---|---|---|---|---|
| test_1 | 305 | 42 | 76 | 77 |
| test_2 | 287 | 127 | 128 | 129 |
| test_3 | 349 | 175 | 189 | 189 |
| test_4 | 130 | 9 | 49 | 77 |
| test_5 | 251 | 262 | 262 | 296 |

The hybrid (`strict_pipeline/src/build_hybrid_submission.py`):

- keeps the strict masks as the base, so region shape and the end of every run come from the strict pipeline;
- adds recall from the aggressive output only at the start of each detected run, so a region can be flagged earlier but never later;
- never inherits the aggressive pipeline's long full-video tails.

Caveat, written the same night: ground truth for the final test set is unavailable, so this is a
risk-balanced candidate, not a proven F1 winner. The generic `generate_mask` implementation shipped for
blind evaluation is the strict one.
