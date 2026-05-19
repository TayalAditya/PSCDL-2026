# Technical Report

## Problem

The goal is persistent scene change detection and localization in fixed-camera
videos. The system must generate one binary PNG mask per video second. A changed
region is reported only after it persists for at least `p` seconds and is
suppressed after `c` seconds from its first introduction.

## Approach

The submitted method uses a deterministic OpenCV pipeline suitable for CPU-only
execution:

1. Sample one representative frame for each integer second.
2. Estimate the clean scene with a median baseline from the initial clean
   segment.
3. Compare each frame against the baseline in Lab color space.
4. Compensate global illumination drift using the median shift of the L channel.
5. Run two foreground detectors: a strict chroma-aware detector for colored
   objects and a luma-sensitive detector for dark or low-chroma objects.
6. Remove transient noise with median filtering, morphology, and connected
   component filtering using tuned shape constraints.
7. Maintain independent per-pixel first-seen timestamps for the two detectors
   with a short missing-frame tolerance.
8. Output the cleaned union only when `age >= p` and `age < c`.

The implementation is generic: `p` and `c` are read from the
`generate_mask(p, c, video_path)` arguments and are not hard-coded in the core
function. The provided final runner uses `p=60` and `c=90` as specified for the
released final test set.

## Assumptions

- Videos are fixed-camera surveillance clips.
- Each clip begins with a clean scene segment.
- Persistent objects remain sufficiently stationary after introduction.
- Short moving objects, illumination variation, and minor background motion are
  treated as noise and filtered through temporal persistence.

## Dependencies

- Python
- OpenCV
- NumPy
- SciPy / scikit-image
- tqdm

## Dataset Usage

The organizer-provided development data was used for validation and parameter
sanity checks. The following external/public datasets were staged and reviewed
for change-mask morphology, robustness, and threshold calibration:

- PSCD.zip
- PSCD_pers
- optional_data.zip
- 1024x224.zip
- VL-CMU-CD.zip

PSCD_pers was used for a full threshold-calibration pass over 3,078 paired
before/after samples with integrated change masks. The final inference pipeline
is training-free and does not require these datasets at runtime. No personal or
self-collected dataset is included.

## Limitations

The method does not use a learned semantic model. It can miss very low-contrast
objects or objects whose appearance is extremely similar to the clean baseline.
Large illumination shifts can still create false positives, although the Lab
normalization and persistence filtering reduce this risk.
