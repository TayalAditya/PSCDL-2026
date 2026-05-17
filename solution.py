"""
PSCDL 2026 - Persistent Scene Change Detection and Localization
Pipeline: Background-Subtraction + Temporal Persistence Filtering
"""

import cv2
import numpy as np
from pathlib import Path


# ── Background model ──────────────────────────────────────────────────────────

def _build_background(cap, fps: float, n_seconds: int = 60) -> np.ndarray:
    """
    Compute a robust median background from the first n_seconds of video.
    Samples one frame every ~3 seconds so moving people average out.
    """
    step = max(3, n_seconds // 20)          # ≤ 20 samples total
    frame_list = []
    for sec in range(0, n_seconds, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
        ret, frame = cap.read()
        if ret:
            frame_list.append(frame.astype(np.float32))

    if not frame_list:
        raise RuntimeError("No frames read for background estimation.")

    bg = np.median(frame_list, axis=0).astype(np.uint8)
    return bg


# ── Per-frame foreground detection ────────────────────────────────────────────

def _foreground_mask(frame: np.ndarray, bg: np.ndarray, threshold: float = 46.0) -> np.ndarray:
    """
    Return a uint8 binary mask (0/1) of foreground pixels.
    Uses CIE-Lab difference so hue shifts don't inflate brightness diffs.
    Global brightness is equalized before comparison to handle cloud/sun shifts.
    """
    # Adaptive global brightness compensation
    frame_f = frame.astype(np.float32)
    bg_f    = bg.astype(np.float32)
    mean_frame = frame_f.mean() + 1e-6
    mean_bg    = bg_f.mean()    + 1e-6
    frame_adj = np.clip(frame_f * (mean_bg / mean_frame), 0, 255).astype(np.uint8)

    # Lab difference
    lab_frame = cv2.cvtColor(frame_adj, cv2.COLOR_BGR2Lab).astype(np.float32)
    lab_bg    = cv2.cvtColor(bg,        cv2.COLOR_BGR2Lab).astype(np.float32)
    diff = np.abs(lab_frame - lab_bg)

    # L-channel contributes less (lighting artefacts); a+b channels dominate
    score = 0.4 * diff[:, :, 0] + 0.8 * diff[:, :, 1] + 0.8 * diff[:, :, 2]
    score = cv2.GaussianBlur(score, (5, 5), 0)

    _, fg = cv2.threshold(score, threshold, 1, cv2.THRESH_BINARY)
    return fg.astype(np.uint8)


# ── Morphological cleanup ─────────────────────────────────────────────────────

_KERNEL_OPEN  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
_MIN_BLOB_AREA = 400   # pixels — smaller blobs treated as noise


def _clean_fg(fg: np.ndarray) -> np.ndarray:
    """Remove noise via open→close morphology and minimum-area filtering."""
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  _KERNEL_OPEN)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _KERNEL_CLOSE)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    clean = np.zeros_like(fg)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= _MIN_BLOB_AREA:
            clean[labels == lbl] = 1
    return clean


# ── Main API ──────────────────────────────────────────────────────────────────

def generate_mask(p: int, c: int, video_path: str,
                  output_dir: str = None) -> None:
    """
    Generate per-second binary change masks for a surveillance video.

    Args:
        p: Persistence threshold (seconds).  An encroachment pixel must be
           continuously present for >= p seconds before it is flagged.
        c: Cooldown period (seconds).  After c seconds from the start of a
           pixel's current foreground streak, it is considered assimilated
           into the background and is no longer flagged.
        video_path: Path to the input MP4 video.
        output_dir: Directory where mask_XXXX.png files are written.
                    Defaults to the video filename stem (e.g. "video_01" for
                    "video_01.mp4") so the folder name matches the spec.

    Output naming: output_dir/mask_0001.png … mask_NNNN.png
    Mask encoding : 255 = persistent change, 0 = background / no change.
    """
    if output_dir is None:
        output_dir = Path(video_path).stem
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_sec    = int(total_frames / fps)

    print(f"  {Path(video_path).name}: {W}x{H} @ {fps:.1f} fps | {total_sec}s | p={p}s c={c}s")

    # ── Background estimation (first 25 % of video, max 90 s) ─────────────
    bg_sec = min(90, max(30, total_sec // 4))
    bg = _build_background(cap, fps, n_seconds=bg_sec)

    # ── Per-pixel persistence state ───────────────────────────────────────
    # consecutive_fg  : consecutive seconds this pixel has been foreground
    # streak_start    : second at which the current streak began (-1 = not active)
    consecutive_fg = np.zeros((H, W), dtype=np.int32)
    streak_start   = np.full((H, W), -1, dtype=np.int32)

    samples_per_sec = max(1, min(5, int(fps)))   # frames to sample per second

    for sec in range(total_sec):
        # ── Aggregate foreground across sampled frames within this second ──
        start_f = int(sec * fps)
        end_f   = int((sec + 1) * fps)
        idxs    = np.linspace(start_f, end_f - 1, samples_per_sec, dtype=int)

        votes = np.zeros((H, W), dtype=np.int32)
        valid = 0
        for fidx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ret, frame = cap.read()
            if not ret:
                continue
            votes += _foreground_mask(frame, bg).astype(np.int32)
            valid += 1

        # Majority vote: foreground if detected in > 50 % of sampled frames
        fg_sec = (votes > valid * 0.5).astype(np.uint8) if valid else np.zeros((H, W), np.uint8)
        fg_sec = _clean_fg(fg_sec)

        # ── Update persistence counters ───────────────────────────────────
        fg_mask = fg_sec == 1
        bg_mask = ~fg_mask

        # New streak beginning
        new_streak = fg_mask & (consecutive_fg == 0)
        streak_start[new_streak] = sec

        # Advance or reset consecutive counter
        consecutive_fg[fg_mask] += 1
        consecutive_fg[bg_mask]  = 0
        streak_start[bg_mask]    = -1

        # ── Compute output mask ───────────────────────────────────────────
        # elapsed: seconds since the start of the current foreground streak
        elapsed = np.where(streak_start != -1, sec - streak_start, c)

        persistent = (
            (consecutive_fg >= p) &   # present for at least p consecutive seconds
            (elapsed < c)             # not yet assimilated
        )

        out_mask = np.zeros((H, W), dtype=np.uint8)
        out_mask[persistent] = 255

        cv2.imwrite(str(out_path / f"mask_{sec + 1:04d}.png"), out_mask)

        if sec % 60 == 0 or persistent.any():
            n_px = int(persistent.sum())
            print(f"    sec {sec + 1:4d}/{total_sec}  streak_max={consecutive_fg.max():4d}  "
                  f"flagged_px={n_px}")

    cap.release()
    print(f"  Done -> {total_sec} masks in '{output_dir}/'")


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PSCDL 2026 – generate change masks")
    parser.add_argument("video_path",             help="Path to input video")
    parser.add_argument("-p", type=int, default=60, help="Persistence threshold (s) [default: 60]")
    parser.add_argument("-c", type=int, default=90, help="Cooldown period (s) [default: 90]")
    parser.add_argument("-o", "--output_dir", default="output_masks",
                        help="Output directory [default: output_masks]")
    args = parser.parse_args()

    generate_mask(p=args.p, c=args.c, video_path=args.video_path,
                  output_dir=args.output_dir)
