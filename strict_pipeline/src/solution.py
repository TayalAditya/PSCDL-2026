from __future__ import annotations

import math
import os
import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    seconds: int


@dataclass
class Component:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    area: int
    centroid: tuple[float, float]


@dataclass
class Track:
    track_id: int
    first_seen: int
    last_seen: int
    bbox: tuple[int, int, int, int]
    support: np.ndarray
    current_mask: np.ndarray
    hits: int = 1
    missed: int = 0
    updated_at: int = -1


class SecondSampler:
    """Sequentially reads one frame for each requested integer second."""

    def __init__(self, video_path: str | Path, fps: float, frame_count: int):
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        self.fps = fps
        self.frame_count = frame_count
        self.current_index = 0
        self.last_frame: np.ndarray | None = None

    def close(self) -> None:
        self.cap.release()

    def read_second(self, second: int) -> np.ndarray:
        target = int(round(second * self.fps))
        if self.frame_count > 0:
            target = min(max(target, 0), self.frame_count - 1)

        if target < self.current_index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            self.current_index = target

        while self.current_index < target:
            ok = self.cap.grab()
            if not ok:
                if self.last_frame is not None:
                    return self.last_frame.copy()
                raise RuntimeError(f"Could not seek to second {second}")
            self.current_index += 1

        ok, frame = self.cap.read()
        if not ok:
            if self.last_frame is not None:
                return self.last_frame.copy()
            raise RuntimeError(f"Could not read second {second}")

        self.current_index += 1
        self.last_frame = frame
        return frame


def _video_info(video_path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if fps <= 0:
        fps = 25.0
    duration = frame_count / fps if frame_count > 0 else 0.0
    seconds = max(1, int(round(duration))) if duration > 0 else 1
    return VideoInfo(width, height, fps, frame_count, duration, seconds)


def _baseline_length(num_seconds: int, p: int) -> int:
    if num_seconds <= 1:
        return 1
    # The challenge guarantees an initial clean scene. Keep this short enough
    # for early insertions and long enough for a robust median baseline.
    candidate = max(5, min(20, max(1, p) // 3))
    return max(1, min(candidate, num_seconds))


def _build_baseline(video_path: str | Path, info: VideoInfo, seconds: int) -> np.ndarray:
    sampler = SecondSampler(video_path, info.fps, info.frame_count)
    frames: list[np.ndarray] = []
    try:
        for sec in range(seconds):
            frames.append(sampler.read_second(sec))
    finally:
        sampler.close()

    if not frames:
        raise RuntimeError("No frames available for baseline construction")
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def _extract_components(mask: np.ndarray, min_area: int, max_area: int) -> list[Component]:
    if cv2.countNonZero(mask) == 0:
        return []

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    components: list[Component] = []
    image_height, image_width = mask.shape
    min_fill_ratio = float(os.environ.get("PSCDL_MIN_FILL_RATIO", "0.38"))
    max_aspect_ratio = float(os.environ.get("PSCDL_MAX_ASPECT_RATIO", "3.8"))
    top_ignore_fraction = float(os.environ.get("PSCDL_TOP_IGNORE_FRACTION", "0.14"))
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < min_area or area > max_area:
            continue
        if width < 3 or height < 3:
            continue
        fill_ratio = area / float(width * height)
        aspect_ratio = max(width / float(height), height / float(width))
        if fill_ratio < min_fill_ratio or aspect_ratio > max_aspect_ratio:
            continue
        if y + height / 2.0 < image_height * top_ignore_fraction:
            continue
        if x <= 1 or y <= 1 or x + width >= image_width - 1 or y + height >= image_height - 1:
            continue
        comp_mask = np.zeros(mask.shape, dtype=np.uint8)
        comp_mask[labels == label] = 255
        components.append(
            Component(
                mask=comp_mask,
                bbox=(x, y, width, height),
                area=area,
                centroid=(float(centroids[label][0]), float(centroids[label][1])),
            )
        )
    return components


def _filter_components(mask: np.ndarray, min_area: int, max_area: int) -> np.ndarray:
    filtered = np.zeros(mask.shape, dtype=np.uint8)
    for component in _extract_components(mask, min_area, max_area):
        filtered = cv2.bitwise_or(filtered, component.mask)
    return filtered


def _foreground_mask(
    frame: np.ndarray,
    baseline_lab: np.ndarray,
    baseline_l_sample_median: float,
    min_area: int,
    max_area: int,
    allow_luma_only: bool = True,
) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.int16)
    base = baseline_lab.astype(np.int16)

    l_shift = float(np.median(lab[::16, ::16, 0])) - baseline_l_sample_median
    corrected_l = np.clip(lab[:, :, 0].astype(np.float32) - l_shift, 0, 255).astype(np.int16)
    l_diff = np.abs(corrected_l - base[:, :, 0])
    a_diff = np.abs(lab[:, :, 1] - base[:, :, 1])
    b_diff = np.abs(lab[:, :, 2] - base[:, :, 2])
    chroma_diff = np.maximum(a_diff, b_diff)

    score = np.maximum(l_diff, (chroma_diff * 3) // 2).astype(np.uint8)
    sample = score[::8, ::8].astype(np.float32)
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    threshold = int(max(22, min(58, median + 6.0 * mad)))

    color_change = chroma_diff > 8
    if allow_luma_only:
        strong_luma_change = l_diff > 32
    else:
        strong_luma_change = (l_diff > 35) & (chroma_diff > 4)
    mask = ((score > threshold) & (color_change | strong_luma_change)).astype(np.uint8) * 255

    total_pixels = mask.shape[0] * mask.shape[1]
    if cv2.countNonZero(mask) > int(total_pixels * 0.18):
        stricter = int(max(threshold + 8, np.percentile(score[::4, ::4], 99.2)))
        if allow_luma_only:
            mask = ((score > stricter) & ((chroma_diff > 10) | (l_diff > 45))).astype(np.uint8) * 255
        else:
            mask = ((score > stricter) & ((chroma_diff > 10) | ((l_diff > 45) & (chroma_diff > 5)))).astype(np.uint8) * 255

    if cv2.countNonZero(mask) == 0:
        return mask

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=2)
    return _filter_components(mask, min_area, max_area)


def _prepare_output_dir() -> Path:
    output_dir = Path(os.environ.get("PSCDL_OUTPUT_DIR", "output_masks"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob("mask_*.png"):
        existing.unlink()
    return output_dir


def _clean_active_mask(mask: np.ndarray, min_area: int, max_area: int) -> np.ndarray:
    if cv2.countNonZero(mask) == 0:
        return mask
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel5, iterations=1)
    return _filter_components(cleaned, min_area, max_area)


def _bbox_intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def _bbox_match_score(track: Track, component: Component) -> float:
    inter = _bbox_intersection(track.bbox, component.bbox)
    tx, ty, tw, th = track.bbox
    cx, cy, cw, ch = component.bbox
    track_area = tw * th
    comp_area = cw * ch
    union = track_area + comp_area - inter
    iou = inter / union if union > 0 else 0.0
    overlap_min = inter / min(track_area, comp_area) if min(track_area, comp_area) > 0 else 0.0

    tcenter = (tx + tw / 2.0, ty + th / 2.0)
    ccenter = component.centroid
    distance = math.hypot(tcenter[0] - ccenter[0], tcenter[1] - ccenter[1])
    scale = max(50.0, math.hypot(max(tw, cw), max(th, ch)))
    proximity = max(0.0, 1.0 - distance / (1.5 * scale))

    if iou > 0.02 or overlap_min > 0.12:
        return 0.70 * iou + 0.25 * overlap_min + 0.05 * proximity
    if proximity > 0.55 and distance < 100.0:
        return 0.04 * proximity
    return 0.0


def generate_mask(p: int, c: int, video_path: str) -> None:
    """Generate one binary persistent-change mask per video second.

    Required PSCDL API. By default masks are written to ./output_masks.
    Set PSCDL_OUTPUT_DIR to redirect outputs without changing this signature.
    """

    if p < 0 or c < 0:
        raise ValueError("p and c must be non-negative seconds")

    info = _video_info(video_path)
    output_dir = _prepare_output_dir()
    baseline_seconds = _baseline_length(info.seconds, p)
    baseline = _build_baseline(video_path, info, baseline_seconds)
    baseline_blurred = cv2.GaussianBlur(baseline, (5, 5), 0)
    baseline_lab = cv2.cvtColor(baseline_blurred, cv2.COLOR_BGR2LAB)
    baseline_l_sample_median = float(np.median(baseline_lab[::16, ::16, 0]))

    total_pixels = info.width * info.height
    min_area = max(300, int(total_pixels * 0.00010))
    strict_min_area = max(700, int(total_pixels * 0.00018))
    final_min_area = max(350, int(total_pixels * 0.00012))
    max_area = max(1, int(total_pixels * 0.018))
    miss_tolerance = 2
    first_seen_luma = np.full((info.height, info.width), -1, dtype=np.int32)
    missing_luma = np.zeros((info.height, info.width), dtype=np.uint8)
    first_seen_strict = np.full((info.height, info.width), -1, dtype=np.int32)
    missing_strict = np.zeros((info.height, info.width), dtype=np.uint8)
    blank = np.zeros((info.height, info.width), dtype=np.uint8)

    def update_persistence(foreground: np.ndarray, first_seen: np.ndarray, missing: np.ndarray, sec: int) -> np.ndarray:
        seen = foreground > 0
        tracked = first_seen >= 0

        if sec >= baseline_seconds + 3:
            new_pixels = seen & (~tracked)
            first_seen[new_pixels] = sec
        missing[seen] = 0

        missed_pixels = (~seen) & tracked
        if np.any(missed_pixels):
            missing[missed_pixels] = np.minimum(missing[missed_pixels] + 1, 255)
            expired = missed_pixels & (missing > miss_tolerance)
            first_seen[expired] = -1
            missing[expired] = 0

        tracked = first_seen >= 0
        age = sec - first_seen
        active = seen & tracked & (age >= p) & (age < c)
        return active.astype(np.uint8) * 255

    iterable = range(info.seconds)
    if tqdm is not None:
        iterable = tqdm(iterable, desc=f"Processing {Path(video_path).name}", unit="sec")

    sampler = SecondSampler(video_path, info.fps, info.frame_count)
    try:
        for sec in iterable:
            frame = sampler.read_second(sec)
            if frame.shape[1] != info.width or frame.shape[0] != info.height:
                frame = cv2.resize(frame, (info.width, info.height), interpolation=cv2.INTER_LINEAR)

            if sec < baseline_seconds or c <= p:
                mask = blank
            else:
                luma_foreground = _foreground_mask(
                    frame, baseline_lab, baseline_l_sample_median, min_area, max_area, allow_luma_only=True
                )
                strict_foreground = _foreground_mask(
                    frame, baseline_lab, baseline_l_sample_median, strict_min_area, max_area, allow_luma_only=False
                )
                luma_active = update_persistence(luma_foreground, first_seen_luma, missing_luma, sec)
                strict_active = update_persistence(strict_foreground, first_seen_strict, missing_strict, sec)
                mask = cv2.bitwise_or(luma_active, strict_active)
                mask = _clean_active_mask(mask, final_min_area, max_area)

            cv2.imwrite(str(output_dir / f"mask_{sec + 1:04d}.png"), mask)
    finally:
        sampler.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PSCDL persistent change masks for one video.")
    parser.add_argument("video_path", help="Path to the input video")
    parser.add_argument("--p", type=int, default=60, help="Persistence threshold in seconds")
    parser.add_argument("--c", type=int, default=90, help="Cooldown period in seconds")
    parser.add_argument("--output-dir", default="output_masks", help="Directory for mask_0001.png outputs")
    args = parser.parse_args()

    os.environ["PSCDL_OUTPUT_DIR"] = args.output_dir
    generate_mask(p=args.p, c=args.c, video_path=args.video_path)


if __name__ == "__main__":
    main()
