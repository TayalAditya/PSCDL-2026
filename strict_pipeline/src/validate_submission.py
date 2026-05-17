from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def _video_index(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return 0


def _video_info(path: Path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0:
        fps = 25.0
    seconds = max(1, int(round(frame_count / fps))) if frame_count > 0 else 1
    return width, height, seconds


def validate(videos_dir: Path, masks_root: Path) -> None:
    videos = sorted(videos_dir.glob("test_*.mp4"), key=_video_index)
    if not videos:
        raise FileNotFoundError(f"No test videos found in {videos_dir}")

    errors: list[str] = []
    for idx, video in enumerate(videos, start=1):
        width, height, expected = _video_info(video)
        folder = masks_root / video.stem
        masks = sorted(folder.glob("mask_*.png"))
        print(f"{video.stem}: expected={expected}, found={len(masks)}, resolution={width}x{height}")

        if len(masks) != expected:
            errors.append(f"{video.stem}: expected {expected} masks, found {len(masks)}")

        for expected_idx, mask_path in enumerate(masks, start=1):
            if mask_path.name != f"mask_{expected_idx:04d}.png":
                errors.append(f"{video.stem}: bad filename order at {mask_path.name}")
                break

            image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                errors.append(f"{video.stem}: unreadable mask {mask_path}")
                break
            if image.shape != (height, width):
                errors.append(f"{video.stem}: bad shape {mask_path.name}: {image.shape}, expected {(height, width)}")
                break

            values = np.unique(image)
            if not np.all(np.isin(values, [0, 255])):
                errors.append(f"{video.stem}: non-binary values in {mask_path.name}: {values[:10]}")
                break

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Validation passed.")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", type=Path, default=root / "input" / "PSCDL2026_Test" / "test_videos")
    parser.add_argument("--masks-root", type=Path, default=root / "output")
    args = parser.parse_args()
    validate(args.videos_dir, args.masks_root)


if __name__ == "__main__":
    main()
