from __future__ import annotations

import argparse
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np


@dataclass
class Component:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    area: int
    centroid: tuple[float, float]


@dataclass
class Track:
    first_seen: int
    last_seen: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    age_seen: int = 1
    missed: int = 0


def _read_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read mask: {path}")
    return ((image > 0).astype(np.uint8) * 255)


def _extract_components(mask: np.ndarray) -> list[Component]:
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    components: list[Component] = []
    total = mask.shape[0] * mask.shape[1]
    min_area = max(180, int(total * 0.000035))
    max_area = int(total * 0.03)
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if w < 3 or h < 3:
            continue
        comp_mask = np.zeros(mask.shape, dtype=np.uint8)
        comp_mask[labels == label] = 255
        components.append(
            Component(
                mask=comp_mask,
                bbox=(x, y, w, h),
                area=area,
                centroid=(float(centroids[label][0]), float(centroids[label][1])),
            )
        )
    return components


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = float((x2 - x1) * (y2 - y1))
    union = float(aw * ah + bw * bh) - inter
    return inter / union if union > 0 else 0.0


def _match_score(track: Track, comp: Component) -> float:
    iou = _bbox_iou(track.bbox, comp.bbox)
    tx, ty = track.centroid
    cx, cy = comp.centroid
    distance = float(np.hypot(tx - cx, ty - cy))
    scale = max(35.0, np.hypot(track.bbox[2], track.bbox[3]), np.hypot(comp.bbox[2], comp.bbox[3]))
    proximity = max(0.0, 1.0 - distance / (2.0 * scale))
    return 0.78 * iou + 0.22 * proximity


def _clean(mask: np.ndarray) -> np.ndarray:
    if cv2.countNonZero(mask) == 0:
        return mask
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel5, iterations=1)
    return out


def _hybrid_video(
    aggressive_dir: Path,
    conservative_dir: Path,
    output_dir: Path,
    active_window: int,
    miss_tolerance: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggressive_masks = sorted(aggressive_dir.glob("mask_*.png"))
    if not aggressive_masks:
        raise FileNotFoundError(f"No aggressive masks in {aggressive_dir}")

    first_seen_map: np.ndarray | None = None
    memory_kernel: np.ndarray | None = None
    for idx, aggressive_path in enumerate(aggressive_masks, start=1):
        conservative_path = conservative_dir / aggressive_path.name
        aggressive = _read_mask(aggressive_path)
        conservative = _read_mask(conservative_path) if conservative_path.exists() else np.zeros_like(aggressive)
        combined = cv2.bitwise_or(aggressive, conservative)
        components = _extract_components(combined)
        if first_seen_map is None:
            first_seen_map = np.full(combined.shape, -1, dtype=np.int32)
            memory_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

        out = np.zeros_like(combined)
        for comp in components:
            assert first_seen_map is not None
            comp_pixels = comp.mask > 0
            known = first_seen_map[comp_pixels]
            known = known[known >= 0]
            if known.size >= max(20, int(comp.area * 0.08)):
                born = int(np.percentile(known, 20))
            else:
                born = idx

            if 0 <= idx - born < active_window:
                out = cv2.bitwise_or(out, comp.mask)

            assert memory_kernel is not None
            memory = cv2.dilate(comp.mask, memory_kernel, iterations=1) > 0
            unknown = memory & (first_seen_map < 0)
            first_seen_map[unknown] = born
            older = memory & (first_seen_map > born)
            first_seen_map[older] = born

        out = _clean(out)
        cv2.imwrite(str(output_dir / aggressive_path.name), out)


def build_hybrid(
    aggressive_zip: Path,
    conservative_zip: Path,
    base_package_zip: Path,
    output_zip: Path,
    active_window: int,
    miss_tolerance: int,
) -> None:
    with tempfile.TemporaryDirectory(prefix="pscdl_hybrid_") as td:
        root = Path(td)
        aggressive_root = root / "aggressive"
        conservative_root = root / "conservative"
        output_root = root / "hybrid"
        package_root = root / "package"

        with ZipFile(aggressive_zip) as zf:
            zf.extractall(aggressive_root)
        with ZipFile(conservative_zip) as zf:
            zf.extractall(conservative_root)
        with ZipFile(base_package_zip) as zf:
            zf.extractall(package_root)

        for video_idx in range(1, 6):
            aggressive_dir = aggressive_root / f"test_{video_idx}"
            if not aggressive_dir.exists():
                aggressive_dir = aggressive_root / f"Video{video_idx}"
            conservative_dir = conservative_root / f"Video{video_idx}"
            out_dir = output_root / f"Video{video_idx}"
            _hybrid_video(aggressive_dir, conservative_dir, out_dir, active_window, miss_tolerance)

            package_video_dir = package_root / f"Video{video_idx}"
            if package_video_dir.exists():
                shutil.rmtree(package_video_dir)
            shutil.copytree(out_dir, package_video_dir)

        if output_zip.exists():
            output_zip.unlink()
        with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as zf:
            for path in sorted(package_root.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(package_root).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggressive-zip", type=Path, default=Path("submission_final.zip"))
    parser.add_argument("--conservative-zip", type=Path, default=Path("strict_pipeline/PSCDL2026_AdityaTayal_Submission.zip"))
    parser.add_argument("--base-package-zip", type=Path, default=Path("strict_pipeline/PSCDL2026_AdityaTayal_Submission.zip"))
    parser.add_argument("--output-zip", type=Path, default=Path("strict_pipeline/PSCDL2026_AdityaTayal_Hybrid_Submission.zip"))
    parser.add_argument("--active-window", type=int, default=34)
    parser.add_argument("--miss-tolerance", type=int, default=3)
    args = parser.parse_args()
    build_hybrid(
        aggressive_zip=args.aggressive_zip,
        conservative_zip=args.conservative_zip,
        base_package_zip=args.base_package_zip,
        output_zip=args.output_zip,
        active_window=args.active_window,
        miss_tolerance=args.miss_tolerance,
    )


if __name__ == "__main__":
    main()
