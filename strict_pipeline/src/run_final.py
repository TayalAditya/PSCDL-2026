from __future__ import annotations

import os
from pathlib import Path

from solution import generate_mask


def _video_index(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return 0


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    videos_dir = root / "input" / "PSCDL2026_Test" / "test_videos"
    output_root = root / "output"
    output_root.mkdir(parents=True, exist_ok=True)

    videos = sorted(videos_dir.glob("test_*.mp4"), key=_video_index)
    if not videos:
        raise FileNotFoundError(f"No test videos found in {videos_dir}")

    for idx, video_path in enumerate(videos, start=1):
        out_dir = output_root / video_path.stem
        os.environ["PSCDL_OUTPUT_DIR"] = str(out_dir)
        print(f"[{idx}/{len(videos)}] {video_path.name} -> {out_dir}")
        generate_mask(p=60, c=90, video_path=str(video_path))


if __name__ == "__main__":
    main()
