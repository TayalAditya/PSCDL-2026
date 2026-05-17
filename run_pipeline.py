"""
Batch runner: processes all test videos and organises masks for submission.

Usage:
    python run_pipeline.py [--p 60] [--c 90] [--video_dir PSCDL2026_Test/test_videos]

Output structure (folder name = video filename stem, as required by spec):
    submission/
        <video_stem>/mask_0001.png … mask_NNNN.png
        …
"""

import argparse
import time
from pathlib import Path

from solution import generate_mask


def run_all(p: int, c: int, video_dir: str, submission_dir: str = "submission") -> None:
    video_dir_path = Path(video_dir)
    videos = sorted(video_dir_path.glob("*.mp4"))

    if not videos:
        raise FileNotFoundError(f"No .mp4 files found in '{video_dir}'")

    print(f"Found {len(videos)} video(s). p={p}s  c={c}s")
    print("=" * 60)

    sub_path = Path(submission_dir)
    sub_path.mkdir(parents=True, exist_ok=True)

    for i, video_path in enumerate(videos, start=1):
        t0 = time.time()
        print(f"\n[Video {i}/{len(videos)}] {video_path.name}")

        # Folder name must match the video filename stem (spec requirement)
        dest_dir = sub_path / video_path.stem

        generate_mask(p=p, c=c, video_path=str(video_path),
                      output_dir=str(dest_dir))

        elapsed = time.time() - t0
        masks   = list(dest_dir.glob("*.png"))
        print(f"  {len(masks)} masks → '{dest_dir}/'  ({elapsed:.0f}s elapsed)")

    print("\n" + "=" * 60)
    print(f"All done. Submission masks are in '{submission_dir}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSCDL 2026 – batch mask generation")
    parser.add_argument("--p",          type=int, default=60,
                        help="Persistence threshold in seconds [default: 60]")
    parser.add_argument("--c",          type=int, default=90,
                        help="Cooldown period in seconds [default: 90]")
    parser.add_argument("--video_dir",  default="PSCDL2026_Test/test_videos",
                        help="Directory containing test .mp4 files")
    parser.add_argument("--output_dir", default="submission",
                        help="Root output directory for organised masks [default: submission]")
    args = parser.parse_args()

    run_all(p=args.p, c=args.c, video_dir=args.video_dir, submission_dir=args.output_dir)
