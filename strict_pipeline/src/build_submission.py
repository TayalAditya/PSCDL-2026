from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    package_dir = root / "submission_package"
    output_dir = root / "output"
    zip_path = root / "PSCDL2026_AdityaTayal_Submission.zip"

    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)

    for video_dir in sorted(output_dir.glob("Video*")):
        if video_dir.is_dir():
            shutil.copytree(video_dir, package_dir / video_dir.name)

    for name in ["solution.py", "run_final.py", "validate_submission.py"]:
        copy_file(root / "src" / name, package_dir / name)

    for name in ["requirements.txt", "README.md", "TEAM_INFO.md", "TECHNICAL_REPORT.md"]:
        copy_file(root / name, package_dir / name)

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package_dir))

    print(f"Created {zip_path}")


if __name__ == "__main__":
    main()
