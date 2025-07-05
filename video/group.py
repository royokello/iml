"""
group.py
```
py -m video.group.py --root "/video" --orientation --sizes 384 512 768 --ffprobe "/programs/ffprobe.exe"
```
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

VIDEO_EXTS = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi",
    ".flv", ".wmv", ".mpeg", ".mpg",
}


def get_dimensions(path: Path, ffprobe: str) -> tuple[int | None, int | None]:
    try:
        out = subprocess.run(
            [
                ffprobe, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
                str(path),
            ],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        w, h = out.split("x")
        return int(w), int(h)
    except Exception as exc:
        print(f"– skipped (no dims): {path} ({exc})")
        return None, None


def size_bucket(shorter: int, thresholds: list[int]) -> str:
    for t in thresholds:
        if shorter <= t:
            return f"<= {t}"
    return f"> {thresholds[-1]}"


def build_prefix(path: Path, root: Path) -> str:
    rel_parent = path.parent.relative_to(root)
    return "" if rel_parent == Path(".") else "_".join(rel_parent.parts) + "_"


def organise(root: Path, by_orientation: bool, thresholds: list[int], ffprobe: str) -> None:
    for src in root.rglob("*"):
        if not src.is_file() or src.suffix.lower() not in VIDEO_EXTS:
            continue

        w, h = get_dimensions(src, ffprobe)
        if not w or not h:
            continue

        orientation = "vertical" if h > w else "horizontal"
        bucket = size_bucket(min(w, h), thresholds)

        dst_dir = root / (orientation if by_orientation else "") / bucket
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst = dst_dir / f"{build_prefix(src, root)}{src.name}"
        if dst.exists():
            print(f"– duplicate, skipping: {dst}")
            continue

        print(f"{src}  →  {dst}")
        shutil.move(src, dst)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Group videos by orientation and size.")
    p.add_argument("--root", required=True, type=Path,
                   help="Directory to scan (and where groups are created).")
    p.add_argument("--orientation", action="store_true",
                   help="First split into horizontal / vertical.")
    p.add_argument("--sizes", nargs="+", type=int, required=True, metavar="N",
                   help="Short-side thresholds, e.g. --sizes 384 512 768.")
    p.add_argument("--ffprobe", required=True, type=Path,
                   help="Full path to ffprobe executable.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: {root} is not a directory.")

    ffprobe_path = args.ffprobe.expanduser().resolve()
    if not ffprobe_path.is_file():
        sys.exit(f"ERROR: ffprobe not found at {ffprobe_path}")

    organise(root, args.orientation, sorted(set(args.sizes)), str(ffprobe_path))


if __name__ == "__main__":
    main()
