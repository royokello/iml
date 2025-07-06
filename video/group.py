"""
group.py

Example:
    py -m video.group --root "/video" --ffprobe "/programs/ffprobe.exe" --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

VIDEO_EXTS: set[str] = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi",
    ".flv", ".wmv", ".mpeg", ".mpg",
}


def get_dimensions(path: Path, ffprobe: str) -> tuple[int | None, int | None]:
    """Return (width, height) of *path* via *ffprobe*, or (None, None) on error."""
    try:
        out = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0:s=x",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        w, h = out.split("x")
        return int(w), int(h)
    except Exception as exc:  # noqa: BLE001
        print(f"– skipped (no dims): {path} ({exc})")
        return None, None


def size_bucket(shorter: int, thresholds: list[int]) -> int:
    """Return the first *threshold* ≥ *shorter*, else the final threshold."""
    for t in thresholds:
        if shorter <= t:
            return t
    return thresholds[-1]


def build_prefix(path: Path, root: Path) -> str:
    """Return a filename prefix like ``video_cats_`` reflecting directory depth."""
    rel_parent = path.parent.relative_to(root)
    parts: list[str] = [root.name]
    if rel_parent != Path('.'):
        parts.extend(rel_parent.parts)
    return "__".join(parts) + "__"


def organise(root: Path, thresholds: list[int], ffprobe: str, *, dry_run: bool) -> None:
    """Scan *root*, grouping videos by orientation + size. Respects *dry_run*."""
    for src in root.rglob("*"):
        if not src.is_file() or src.suffix.lower() not in VIDEO_EXTS:
            continue

        dims = get_dimensions(src, ffprobe)
        if not dims or dims[0] is None or dims[1] is None:
            continue
        w, h = dims

        orientation = "vertical" if h > w else "horizontal"
        bucket = size_bucket(min(w, h), thresholds)

        dst_dir = root / f"_{orientation}" / f"_{bucket}"
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst = dst_dir / f"{build_prefix(src, root)}{src.name}"
        if dst.exists():
            print(f"– duplicate, skipping: {dst}")
            continue

        print(f"{src}  →  {dst}")
        if not dry_run:
            shutil.move(src, dst)


def parse_args() -> argparse.Namespace:  # noqa: D401 – argparse style
    p = argparse.ArgumentParser(description="Group videos by orientation and size.")
    p.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Directory to scan (and where groups are created).",
    )
    p.add_argument(
        "--sizes",
        nargs="*",
        type=int,
        metavar="N",
        default=[256, 384, 512, 768, 1024],
        help="Optional short‑side thresholds (default: 256 384 512 768 1024).",
    )
    p.add_argument(
        "--ffprobe",
        required=True,
        type=Path,
        help="Full path to ffprobe executable.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned moves without changing any files.",
    )
    return p.parse_args()


def main() -> None:  # noqa: D401 – entry point style
    args = parse_args()

    root: Path = args.root.expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: {root} is not a directory.")

    ffprobe_path: Path = args.ffprobe.expanduser().resolve()
    if not ffprobe_path.is_file():
        sys.exit(f"ERROR: ffprobe not found at {ffprobe_path}")

    thresholds = sorted(set(args.sizes))
    organise(root, thresholds, str(ffprobe_path), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
