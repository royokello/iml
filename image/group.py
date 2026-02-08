# image/group.py
"""
Example:
    py -m image.group --input "/images" --orientation height --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

IMAGE_EXTS: set[str] = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
}


def get_dimensions(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as im:
            return im.size
    except Exception as exc:
        print(f"– skipped (no dims): {path} ({exc})")
        return None, None


def size_bucket(value: int, thresholds: list[int]) -> int:
    for t in thresholds:
        if value < t:
            return t
    return thresholds[-1]


def organise(root: Path, thresholds: list[int], orientation: str, *, dry_run: bool) -> None:
    for src in root.rglob("*"):
        if not src.is_file() or src.suffix.lower() not in IMAGE_EXTS:
            continue

        dims = get_dimensions(src)
        if not dims or dims[0] is None or dims[1] is None:
            continue
        w, h = dims

        if orientation == "width":
            measure = w
            dst_dir = root / f"_{size_bucket(measure, thresholds)}"
        elif orientation == "height":
            measure = h
            dst_dir = root / f"_{size_bucket(measure, thresholds)}"
        else:
            measure = max(w, h)
            orient_label = "vertical" if h > w else "horizontal"
            dst_dir = root / f"_{orient_label}" / f"_{size_bucket(measure, thresholds)}"

        dst_dir.mkdir(parents=True, exist_ok=True)

        dst = dst_dir / src.name

        if dst.exists():
            print(f"– duplicate, skipping: {dst}")
            continue

        print(f"{src}  →  {dst}")
        if not dry_run:
            shutil.move(str(src), str(dst))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Group images by chosen dimension and size.")
    p.add_argument("-i", "--input", required=True, type=Path, help="Directory to scan (and where groups are created).")
    p.add_argument("--sizes", nargs="*", type=int, metavar="N", default=[256, 384, 512, 768, 1024])
    p.add_argument("--orientation", choices=["width", "height", "longest"], default="height")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root: Path = args.input.expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: {root} is not a directory.")

    thresholds = sorted(set(args.sizes))
    organise(root, thresholds, args.orientation, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
