# image/rename.py

"""
Rename images, captions, and subdirectories to sequential zero-padded numbers.

Two-pass approach to avoid name collisions with existing numbered artifacts:
  Pass 1: rename to _<num>.* / _<num>/
  Pass 2: strip the underscore prefix -> <num>.* / <num>/

Usage:
    py -m image.rename -i "/path/to/input"
    py -m image.rename -i "/path/to/input" --dry-run
    py -m image.rename -i "/path/to/input" --padding 4
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

IMAGE_EXTS: set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rename images, captions, and subdirectories to sequential numbers."
    )
    p.add_argument(
        "-i", "--input", required=True, type=Path,
        help="Directory containing images to rename.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print actions without executing.",
    )
    p.add_argument(
        "--padding", type=int, default=6,
        help="Zero-pad width (default: 6).",
    )
    return p.parse_args()


def _safe_rename(src: Path, dst: Path, *, dry_run: bool) -> None:
    """Rename src to dst. Abort on collision unless src == dst."""
    if src.resolve() == dst.resolve():
        return
    if dst.exists():
        sys.exit(f"ERROR: target already exists: {dst}")
    print(f"  {src.name} -> {dst.name}")
    if not dry_run:
        src.rename(dst)


def _strip_underscore(root: Path, *, dry_run: bool) -> None:
    """Pass 2: rename _<num>[...] -> <num>[...] for files and directories."""
    # Collect entries prefixed with underscore (sorted for determinism)
    under_entries = sorted(
        [e for e in root.iterdir() if e.name.startswith("_")],
        key=lambda p: p.name,
    )
    for entry in under_entries:
        dst = root / entry.name[1:]  # drop leading "_"
        if dst.exists():
            # Both exist — only a problem if they're different paths
            if entry.resolve() != dst.resolve():
                sys.exit(f"ERROR: target already exists: {dst}")
            continue
        print(f"  {entry.name} -> {dst.name}")
        if not dry_run:
            entry.rename(dst)


def main() -> None:
    args = parse_args()
    root: Path = args.input.expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: {root} is not a directory.")

    padding = args.padding
    dry_run = args.dry_run

    # --- Collect top-level entries (skip hidden) ---
    all_entries = sorted(
        [e for e in root.iterdir() if not e.name.startswith(".") and not e.name.startswith("_")],
        key=lambda p: p.name,
    )

    # --- Group by stem ---
    # groups[stem] = {"image": Path|None, "caption": Path|None, "subdir": Path|None}
    groups: dict[str, dict[str, Path | None]] = {}

    for entry in all_entries:
        if not entry.is_file():
            continue
        stem = entry.stem
        suffix = entry.suffix.lower()
        if suffix in IMAGE_EXTS:
            groups.setdefault(stem, {"image": None, "caption": None, "subdir": None})["image"] = entry
        elif suffix == ".txt":
            groups.setdefault(stem, {"image": None, "caption": None, "subdir": None})["caption"] = entry

    # Match subdirectories to groups by name
    for entry in all_entries:
        if entry.is_dir():
            stem = entry.name
            if stem in groups and groups[stem]["subdir"] is None:
                groups[stem]["subdir"] = entry

    # Keep only groups anchored by an image
    anchored: dict[str, dict[str, Path | None]] = {
        stem: g for stem, g in groups.items() if g["image"] is not None
    }

    orphans = [g["caption"] for g in groups.values() if g["image"] is None and g["caption"] is not None]

    if not anchored:
        sys.exit("No images found to rename.")

    sorted_stems = sorted(anchored.keys())
    total_anchors = 0
    total_captions = 0
    total_subdirs = 0
    total_subdir_images = 0

    # --- Pass 1: rename to _<num> ---
    for i, stem in enumerate(reversed(sorted_stems)):
        num = len(sorted_stems) - i  # reverse order -> highest number first
        group = anchored[stem]
        num_str = str(num).zfill(padding)

        # 1. Anchor image -> _<num>.<ext>
        src_img = group["image"]
        assert src_img is not None
        dst_img = root / f"_{num_str}{src_img.suffix.lower()}"

        if src_img.resolve() != dst_img.resolve():
            if dst_img.exists():
                sys.exit(f"ERROR: target already exists: {dst_img}")
            print(f"{src_img.name} -> {dst_img.name}")
            if not dry_run:
                src_img.rename(dst_img)
        total_anchors += 1

        # 2. Caption -> _<num>.txt
        src_cap = group["caption"]
        if src_cap is not None:
            dst_cap = root / f"_{num_str}.txt"
            if src_cap.resolve() != dst_cap.resolve():
                if dst_cap.exists():
                    sys.exit(f"ERROR: target already exists: {dst_cap}")
                print(f"  {src_cap.name} -> {dst_cap.name}")
                if not dry_run:
                    src_cap.rename(dst_cap)
            total_captions += 1

        # 3. Subdirectory and its contents
        src_sub = group["subdir"]
        if src_sub is not None:
            dst_sub = root / f"_{num_str}"

            # Rename images inside the subdirectory first (while subdir still has its old name)
            sub_entries = sorted(
                [e for e in src_sub.iterdir() if not e.name.startswith(".")],
                key=lambda p: p.name,
            )
            sub_images = [
                e for e in sub_entries
                if e.is_file() and e.suffix.lower() in IMAGE_EXTS
            ]

            for j, img in enumerate(sub_images):
                img_num = str(j + 1).zfill(padding)
                dst_sub_img = src_sub / f"{img_num}{img.suffix.lower()}"
                if img.resolve() != dst_sub_img.resolve():
                    if dst_sub_img.exists():
                        sys.exit(f"ERROR: target already exists: {dst_sub_img}")
                    print(f"  {img.name} -> _{num_str}/{dst_sub_img.name}")
                    if not dry_run:
                        img.rename(dst_sub_img)
                total_subdir_images += 1

            # Rename the subdirectory itself -> _<num>
            if src_sub.resolve() != dst_sub.resolve():
                if dst_sub.exists():
                    sys.exit(f"ERROR: target already exists: {dst_sub}")
                print(f"  {src_sub.name}/ -> {dst_sub.name}/")
                if not dry_run:
                    src_sub.rename(dst_sub)
            total_subdirs += 1

    # --- Pass 2: strip underscore prefix ---
    print()
    print("Pass 2: stripping underscore prefix...")
    _strip_underscore(root, dry_run=dry_run)

    # Summary
    parts = [
        f"Renamed {total_anchors} anchor image{'' if total_anchors == 1 else 's'}",
    ]
    if total_captions:
        parts.append(f"{total_captions} caption{'' if total_captions == 1 else 's'}")
    if total_subdirs:
        parts.append(f"{total_subdirs} subdirector{'' if total_subdirs == 1 else 'ies'}")
    if total_subdir_images:
        parts.append(f"{total_subdir_images} subdir image{'' if total_subdir_images == 1 else 's'}")
    print()
    print(", ".join(parts) + ".")

    if orphans:
        print(f"Skipped {len(orphans)} .txt file{'' if len(orphans) == 1 else 's'} with no paired image.")


if __name__ == "__main__":
    main()
