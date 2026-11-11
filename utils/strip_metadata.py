# utils/strip_metadata.py

"""
Write metadata-free copies of PNG images, named by content hash, for sharing.

Originals are never changed. Every output file is named with the first 12 hex
characters of the SHA-256 of the clean file, so names reveal nothing and the
same image always gets the same name. Existing output files are never
overwritten.

Only the colour profile and transparency are kept. All text chunks, EXIF, and
other metadata are dropped.

Usage:
    python -m utils.strip_metadata -i "path/to/image.png" -o "path/to/share"
    python -m utils.strip_metadata -i "path/to/folder" -o "path/to/share"
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

from PIL import Image

KEPT_INFO_KEYS = ("icc_profile", "transparency")
NAME_LENGTH = 12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write metadata-free, hash-named copies of PNG images.")
    p.add_argument("-i", "--input", required=True, type=Path, help="A PNG file, or a folder of PNG files (top level only).")
    p.add_argument("-o", "--output", required=True, type=Path, help="Folder for the clean copies. Created if missing.")
    return p.parse_args()


def collect_inputs(source: Path) -> tuple[list[Path], list[Path]]:
    if source.is_file():
        candidates = [source]
    else:
        candidates = sorted(e for e in source.iterdir() if e.is_file() and not e.name.startswith("."))
    pngs = [c for c in candidates if c.suffix.lower() == ".png"]
    skipped = [c for c in candidates if c.suffix.lower() != ".png"]
    return pngs, skipped


def strip_png(path: Path) -> tuple[bytes, int]:
    """Return the clean PNG bytes and the number of metadata entries dropped."""
    with Image.open(path) as img:
        if img.format != "PNG":
            raise ValueError(f"not a PNG file (format: {img.format})")
        img.load()
        dropped = sum(1 for key in img.info if key not in KEPT_INFO_KEYS)
        img.info = {key: value for key, value in img.info.items() if key in KEPT_INFO_KEYS}
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
    return buffer.getvalue(), dropped


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not source.exists():
        sys.exit(f"ERROR: {source} does not exist.")
    input_dir = source if source.is_dir() else source.parent
    if output == input_dir:
        sys.exit(f"ERROR: --output must not be the input folder: {output}")

    pngs, skipped = collect_inputs(source)
    if not pngs:
        sys.exit(f"ERROR: no PNG files found in {source}")

    output.mkdir(parents=True, exist_ok=True)

    written = 0
    present = 0
    dropped_total = 0
    failed = 0
    for path in pngs:
        try:
            data, dropped = strip_png(path)
        except Exception as exc:
            print(f"FAILED  {path.name}: {exc}")
            failed += 1
            continue
        name = hashlib.sha256(data).hexdigest()[:NAME_LENGTH] + ".png"
        target = output / name
        dropped_total += dropped
        if target.exists():
            print(f"{path.name} -> {name} (already present)")
            present += 1
            continue
        with open(target, "xb") as handle:
            handle.write(data)
        print(f"{path.name} -> {name}")
        written += 1

    for path in skipped:
        print(f"SKIPPED {path.name} (not a PNG)")

    print(
        f"\nWrote {written}, already present {present}, failed {failed}, skipped {len(skipped)}. "
        f"Removed {dropped_total} metadata entries. Output: {output}"
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
