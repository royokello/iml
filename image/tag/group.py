from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

IMAGE_EXTS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
)


def parse_caption(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def tag_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return len(parse_caption(text))


def index_images(root: Path) -> dict[Path, dict[str, Path]]:
    index: dict[Path, dict[str, Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        by_stem = index.setdefault(path.parent, {})
        existing = by_stem.get(path.stem)
        if existing is None or path.suffix.lower() < existing.suffix.lower():
            by_stem[path.stem] = path
    return index


def iter_pairs(root: Path) -> list[tuple[Path, Path]]:
    image_index = index_images(root)
    pairs: list[tuple[Path, Path]] = []
    for caption_path in root.rglob("*.txt"):
        if not caption_path.is_file():
            continue
        image_path = image_index.get(caption_path.parent, {}).get(caption_path.stem)
        if image_path is None:
            continue
        pairs.append((image_path, caption_path))
    return pairs


def move_pair(image_path: Path, caption_path: Path, dest_dir: Path) -> bool:
    if image_path.parent == dest_dir and caption_path.parent == dest_dir:
        print(f"skip (already grouped): {image_path.name}")
        return False

    dst_image = dest_dir / image_path.name
    dst_caption = dest_dir / caption_path.name
    if dst_image.exists() or dst_caption.exists():
        print(f"skip (exists): {image_path.name}")
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(image_path), str(dst_image))
    shutil.move(str(caption_path), str(dst_caption))
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Group image/caption pairs by caption tag count.")
    p.add_argument("-i", "--input", required=True, type=Path, help="Directory with images + captions.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = args.input.expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: {root} is not a directory.")

    pairs = iter_pairs(root)
    if not pairs:
        print(f"[group] no image/caption pairs found under {root}")
        return

    counts_by_pair: list[tuple[Path, Path, int]] = []
    bucket_sizes: Counter[int] = Counter()
    for image_path, caption_path in pairs:
        count = tag_count(caption_path)
        counts_by_pair.append((image_path, caption_path, count))
        bucket_sizes[count] += 1

    moved = 0
    for image_path, caption_path, count in counts_by_pair:
        try:
            dest_dir = root / f"{count}_{bucket_sizes[count]}"
            if move_pair(image_path, caption_path, dest_dir):
                moved += 1
                print(f"move ({count}_{bucket_sizes[count]}): {image_path.name}")
        except Exception as exc:
            print(f"[group] error: {caption_path} ({exc})")

    print(f"[group] moved {moved} pair(s)")


if __name__ == "__main__":
    main()
