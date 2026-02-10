from __future__ import annotations

import argparse
import shutil
import sys
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

INVALID_CHARS = set('<>:"/\\|?*')


def parse_tags(raw_tags: str) -> list[str]:
    tags: list[str] = []
    for part in raw_tags.split(","):
        tag = part.strip()
        if tag:
            tags.append(tag)
    return tags


def normalize_tag(tag: str) -> str:
    return tag.strip().lower()


def validate_tag_names(tags: list[str]) -> None:
    for tag in tags:
        if tag in {".", ".."}:
            sys.exit(f"ERROR: invalid tag: {tag}")
        if any(ch in INVALID_CHARS for ch in tag):
            sys.exit(f"ERROR: tag has invalid path characters: {tag}")


def caption_tags(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    parts = [p.strip() for p in text.split(",")]
    return {normalize_tag(p) for p in parts if p}


def index_images(root: Path, dest_dir: Path) -> dict[Path, dict[str, Path]]:
    index: dict[Path, dict[str, Path]] = {}
    for path in root.rglob("*"):
        if dest_dir in path.parents:
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        by_stem = index.setdefault(path.parent, {})
        existing = by_stem.get(path.stem)
        if existing is None:
            by_stem[path.stem] = path
            continue
        if path.suffix.lower() < existing.suffix.lower():
            by_stem[path.stem] = path
    return index


def iter_caption_files(root: Path, dest_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.txt"):
        if dest_dir in path.parents:
            continue
        files.append(path)
    return files


def build_caption_triplets(
    root: Path, dest_dir: Path
) -> list[tuple[Path, Path, set[str]]]:
    image_index = index_images(root, dest_dir)
    triplets: list[tuple[Path, Path, set[str]]] = []
    for caption_path in iter_caption_files(root, dest_dir):
        tags_in_caption = caption_tags(caption_path)
        image_path = image_index.get(caption_path.parent, {}).get(caption_path.stem)
        if image_path is None:
            continue
        triplets.append((image_path, caption_path, tags_in_caption))
    return triplets


def move_pair(image_path: Path, caption_path: Path, dest_dir: Path) -> bool:
    dst_image = dest_dir / image_path.name
    dst_caption = dest_dir / caption_path.name
    if dst_image.exists() or dst_caption.exists():
        print(f"skip (exists): {image_path.name}")
        return False
    shutil.move(str(image_path), str(dst_image))
    shutil.move(str(caption_path), str(dst_caption))
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract image/caption pairs by tags.")
    p.add_argument("-i", "--input", required=True, type=Path, help="Directory with images + captions.")
    p.add_argument(
        "--tags",
        required=True,
        help="Comma-separated tags to match (all must be present).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = args.input.expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: {root} is not a directory.")

    tags = parse_tags(args.tags)
    if not tags:
        sys.exit("ERROR: no tags provided.")

    validate_tag_names(tags)
    dest_dir = root / "__".join(tags)
    dest_dir.mkdir(parents=True, exist_ok=True)

    required = {normalize_tag(t) for t in tags}

    caption_triplets = build_caption_triplets(root, dest_dir)
    if not caption_triplets:
        print(f"[extract] no captions found under {root}")
        return

    moved = 0
    for image_path, caption_path, tags_in_caption in caption_triplets:
        if not required.issubset(tags_in_caption):
            continue

        if move_pair(image_path, caption_path, dest_dir):
            moved += 1

    if moved == 0:
        print("[extract] no matches found.")
    else:
        print(f"[extract] moved {moved} pair(s) to {dest_dir}")


if __name__ == "__main__":
    main()
