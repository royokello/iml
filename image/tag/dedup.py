from __future__ import annotations

import argparse
import sys
from pathlib import Path


def iter_caption_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*.txt") if path.is_file()]


def parse_tags(raw_tags: str) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for part in raw_tags.split(","):
        tag = part.strip()
        if not tag:
            continue
        key = normalize_tag(tag)
        if key in seen:
            continue
        seen.add(key)
        tags.append(tag)
    return tags


def parse_caption(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def normalize_tag(tag: str) -> str:
    return tag.strip().lower()


def dedup_group(tags: list[str], group: list[str], keep: str) -> tuple[list[str], bool, int]:
    group_keys = {normalize_tag(tag) for tag in group}
    keep_key = normalize_tag(keep)
    caption_keys = {normalize_tag(tag) for tag in tags}

    if not group_keys.issubset(caption_keys):
        return tags, False, 0

    updated: list[str] = []
    removed = 0
    kept_seen = False
    for tag in tags:
        key = normalize_tag(tag)
        if key not in group_keys:
            updated.append(tag)
            continue
        if key == keep_key and not kept_seen:
            updated.append(tag)
            kept_seen = True
            continue
        removed += 1

    return updated, removed > 0, removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove competing tags from captions when every tag in a group is present."
    )
    parser.add_argument("-i", "--input", required=True, type=Path, help="Directory containing caption .txt files.")
    parser.add_argument("--group", required=True, help="Comma-separated group of mutually exclusive tags.")
    parser.add_argument("--keep", required=True, help="The one tag from the group to retain.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = args.input.expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: {root} is not a directory.")

    group = parse_tags(args.group)
    if len(group) < 2:
        sys.exit("ERROR: --group must contain at least two tags.")
    if normalize_tag(args.keep) not in {normalize_tag(tag) for tag in group}:
        sys.exit("ERROR: --keep must be one of the tags in --group.")

    caption_files = iter_caption_files(root)
    if not caption_files:
        print(f"[dedup] no captions found under {root}")
        return

    updated = 0
    removed = 0
    for caption_path in caption_files:
        try:
            text = caption_path.read_text(encoding="utf-8", errors="ignore")
            tags = parse_caption(text)
            new_tags, changed, count = dedup_group(tags, group, args.keep)
            if not changed:
                continue

            caption_path.write_text(", ".join(new_tags), encoding="utf-8")
            updated += 1
            removed += count
        except Exception as exc:
            print(f"[dedup] error: {caption_path} ({exc})")

    print(f"[dedup] updated {updated} caption(s)")
    print(f"[dedup] removed {removed} tag(s)")


if __name__ == "__main__":
    main()
