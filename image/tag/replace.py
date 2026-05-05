from __future__ import annotations

import argparse
import sys
from pathlib import Path


def iter_caption_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*.txt") if path.is_file()]


def parse_caption(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def replace_tags(tags: list[str], old: str, new: str) -> tuple[list[str], int]:
    updated: list[str] = []
    replacements = 0
    for tag in tags:
        if tag == old:
            replacements += 1
            if new:
                updated.append(new)
        else:
            updated.append(tag)
    return updated, replacements


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace comma-separated tags in caption files in a dataset directory.")
    parser.add_argument("--dataset", required=True, type=Path, help="Directory containing caption .txt files.")
    parser.add_argument("--old", required=True, help="Tag to replace.")
    parser.add_argument("--new", required=True, help="Replacement tag. Use an empty string to remove the tag.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = args.dataset.expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: {root} is not a directory.")
    if args.old == "":
        sys.exit("ERROR: --old cannot be empty.")

    caption_files = iter_caption_files(root)
    if not caption_files:
        print(f"[replace] no captions found under {root}")
        return

    updated = 0
    replacements = 0
    for caption_path in caption_files:
        try:
            text = caption_path.read_text(encoding="utf-8", errors="ignore")
            tags = parse_caption(text)
            new_tags, count = replace_tags(tags, args.old, args.new)
            if count == 0:
                continue

            caption_path.write_text(", ".join(new_tags), encoding="utf-8")
            updated += 1
            replacements += count
        except Exception as exc:
            print(f"[replace] error: {caption_path} ({exc})")

    print(f"[replace] updated {updated} caption(s)")
    print(f"[replace] replacements: {replacements}")


if __name__ == "__main__":
    main()
