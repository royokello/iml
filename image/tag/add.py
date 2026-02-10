from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_tags(raw_tags: str) -> list[str]:
    tags: list[str] = []
    for part in raw_tags.split(","):
        tag = part.strip()
        if tag:
            tags.append(tag)
    # Deduplicate while preserving order (case-insensitive).
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
    return out


def iter_caption_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.txt") if p.is_file()]


def parse_caption(text: str) -> list[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


def build_caption(existing: list[str], to_add: list[str], position: str) -> str:
    seen_existing: set[str] = set()
    unique_existing: list[str] = []
    for tag in existing:
        key = tag.lower()
        if key in seen_existing:
            continue
        seen_existing.add(key)
        unique_existing.append(tag)

    existing_keys = [t.lower() for t in unique_existing]
    add_keys = {t.lower() for t in to_add}
    if position == "start":
        kept = [t for t, key in zip(unique_existing, existing_keys) if key not in add_keys]
        merged = to_add + kept
    else:
        kept = [t for t, key in zip(unique_existing, existing_keys) if key not in add_keys]
        merged = kept + to_add
    return ", ".join(merged)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add tags to caption files in a directory.")
    p.add_argument("-i", "--input", required=True, type=Path, help="Directory containing caption .txt files.")
    p.add_argument(
        "--tags",
        required=True,
        help="Comma-separated tags to append to every caption.",
    )
    p.add_argument(
        "--position",
        choices=["start", "end"],
        default="end",
        help="Where to place tags relative to existing caption.",
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

    caption_files = iter_caption_files(root)
    if not caption_files:
        print(f"[add] no captions found under {root}")
        return

    updated = 0
    for caption_path in caption_files:
        try:
            text = caption_path.read_text(encoding="utf-8", errors="ignore")
            existing = parse_caption(text)
            new_text = build_caption(existing, tags, args.position)
            caption_path.write_text(new_text, encoding="utf-8")
            updated += 1
        except Exception as exc:
            print(f"[add] error: {caption_path} ({exc})")

    print(f"[add] updated {updated} caption(s)")


if __name__ == "__main__":
    main()
