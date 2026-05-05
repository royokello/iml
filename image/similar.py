from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from image.dedup.grouping import components_for_items, scan_images

DHASH_SIZE = 8
MIN_GROUP_SIZE = 2


@dataclass(frozen=True)
class TargetGroup:
    threshold: int
    group: list[tuple[Path, str, int, int, int]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Copy a target number of similar-filtered images.")
    p.add_argument("-i", "--input", required=True, type=Path, help="Directory to scan for images.")
    p.add_argument("-o", "--output", required=True, type=Path, help="Directory to write selected images.")
    p.add_argument("--target", required=True, type=int, help="Number of images to write.")
    return p.parse_args()


def copy_images(output: Path, items: list[tuple[Path, str, int, int, int]]) -> tuple[int, int]:
    copied_images = 0
    copied_captions = 0

    if output.exists():
        if not output.is_dir():
            raise ValueError(f"--output exists and is not a directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    for src_path, rel_path, *_rest in items:
        dst_image = output / rel_path
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_image)
        copied_images += 1

        src_caption = src_path.with_suffix(".txt")
        if src_caption.is_file():
            dst_caption = dst_image.with_suffix(".txt")
            dst_caption.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_caption, dst_caption)
            copied_captions += 1

    return copied_images, copied_captions


def group_rank(group_size: int, threshold: int, target: int) -> tuple[int, int, int]:
    return (
        abs(group_size - target),
        0 if group_size >= target else 1,
        threshold,
    )


def find_target_group(
    items: list[tuple[Path, str, int, int, int]],
    target: int,
    max_threshold: int,
) -> TargetGroup:
    best: TargetGroup | None = None
    best_rank: tuple[int, int, int] | None = None

    for threshold in range(max_threshold + 1):
        components = components_for_items(items, threshold)
        groups = [group for group in components if len(group) >= MIN_GROUP_SIZE]
        if not groups:
            continue

        candidate = min(
            groups,
            key=lambda group: (
                *group_rank(len(group), threshold, target),
                len(group),
                group[0][1],
            ),
        )
        candidate_rank = group_rank(len(candidate), threshold, target)
        if best is None or best_rank is None or candidate_rank < best_rank:
            best = TargetGroup(threshold=threshold, group=candidate)
            best_rank = candidate_rank
            print(f"threshold eval: threshold={threshold} group={len(candidate)} delta={candidate_rank[0]}")
            if len(candidate) == target:
                break

    if best is None:
        raise ValueError("no similar groups found.")
    return best


def main() -> None:
    args = parse_args()
    root = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not root.is_dir():
        sys.exit(f"ERROR: {root} is not a directory.")
    if output == root or root in output.parents:
        sys.exit("ERROR: --output cannot be the same directory as --input or a subdirectory of --input.")
    if args.target < 1:
        sys.exit("ERROR: --target must be >= 1.")

    max_threshold = DHASH_SIZE * DHASH_SIZE
    items = scan_images(root, DHASH_SIZE)
    if not items:
        sys.exit("ERROR: no images found.")
    if args.target > len(items):
        sys.exit(f"ERROR: --target must be <= total images ({len(items)}).")

    try:
        target_group = find_target_group(
            items=items,
            target=args.target,
            max_threshold=max_threshold,
        )
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    selected = sorted(target_group.group, key=lambda item: item[1])[: args.target]
    try:
        copied_images, copied_captions = copy_images(output, selected)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    print(f"threshold: {target_group.threshold}")
    print(f"group images: {len(target_group.group)}")
    print(f"target images: {args.target}")
    print(f"copied images: {copied_images}")
    print(f"copied captions: {copied_captions}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
