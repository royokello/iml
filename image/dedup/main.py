from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

from .grouping import (
    ImageItem,
    ItemsByResolution,
    ThresholdStats,
    evaluate_threshold,
    group_by_hash,
    scan_images,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Group near-duplicate images with dHash.")
    p.add_argument("-i", "--input", required=True, type=Path, help="Directory to scan for images.")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Directory to copy deduplicated images and optional .txt captions.",
    )

    p.add_argument("--dhash-size", type=int, default=8)
    p.add_argument("--dhash-threshold", type=int, default=8)
    p.add_argument("--target", type=int, default=None)

    p.add_argument("--min-group-size", type=int, default=2)
    return p.parse_args()


def choose_better(a: ThresholdStats, b: ThresholdStats, target: int) -> ThresholdStats:
    delta_a = abs(a.unique_count - target)
    delta_b = abs(b.unique_count - target)
    if delta_a != delta_b:
        return a if delta_a < delta_b else b
    if a.threshold != b.threshold:
        return a if a.threshold > b.threshold else b
    return a


def find_closest_threshold(
    items_by_res: ItemsByResolution,
    target: int,
    min_group_size: int,
    max_threshold: int,
) -> tuple[ThresholdStats, list[ThresholdStats]]:
    cache: dict[int, ThresholdStats] = {}
    trials: list[ThresholdStats] = []

    def evaluate(threshold: int) -> ThresholdStats:
        cached = cache.get(threshold)
        if cached is not None:
            return cached
        stats = evaluate_threshold(
            items_by_res=items_by_res,
            dhash_threshold=threshold,
            min_group_size=min_group_size,
        )
        cache[threshold] = stats
        trials.append(stats)
        return stats

    lo = 0
    hi = max_threshold
    while lo < hi:
        mid = (lo + hi) // 2
        stats = evaluate(mid)
        if stats.unique_count > target:
            lo = mid + 1
        else:
            hi = mid

    candidates: list[ThresholdStats] = []
    for threshold in (lo - 1, lo, lo + 1):
        if 0 <= threshold <= max_threshold:
            candidates.append(evaluate(threshold))

    best = candidates[0]
    for candidate in candidates[1:]:
        best = choose_better(best, candidate, target)
    return best, trials


def print_stats(stats: ThresholdStats, min_group_size: int) -> None:
    print(f"[dedup] threshold: {stats.threshold}")
    print(f"[dedup] total images: {stats.total_images}")
    print(f"[dedup] unique images: {stats.unique_count}")
    print(f"[dedup] removable duplicates: {stats.total_images - stats.unique_count}")
    print(f"[dedup] groups (min size {min_group_size}): {len(stats.groups)}")
    if not stats.groups:
        print("[dedup] no groups found")
        return
    sizes = [len(group) for group in stats.groups]
    grouped = sum(sizes)
    print(f"[dedup] grouped images: {grouped}")
    print(f"[dedup] group size min/avg/max: {min(sizes)}/{grouped / len(sizes):.2f}/{max(sizes)}")


def sharpness_score(path: Path, max_size: int = 256) -> float:
    try:
        with Image.open(path) as im:
            im = im.convert("L")
            if max(im.size) > max_size:
                im = im.copy()
                im.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            w, h = im.size
            if w < 3 or h < 3:
                return 0.0
            data = im.get_flattened_data() if hasattr(im, "get_flattened_data") else im.getdata()
            pixels = list(data)
    except Exception:
        return -1.0

    n = 0
    sum_l = 0.0
    sum_l2 = 0.0
    for y in range(1, h - 1):
        row = y * w
        row_up = row - w
        row_dn = row + w
        for x in range(1, w - 1):
            idx = row + x
            c = pixels[idx]
            lap = 4 * c - (pixels[row_up + x] + pixels[row_dn + x] + pixels[idx - 1] + pixels[idx + 1])
            sum_l += lap
            sum_l2 += lap * lap
            n += 1
    if n == 0:
        return 0.0
    mean = sum_l / n
    return (sum_l2 / n) - (mean * mean)


def select_representative_images(items_by_res: ItemsByResolution, threshold: int) -> list[ImageItem]:
    keepers: list[ImageItem] = []
    score_cache: dict[Path, float] = {}

    def score_item(item: ImageItem) -> float:
        path = item[0]
        cached = score_cache.get(path)
        if cached is not None:
            return cached
        score = sharpness_score(path)
        score_cache[path] = score
        return score

    for items in items_by_res.values():
        if not items:
            continue
        if len(items) == 1:
            keepers.append(items[0])
            continue
        components = group_by_hash(items, threshold)
        for comp in components:
            best_idx = max(comp, key=lambda idx: (score_item(items[idx]), items[idx][1]))
            keepers.append(items[best_idx])
    keepers.sort(key=lambda item: item[1])
    return keepers


def copy_dedup_output(output: Path, keepers: list[ImageItem]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    copied_images = 0
    copied_captions = 0

    for src_path, rel, *_ in keepers:
        dst_image = output / rel
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_image)
        copied_images += 1

        src_caption = src_path.with_suffix(".txt")
        if src_caption.is_file():
            dst_caption = output / Path(rel).with_suffix(".txt")
            dst_caption.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_caption, dst_caption)
            copied_captions += 1

    print(f"[dedup] copied unique images: {copied_images}")
    print(f"[dedup] copied captions: {copied_captions}")
    print(f"[dedup] output: {output}")


def main() -> None:
    args = parse_args()
    root: Path = args.input.expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: {root} is not a directory.")
    output: Path | None = None
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if output == root or root in output.parents:
            sys.exit("ERROR: --output cannot be the same directory as --input or a subdirectory of --input.")

    if args.target is not None and args.target < 1:
        sys.exit("ERROR: --target must be >= 1.")

    max_threshold = args.dhash_size * args.dhash_size
    if args.dhash_threshold < 0 or args.dhash_threshold > max_threshold:
        sys.exit(f"ERROR: --dhash-threshold must be in [0, {max_threshold}] for --dhash-size {args.dhash_size}.")

    items_by_res = scan_images(root, args.dhash_size)
    if args.target is not None:
        best, trials = find_closest_threshold(
            items_by_res=items_by_res,
            target=args.target,
            min_group_size=args.min_group_size,
            max_threshold=max_threshold,
        )
        print(f"[dedup][auto] target unique: {args.target}")
        for trial in trials:
            delta = abs(trial.unique_count - args.target)
            print(f"[dedup][auto] tried threshold={trial.threshold} unique={trial.unique_count} delta={delta}")
        print(f"[dedup][auto] tried {len(trials)} thresholds")
        stats = best
    else:
        stats = evaluate_threshold(
            items_by_res=items_by_res,
            dhash_threshold=args.dhash_threshold,
            min_group_size=args.min_group_size,
        )
    print_stats(stats, args.min_group_size)

    if output is None:
        return

    keepers = select_representative_images(items_by_res, stats.threshold)
    if len(keepers) != stats.unique_count:
        print(
            f"[dedup] warning: selected keepers ({len(keepers)}) != unique images ({stats.unique_count})",
            file=sys.stderr,
        )
    copy_dedup_output(output=output, keepers=keepers)


if __name__ == "__main__":
    main()
