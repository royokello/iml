from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .grouping import (
    ImageItem,
    ImageItems,
    SET_CHOICES,
    ThresholdStats,
    components_in_sets,
    evaluate_threshold,
    partition_items,
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
    p.add_argument(
        "--first-set",
        choices=SET_CHOICES,
        default=None,
        help="Optional first comparison set: compare only within matching width/height/longest/shortest/ratio values.",
    )
    p.add_argument(
        "--second-set",
        choices=SET_CHOICES,
        default=None,
        help="Optional second comparison set applied inside each first-set bucket.",
    )

    p.add_argument("--min-group-size", type=int, default=2)
    p.add_argument(
        "--sharpness-weight",
        type=float,
        default=1.0,
        help="Weight for sharpness (higher keeps sharper images). Default: 1.0",
    )
    p.add_argument(
        "--exposure-weight",
        type=float,
        default=0.4,
        help="Weight for exposure penalty (higher avoids under/over exposed images). Default: 0.4",
    )
    p.add_argument(
        "--noise-weight",
        type=float,
        default=0.3,
        help="Weight for noise penalty (higher avoids noisy images). Default: 0.3",
    )
    p.add_argument(
        "--artifact-weight",
        type=float,
        default=0.55,
        help="Weight for compression artifact penalty (higher avoids blocky images). Default: 0.55",
    )
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
    items: ImageItems,
    target: int,
    min_group_size: int,
    max_threshold: int,
    first_set: str | None = None,
    second_set: str | None = None,
) -> tuple[ThresholdStats, list[ThresholdStats]]:
    cache: dict[int, ThresholdStats] = {}
    trials: list[ThresholdStats] = []

    def evaluate(threshold: int) -> ThresholdStats:
        cached = cache.get(threshold)
        if cached is not None:
            return cached
        stats = evaluate_threshold(
            items=items,
            dhash_threshold=threshold,
            min_group_size=min_group_size,
            first_set=first_set,
            second_set=second_set,
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
    print(f"threshold: {stats.threshold}")
    print(f"total images: {stats.total_images}")
    print(f"unique images: {stats.unique_count}")
    print(f"removable duplicates: {stats.total_images - stats.unique_count}")
    print(f"groups (min size {min_group_size}): {len(stats.groups)}")
    if not stats.groups:
        print("no groups found")
        return
    sizes = [len(group) for group in stats.groups]
    grouped = sum(sizes)
    print(f"grouped images: {grouped}")
    print(f"group size min/avg/max: {min(sizes)}/{grouped / len(sizes):.2f}/{max(sizes)}")


@dataclass(frozen=True)
class QualityWeights:
    sharpness: float
    exposure: float
    noise: float
    artifact: float


@dataclass(frozen=True)
class ImageQuality:
    sharpness: float
    exposure_penalty: float
    noise_penalty: float
    artifact_penalty: float


def clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def load_grayscale_pixels(path: Path, max_size: int = 256) -> tuple[int, int, list[int]] | None:
    try:
        with Image.open(path) as im:
            im = im.convert("L")
            if max(im.size) > max_size:
                im = im.copy()
                im.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            w, h = im.size
            if w < 3 or h < 3:
                return None
            data = im.get_flattened_data() if hasattr(im, "get_flattened_data") else im.getdata()
            pixels = list(data)
    except Exception:
        return None
    return w, h, pixels


def sharpness_score_pixels(w: int, h: int, pixels: list[int]) -> float:
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


def exposure_penalty_pixels(pixels: list[int]) -> float:
    total = len(pixels)
    if total == 0:
        return 1.0

    sum_lum = 0.0
    clipped_dark = 0
    clipped_bright = 0
    for px in pixels:
        sum_lum += px
        if px <= 5:
            clipped_dark += 1
        elif px >= 250:
            clipped_bright += 1

    mean_norm = (sum_lum / total) / 255.0
    mean_penalty = abs(mean_norm - 0.5) / 0.5
    dark_ratio = clipped_dark / total
    bright_ratio = clipped_bright / total
    clip_penalty = clamp01((dark_ratio + bright_ratio) / 0.25)
    return clamp01(0.7 * mean_penalty + 0.3 * clip_penalty)


def noise_penalty_pixels(w: int, h: int, pixels: list[int]) -> float:
    residual_sum = 0.0
    count = 0
    edge_threshold = 20

    for y in range(1, h - 1):
        row = y * w
        row_up = row - w
        row_dn = row + w
        for x in range(1, w - 1):
            idx = row + x
            left = pixels[idx - 1]
            right = pixels[idx + 1]
            up = pixels[row_up + x]
            down = pixels[row_dn + x]
            grad = abs(right - left) + abs(down - up)
            if grad > edge_threshold:
                continue
            smooth = (left + right + up + down) * 0.25
            residual_sum += abs(pixels[idx] - smooth)
            count += 1

    if count == 0:
        return 0.0
    mean_residual = residual_sum / count
    return clamp01(mean_residual / 14.0)


def artifact_penalty_pixels(w: int, h: int, pixels: list[int]) -> float:
    if w < 16 or h < 16:
        return 0.0

    b_sum = 0.0
    b_count = 0
    nb_sum = 0.0
    nb_count = 0

    for y in range(h):
        row = y * w
        for x in range(1, w):
            diff = abs(pixels[row + x] - pixels[row + x - 1])
            if x % 8 == 0:
                b_sum += diff
                b_count += 1
            else:
                nb_sum += diff
                nb_count += 1

    for y in range(1, h):
        for x in range(w):
            diff = abs(pixels[y * w + x] - pixels[(y - 1) * w + x])
            if y % 8 == 0:
                b_sum += diff
                b_count += 1
            else:
                nb_sum += diff
                nb_count += 1

    if b_count == 0 or nb_count == 0:
        return 0.0

    boundary_mean = b_sum / b_count
    non_boundary_mean = nb_sum / nb_count
    blockiness = max(0.0, boundary_mean - non_boundary_mean)
    return clamp01(blockiness / 8.0)


def image_quality(path: Path) -> ImageQuality:
    loaded = load_grayscale_pixels(path)
    if loaded is None:
        return ImageQuality(
            sharpness=-1.0,
            exposure_penalty=1.0,
            noise_penalty=1.0,
            artifact_penalty=1.0,
        )
    w, h, pixels = loaded
    return ImageQuality(
        sharpness=sharpness_score_pixels(w, h, pixels),
        exposure_penalty=exposure_penalty_pixels(pixels),
        noise_penalty=noise_penalty_pixels(w, h, pixels),
        artifact_penalty=artifact_penalty_pixels(w, h, pixels),
    )


def select_representative_images(
    items: ImageItems,
    threshold: int,
    weights: QualityWeights,
    first_set: str | None = None,
    second_set: str | None = None,
) -> list[ImageItem]:
    keepers: list[ImageItem] = []
    quality_cache: dict[Path, ImageQuality] = {}

    def quality_item(item: ImageItem) -> ImageQuality:
        path = item[0]
        cached = quality_cache.get(path)
        if cached is not None:
            return cached
        score = image_quality(path)
        quality_cache[path] = score
        return score

    if not items:
        return keepers

    components = components_in_sets(
        items,
        threshold=threshold,
        first_set=first_set,
        second_set=second_set,
    )
    for group in components:
        if len(group) == 1:
            keepers.append(group[0])
            continue

        qualities = {item[0]: quality_item(item) for item in group}
        sharp_values = [quality.sharpness for quality in qualities.values()]
        sharp_min = min(sharp_values)
        sharp_max = max(sharp_values)
        sharp_range = sharp_max - sharp_min

        def composite_score(item: ImageItem) -> float:
            q = qualities[item[0]]
            if sharp_range <= 1e-9:
                sharp_norm = 0.5
            else:
                sharp_norm = clamp01((q.sharpness - sharp_min) / sharp_range)
            return (
                (weights.sharpness * sharp_norm)
                - (weights.exposure * q.exposure_penalty)
                - (weights.noise * q.noise_penalty)
                - (weights.artifact * q.artifact_penalty)
            )

        best_item = min(group, key=lambda item: (-composite_score(item), item[1]))
        keepers.append(best_item)
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

    print(f"copied unique images: {copied_images}")
    print(f"copied captions: {copied_captions}")
    print(f"output: {output}")


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
    if args.second_set is not None and args.first_set is None:
        sys.exit("ERROR: --second-set requires --first-set.")
    if args.sharpness_weight < 0:
        sys.exit("ERROR: --sharpness-weight must be >= 0.")
    if args.exposure_weight < 0:
        sys.exit("ERROR: --exposure-weight must be >= 0.")
    if args.noise_weight < 0:
        sys.exit("ERROR: --noise-weight must be >= 0.")
    if args.artifact_weight < 0:
        sys.exit("ERROR: --artifact-weight must be >= 0.")

    weights = QualityWeights(
        sharpness=args.sharpness_weight,
        exposure=args.exposure_weight,
        noise=args.noise_weight,
        artifact=args.artifact_weight,
    )
    if (weights.sharpness + weights.exposure + weights.noise + weights.artifact) == 0:
        sys.exit("ERROR: At least one quality weight must be > 0.")

    max_threshold = args.dhash_size * args.dhash_size
    if args.dhash_threshold < 0 or args.dhash_threshold > max_threshold:
        sys.exit(f"ERROR: --dhash-threshold must be in [0, {max_threshold}] for --dhash-size {args.dhash_size}.")

    items = scan_images(root, args.dhash_size)
    if args.first_set is None:
        print("comparison sets: global")
    elif args.second_set is None:
        print(f"comparison sets: first={args.first_set}")
    else:
        print(f"comparison sets: first={args.first_set} second={args.second_set}")
    partition_items(items, args.first_set, args.second_set, report=True)
    if args.target is not None:
        best, trials = find_closest_threshold(
            items=items,
            target=args.target,
            min_group_size=args.min_group_size,
            max_threshold=max_threshold,
            first_set=args.first_set,
            second_set=args.second_set,
        )
        print(f"auto target unique: {args.target}")
        for trial in trials:
            delta = abs(trial.unique_count - args.target)
            print(f"auto tried threshold={trial.threshold} unique={trial.unique_count} delta={delta}")
        print(f"auto tried {len(trials)} thresholds")
        stats = best
    else:
        stats = evaluate_threshold(
            items=items,
            dhash_threshold=args.dhash_threshold,
            min_group_size=args.min_group_size,
            first_set=args.first_set,
            second_set=args.second_set,
        )
    print_stats(stats, args.min_group_size)

    if output is None:
        return

    print(
        "quality weights: "
        f"sharpness={weights.sharpness:g} "
        f"exposure={weights.exposure:g} "
        f"noise={weights.noise:g} "
        f"artifact={weights.artifact:g}"
    )
    keepers = select_representative_images(
        items=items,
        threshold=stats.threshold,
        weights=weights,
        first_set=args.first_set,
        second_set=args.second_set,
    )
    if len(keepers) != stats.unique_count:
        print(
            f"warning: selected keepers ({len(keepers)}) != unique images ({stats.unique_count})",
            file=sys.stderr,
        )
    copy_dedup_output(output=output, keepers=keepers)


if __name__ == "__main__":
    main()
