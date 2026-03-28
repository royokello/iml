from __future__ import annotations

from math import gcd
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from PIL import Image

IMAGE_EXTS: set[str] = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
}


ImageItem: TypeAlias = tuple[Path, str, int, int, int]
ImageItems: TypeAlias = list[ImageItem]
SetKey: TypeAlias = int | tuple[int, int]

SET_CHOICES: tuple[str, ...] = ("height", "width", "longest", "shortest", "ratio")


@dataclass(frozen=True)
class ThresholdStats:
    threshold: int | None
    total_images: int
    unique_count: int
    duplicate_groups: int
    duplicate_images: int
    groups: list[list[ImageItem]]


@dataclass(frozen=True)
class ImagePartition:
    label: str
    items: ImageItems


def uf_init(size: int) -> tuple[list[int], list[int]]:
    return list(range(size)), [0] * size


def uf_find(parent: list[int], x: int) -> int:
    while x != parent[x]:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def uf_union(parent: list[int], rank: list[int], a: int, b: int) -> None:
    ra = uf_find(parent, a)
    rb = uf_find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1


def uf_groups(parent: list[int]) -> list[list[int]]:
    buckets: dict[int, list[int]] = {}
    for i in range(len(parent)):
        root = uf_find(parent, i)
        buckets.setdefault(root, []).append(i)
    return list(buckets.values())


def dhash_image(im: Image.Image, hash_size: int) -> int:
    if hash_size < 2:
        raise ValueError("--dhash-size must be >= 2")
    im = im.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(im.getdata())
    value = 0
    stride = hash_size + 1
    for row in range(hash_size):
        row_start = row * stride
        for col in range(hash_size):
            left = pixels[row_start + col]
            right = pixels[row_start + col + 1]
            value = (value << 1) | (1 if right > left else 0)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def normalized_ratio(width: int, height: int) -> tuple[int, int]:
    factor = gcd(width, height)
    if factor == 0:
        return width, height
    return width // factor, height // factor


def item_set_key(item: ImageItem, set_name: str) -> SetKey:
    width = item[2]
    height = item[3]
    if set_name == "width":
        return width
    if set_name == "height":
        return height
    if set_name == "longest":
        return max(width, height)
    if set_name == "shortest":
        return min(width, height)
    if set_name == "ratio":
        return normalized_ratio(width, height)
    raise ValueError(f"unsupported set name: {set_name}")


def set_key_sort_value(key: SetKey) -> tuple[int, ...]:
    if isinstance(key, tuple):
        return key
    return (key,)


def format_set_key(key: SetKey) -> str:
    if isinstance(key, tuple):
        return f"{key[0]}x{key[1]}"
    return str(key)


def split_items_by_set(items: ImageItems, set_name: str) -> list[tuple[SetKey, ImageItems]]:
    buckets: dict[SetKey, ImageItems] = {}
    for item in items:
        key = item_set_key(item, set_name)
        buckets.setdefault(key, []).append(item)
    return sorted(buckets.items(), key=lambda pair: set_key_sort_value(pair[0]))


def partition_label(
    first_set: str | None,
    first_key: SetKey | None = None,
    second_set: str | None = None,
    second_key: SetKey | None = None,
) -> str:
    if first_set is None:
        return "global"
    if second_set is None or second_key is None:
        return f"{first_set}={format_set_key(first_key)}"
    return f"{first_set}={format_set_key(first_key)} {second_set}={format_set_key(second_key)}"


def build_partitions(
    items: ImageItems,
    first_set: str | None = None,
    second_set: str | None = None,
    *,
    report: bool = False,
) -> list[ImagePartition]:
    if second_set is not None and first_set is None:
        raise ValueError("--second-set requires --first-set")
    if first_set is None:
        return [ImagePartition(label="global", items=items)] if items else []

    first_level = split_items_by_set(items, first_set)
    if report:
        print(f"total first-level sets: {len(first_level)}")

    if second_set is None:
        partitions = [
            ImagePartition(label=partition_label(first_set, first_key), items=subset)
            for first_key, subset in first_level
        ]
        if report:
            for partition in partitions:
                print(f"set {partition.label}: {len(partition.items)} images")
        return partitions

    partitions: list[ImagePartition] = []
    for first_key, subset in first_level:
        for second_key, subsubset in split_items_by_set(subset, second_set):
            partitions.append(
                ImagePartition(
                    label=partition_label(first_set, first_key, second_set, second_key),
                    items=subsubset,
                )
            )

    if report:
        print(f"total second-level sets: {len(partitions)}")
        for partition in partitions:
            print(f"set {partition.label}: {len(partition.items)} images")
    return partitions


def partition_items(
    items: ImageItems,
    first_set: str | None = None,
    second_set: str | None = None,
    *,
    report: bool = False,
) -> list[ImageItems]:
    return [partition.items for partition in build_partitions(items, first_set, second_set, report=report)]


def scan_images(root: Path, hash_size: int) -> ImageItems:
    items: ImageItems = []
    total = 0
    skipped = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            with Image.open(path) as im:
                w, h = im.size
                dh = dhash_image(im, hash_size)
            rel = path.relative_to(root).as_posix()
            items.append((path, rel, w, h, dh))
            total += 1
        except Exception as exc:
            skipped += 1
            print(f"skipped {path}: {exc}", file=sys.stderr)
    print(f"scanned {total} images ({skipped} skipped)")
    items.sort(key=lambda x: x[1])
    return items


def group_by_hash(items: list[ImageItem], threshold: int) -> list[list[int]]:
    n = len(items)
    parent, rank = uf_init(n)
    for i in range(n):
        hi = items[i][4]
        for j in range(i + 1, n):
            if hamming(hi, items[j][4]) <= threshold:
                uf_union(parent, rank, i, j)
    return uf_groups(parent)


def components_for_items(items: ImageItems, threshold: int) -> list[list[ImageItem]]:
    components: list[list[ImageItem]] = []
    for comp in group_by_hash(items, threshold):
        components.append([items[i] for i in comp])
    components.sort(key=lambda group: group[0][1])
    return components


def components_in_sets(
    items: ImageItems,
    threshold: int,
    first_set: str | None = None,
    second_set: str | None = None,
) -> list[list[ImageItem]]:
    components: list[list[ImageItem]] = []
    for partition in build_partitions(items, first_set=first_set, second_set=second_set):
        components.extend(components_for_items(partition.items, threshold))
    components.sort(key=lambda group: group[0][1])
    return components


def evaluate_threshold(
    items: ImageItems,
    dhash_threshold: int,
    min_group_size: int,
    first_set: str | None = None,
    second_set: str | None = None,
) -> ThresholdStats:
    all_groups: list[list[ImageItem]] = []
    total_images = len(items)
    unique_count = 0
    duplicate_groups = 0
    duplicate_images = 0

    components = components_in_sets(
        items,
        threshold=dhash_threshold,
        first_set=first_set,
        second_set=second_set,
    )
    unique_count = len(components)

    for group in components:
        if len(group) >= 2:
            duplicate_groups += 1
            duplicate_images += len(group)
        if len(group) >= min_group_size:
            all_groups.append(group)

    all_groups.sort(key=lambda g: g[0][1])
    return ThresholdStats(
        threshold=dhash_threshold,
        total_images=total_images,
        unique_count=unique_count,
        duplicate_groups=duplicate_groups,
        duplicate_images=duplicate_images,
        groups=all_groups,
    )


def combine_threshold_stats(stats_list: list[ThresholdStats]) -> ThresholdStats:
    if not stats_list:
        return ThresholdStats(
            threshold=None,
            total_images=0,
            unique_count=0,
            duplicate_groups=0,
            duplicate_images=0,
            groups=[],
        )

    first_threshold = stats_list[0].threshold
    threshold = first_threshold if all(stats.threshold == first_threshold for stats in stats_list) else None
    groups: list[list[ImageItem]] = []
    for stats in stats_list:
        groups.extend(stats.groups)
    groups.sort(key=lambda group: group[0][1])
    return ThresholdStats(
        threshold=threshold,
        total_images=sum(stats.total_images for stats in stats_list),
        unique_count=sum(stats.unique_count for stats in stats_list),
        duplicate_groups=sum(stats.duplicate_groups for stats in stats_list),
        duplicate_images=sum(stats.duplicate_images for stats in stats_list),
        groups=groups,
    )


def build_groups(
    root: Path,
    dhash_size: int,
    dhash_threshold: int,
    min_group_size: int,
    first_set: str | None = None,
    second_set: str | None = None,
) -> list[list[ImageItem]]:
    items = scan_images(root, dhash_size)
    return evaluate_threshold(
        items=items,
        dhash_threshold=dhash_threshold,
        min_group_size=min_group_size,
        first_set=first_set,
        second_set=second_set,
    ).groups
