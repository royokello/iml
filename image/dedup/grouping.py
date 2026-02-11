from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from PIL import Image

IMAGE_EXTS: set[str] = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
}


ImageItem: TypeAlias = tuple[Path, str, int, int, int]
ItemsByResolution: TypeAlias = dict[tuple[int, int], list[ImageItem]]


@dataclass(frozen=True)
class ThresholdStats:
    threshold: int
    total_images: int
    unique_count: int
    duplicate_groups: int
    duplicate_images: int
    groups: list[list[ImageItem]]


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


def scan_images(root: Path, hash_size: int) -> ItemsByResolution:
    items_by_res: ItemsByResolution = {}
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
            items_by_res.setdefault((w, h), []).append((path, rel, w, h, dh))
            total += 1
        except Exception as exc:
            skipped += 1
            print(f"[dedup] skipped {path}: {exc}", file=sys.stderr)
    print(f"[dedup] scanned {total} images ({skipped} skipped)")
    print(f"[dedup] {len(items_by_res)} resolution buckets")
    for items in items_by_res.values():
        items.sort(key=lambda x: x[1])
    return items_by_res


def group_by_hash(items: list[ImageItem], threshold: int) -> list[list[int]]:
    n = len(items)
    parent, rank = uf_init(n)
    for i in range(n):
        hi = items[i][4]
        for j in range(i + 1, n):
            if hamming(hi, items[j][4]) <= threshold:
                uf_union(parent, rank, i, j)
    return uf_groups(parent)


def evaluate_threshold(
    items_by_res: ItemsByResolution,
    dhash_threshold: int,
    min_group_size: int,
) -> ThresholdStats:
    all_groups: list[list[ImageItem]] = []
    total_images = 0
    unique_count = 0
    duplicate_groups = 0
    duplicate_images = 0

    for items in items_by_res.values():
        if not items:
            continue

        total_images += len(items)
        if len(items) == 1:
            unique_count += 1
            if min_group_size <= 1:
                all_groups.append([items[0]])
            continue

        components = group_by_hash(items, dhash_threshold)
        unique_count += len(components)

        for comp in components:
            group = [items[i] for i in comp]
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


def build_groups(
    root: Path,
    dhash_size: int,
    dhash_threshold: int,
    min_group_size: int,
) -> list[list[ImageItem]]:
    items_by_res = scan_images(root, dhash_size)
    return evaluate_threshold(
        items_by_res=items_by_res,
        dhash_threshold=dhash_threshold,
        min_group_size=min_group_size,
    ).groups
