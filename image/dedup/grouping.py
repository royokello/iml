from __future__ import annotations

import sys
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTS: set[str] = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
}


ImageItem: TypeAlias = tuple[Path, str, int, int, int]


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


def scan_images(root: Path, hash_size: int) -> dict[tuple[int, int], list[ImageItem]]:
    items_by_res: dict[tuple[int, int], list[ImageItem]] = {}
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


def _adjust_window(window: int, shape: tuple[int, int]) -> int:
    min_dim = min(shape)
    win = min(window, min_dim)
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return 0
    return win


def _load_ssim_image(path: Path, width: int | None) -> np.ndarray | None:
    try:
        with Image.open(path) as im:
            im = im.convert("L")
            w, h = im.size
            if width and width > 0 and width < w:
                new_h = max(1, int(round(h * (width / w))))
                im = im.resize((width, new_h), Image.Resampling.LANCZOS)
            return np.asarray(im, dtype=np.float32)
    except Exception:
        return None


def compute_ssim(img1: np.ndarray, img2: np.ndarray, window: int, gaussian: bool) -> float:
    if img1.shape != img2.shape:
        return 0.0
    win = _adjust_window(window, img1.shape)
    if win == 0:
        return 1.0 if np.array_equal(img1, img2) else 0.0

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    if gaussian:
        mu1 = cv2.GaussianBlur(img1, (win, win), 1.5)
        mu2 = cv2.GaussianBlur(img2, (win, win), 1.5)
        sigma1 = cv2.GaussianBlur(img1 * img1, (win, win), 1.5) - mu1 * mu1
        sigma2 = cv2.GaussianBlur(img2 * img2, (win, win), 1.5) - mu2 * mu2
        sigma12 = cv2.GaussianBlur(img1 * img2, (win, win), 1.5) - mu1 * mu2
    else:
        mu1 = cv2.blur(img1, (win, win))
        mu2 = cv2.blur(img2, (win, win))
        sigma1 = cv2.blur(img1 * img1, (win, win)) - mu1 * mu1
        sigma2 = cv2.blur(img2 * img2, (win, win)) - mu2 * mu2
        sigma12 = cv2.blur(img1 * img2, (win, win)) - mu1 * mu2

    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1 * mu1 + mu2 * mu2 + c1) * (sigma1 + sigma2 + c2)
    ssim_map = numerator / (denominator + 1e-8)
    return float(ssim_map.mean())


def group_by_ssim(
    items: list[ImageItem],
    threshold: float,
    width: int | None,
    window: int,
    gaussian: bool,
) -> list[list[int]]:
    n = len(items)
    parent, rank = uf_init(n)
    cache: dict[Path, np.ndarray | None] = {}

    for i in range(n):
        item_i = items[i]
        img_i = cache.get(item_i[0])
        if img_i is None:
            img_i = _load_ssim_image(item_i[0], width)
            cache[item_i[0]] = img_i
        if img_i is None:
            continue

        for j in range(i + 1, n):
            item_j = items[j]
            img_j = cache.get(item_j[0])
            if img_j is None:
                img_j = _load_ssim_image(item_j[0], width)
                cache[item_j[0]] = img_j
            if img_j is None:
                continue

            score = compute_ssim(img_i, img_j, window, gaussian)
            if score >= threshold:
                uf_union(parent, rank, i, j)

    return uf_groups(parent)


def build_groups(
    root: Path,
    dhash_size: int,
    dhash_threshold: int,
    ssim_threshold: float,
    ssim_width: int | None,
    ssim_window: int,
    ssim_gaussian: bool,
    min_group_size: int,
) -> list[list[ImageItem]]:
    items_by_res = scan_images(root, dhash_size)
    all_groups: list[list[ImageItem]] = []

    for _res, items in items_by_res.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda x: x[1])
        candidates = group_by_hash(items, dhash_threshold)
        for comp in candidates:
            bucket = [items[i] for i in comp]
            if len(bucket) <= 1:
                all_groups.append(bucket)
                continue
            if ssim_threshold <= 0:
                all_groups.append(bucket)
                continue
            refined = group_by_ssim(bucket, ssim_threshold, ssim_width, ssim_window, ssim_gaussian)
            for refined_comp in refined:
                all_groups.append([bucket[i] for i in refined_comp])

    all_groups = [g for g in all_groups if len(g) >= min_group_size]
    all_groups.sort(key=lambda g: g[0][1])
    print(f"[dedup] {len(all_groups)} groups after ssim (min size {min_group_size})")
    return all_groups
