# image/prepare.py
#
"""
Prepare image groups: resize and copy to base/ and high/ output dirs.

Groups are numbered sequentially (0000001, 0000002, ...) by sorted order.
Base thresholds are the gatekeeper — a group must pass base to be included.
If it also passes high thresholds, a copy at higher resolution goes into
high/ with the same number.

Captions and pre-computed text encodings ({stem}.text.safetensors) are
copied as-is into base/. Original filenames are preserved.

Usage:
    py -m image.prepare -i /path/to/input -o /path/to/output
    py -m image.prepare -i input -o output --base-target-res "256,384"
    py -m image.prepare -i input -o output --dry-run --verbose
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTS: set[str] = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
}


def parse_res_chain(s: str) -> list[int]:
    """Parse '384,256' -> [384, 256], sorted descending."""
    values = sorted((int(x.strip()) for x in s.split(",") if x.strip()), reverse=True)
    if not values:
        raise argparse.ArgumentTypeError("must be a comma-separated list of integers")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter and resize image groups by resolution thresholds.",
    )
    p.add_argument(
        "-i", "--input", required=True, type=Path,
        help="Directory containing image groups (anchor images + .txt captions + same-basename subdirectories).",
    )
    p.add_argument(
        "-o", "--output", required=True, type=Path,
        help="Destination directory for prepared images.",
    )
    p.add_argument(
        "--base-target-res", type=parse_res_chain, default=[384],
        help="Comma-separated fallback chain for base target res (default: 384).",
    )
    p.add_argument(
        "--base-ref-res", type=parse_res_chain, default=[256],
        help="Comma-separated fallback chain for base ref res (default: 256).",
    )
    p.add_argument(
        "--high-target-res", type=parse_res_chain, default=[512],
        help="Comma-separated fallback chain for high target res (default: 512).",
    )
    p.add_argument(
        "--high-ref-res", type=parse_res_chain, default=[384],
        help="Comma-separated fallback chain for high ref res (default: 384).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print actions without executing.",
    )
    p.add_argument(
        "--require-ref", action="store_true",
        help="Require ref images (subdir) at resolution — skip groups with no ref.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print every action (default: only summary).",
    )
    return p.parse_args()


def log(msg: str, *, verbose: bool = False) -> None:
    if verbose:
        print(msg)


def detect_groups(root: Path) -> dict[str, dict[str, Path | None]]:
    """Return {stem: {image, caption, subdir, text_encoding}} per image group."""
    all_entries = sorted(
        [e for e in root.iterdir() if not e.name.startswith(".")],
        key=lambda p: p.name,
    )
    group_template: dict[str, Path | None] = {
        "image": None, "caption": None, "subdir": None, "text_encoding": None,
    }
    groups: dict[str, dict[str, Path | None]] = {}
    for entry in all_entries:
        if not entry.is_file():
            continue
        stem = entry.stem
        suffix = entry.suffix.lower()
        if suffix in IMAGE_EXTS:
            groups.setdefault(stem, dict(group_template))["image"] = entry
        elif suffix == ".txt":
            groups.setdefault(stem, dict(group_template))["caption"] = entry
    # Detect text encoding: {stem}.text.safetensors
    for entry in all_entries:
        if not (entry.is_file() and entry.suffix.lower() == ".safetensors"):
            continue
        parts = entry.name.split(".")
        if len(parts) >= 3 and parts[-2] == "text":
            stem = ".".join(parts[:-2])
            if stem in groups:
                groups[stem]["text_encoding"] = entry
    for entry in all_entries:
        if entry.is_dir():
            stem = entry.name
            if stem in groups and groups[stem]["subdir"] is None:
                groups[stem]["subdir"] = entry
    return {s: g for s, g in groups.items() if g["image"] is not None}


def image_shortest_side(path: Path) -> int | None:
    """Return the shortest side of an image, or None on failure."""
    try:
        with Image.open(path) as im:
            return min(im.size)
    except Exception:
        return None


def resize_to_fit(src: Path, dst: Path, target: int) -> None:
    """Resize so shortest side = target, preserving aspect ratio (LANCZOS)."""
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        w, h = im.size
        scale = target / min(w, h)
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        resized = im.resize(new_size, Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        resized.save(dst)


def process_subdir_images(
    src_sub: Path, dst_sub: Path, *,
    ref_res: int, dry_run: bool, verbose: bool,
) -> tuple[int, int]:
    """
    Resize each subdir image to --ref-res. Images < ref_res are filtered out.

    Returns (skipped, resized_count).
    """
    skipped = 0
    resized = 0

    for entry in sorted(src_sub.iterdir()):
        if not (entry.is_file() and entry.suffix.lower() in IMAGE_EXTS):
            continue

        side = image_shortest_side(entry)
        if side is None:
            log(f"  SKIP {entry.name} (unreadable)", verbose=verbose)
            skipped += 1
            continue

        if side < ref_res:
            log(f"  SKIP {entry.name} ({side}px < {ref_res})", verbose=verbose)
            skipped += 1
            continue

        dst_entry = dst_sub / entry.name
        if not dry_run:
            resize_to_fit(entry, dst_entry, ref_res)
        log(f"  {entry.name} -> ref ({side}px -> {ref_res})", verbose=verbose)
        resized += 1

    return skipped, resized


def resolve_chain(image_short: int, chain: list[int]) -> int | None:
    """Walk chain descending, return the first res the image meets, or None."""
    for res in chain:
        if image_short >= res:
            return res
    return None


def check_group(
    anchor_side: int,
    subdir_max_short: int | None,
    base_target_chain: list[int],
    base_ref_chain: list[int],
    high_target_chain: list[int],
    high_ref_chain: list[int],
    *,
    require_ref: bool = False,
) -> tuple[int | None, int | None, int | None, int | None]:
    """Return (base_target, base_ref, high_target, high_ref).

    Base thresholds are the gatekeeper — group must pass base to be included.
    If high_target is not None, the group also qualifies for a high/ copy.

    When subdir_max_short is None (no ref images), ref chain checks are
    waived — the group passes base and high on target alone — UNLESS
    require_ref is True, in which case the group is skipped entirely.

    Returns (None, None, None, None) to skip entirely.
    """
    if require_ref and subdir_max_short is None:
        return None, None, None, None
    base_t = resolve_chain(anchor_side, base_target_chain)
    if base_t is None:
        return None, None, None, None

    # Ref chains only matter when there are actual ref images
    base_r = resolve_chain(subdir_max_short, base_ref_chain) if subdir_max_short is not None else base_ref_chain[0]
    if base_r is None:
        return None, None, None, None

    high_t = resolve_chain(anchor_side, high_target_chain)
    high_r = resolve_chain(subdir_max_short, high_ref_chain) if subdir_max_short is not None else high_ref_chain[0]

    return base_t, base_r, high_t, high_r


def copy_group(
    group: dict[str, Path | None],
    output_root: Path,
    output_stem: str,
    target_res: int,
    ref_res: int | None,
    *,
    dry_run: bool,
    verbose: bool,
) -> tuple[int, int]:
    """Copy anchor and optionally subdir to output_root at given resolutions.

    When subdir is missing or ref_res is None, subdir processing is skipped.

    Caption is NOT handled here — base/ gets it in main() and high/ doesn't
    need one (same stem in base has it).

    Returns (skipped_ref, resized_ref).
    """
    src_img = group["image"]
    src_sub = group["subdir"]
    assert src_img is not None

    anchor_side = image_shortest_side(src_img)
    assert anchor_side is not None

    # Resize anchor (target) image
    dst_img = output_root / f"{output_stem}{src_img.suffix.lower()}"
    if not dry_run:
        resize_to_fit(src_img, dst_img, target_res)
    log(f"  {dst_img.name} -> target ({anchor_side}px -> {target_res})", verbose=verbose)

    # Process subdir (reference) images
    if src_sub is not None and ref_res is not None:
        dst_sub = output_root / output_stem
        return process_subdir_images(
            src_sub, dst_sub,
            ref_res=ref_res, dry_run=dry_run, verbose=verbose,
        )
    return 0, 0


def main() -> None:
    args = parse_args()
    root: Path = args.input.expanduser().resolve()
    output: Path = args.output.expanduser().resolve()

    if not root.is_dir():
        sys.exit(f"ERROR: {root} is not a directory.")

    dry_run = args.dry_run
    verbose = args.verbose

    groups = detect_groups(root)
    if not groups:
        print("[prepare] no image groups found")
        return

    if verbose:
        print(f"[prepare] scanning {root}")
        print(f"[prepare] output -> {output}")
        print(f"[prepare] base target={','.join(str(v) for v in args.base_target_res)}"
              f" ref={','.join(str(v) for v in args.base_ref_res)}")
        print(f"[prepare] high target={','.join(str(v) for v in args.high_target_res)}"
              f" ref={','.join(str(v) for v in args.high_ref_res)}")

    base_dir = output / "base"
    high_dir = output / "high"

    counts: dict[str, dict[str, int]] = {
        "base": {"groups": 0, "ref": 0, "skip_ref": 0},
        "high": {"groups": 0, "ref": 0, "skip_ref": 0},
    }
    skipped_groups = 0
    counter = 0

    for stem, g in sorted(groups.items()):
        src_img = g["image"]
        src_sub = g["subdir"]

        if src_img is None:
            log(f"  SKIP group '{stem}' (no anchor)", verbose=verbose)
            skipped_groups += 1
            continue

        anchor_side = image_shortest_side(src_img)
        if anchor_side is None:
            log(f"  SKIP group '{stem}' (unreadable anchor)", verbose=verbose)
            skipped_groups += 1
            continue

        if src_sub is None or not src_sub.exists():
            # No ref images — treat as anchor-only group
            if verbose:
                print(f"  {stem}: anchor only (no subdir)")

        # Measure subdir images to find max short side (if any)
        subdir_max: int | None = None
        if src_sub is not None and src_sub.exists():
            subdir_shorts: list[int] = []
            for entry in sorted(src_sub.iterdir()):
                if entry.is_file() and entry.suffix.lower() in IMAGE_EXTS:
                    s = image_shortest_side(entry)
                    if s is not None:
                        subdir_shorts.append(s)
            subdir_max = max(subdir_shorts) if subdir_shorts else 0

        base_t, base_r, high_t, high_r = check_group(
            anchor_side, subdir_max,
            args.base_target_res, args.base_ref_res,
            args.high_target_res, args.high_ref_res,
            require_ref=args.require_ref,
        )
        if base_t is None:
            log(f"  SKIP group '{stem}' (anchor={anchor_side}, subdir_max={subdir_max} below thresholds)", verbose=verbose)
            skipped_groups += 1
            continue

        counter += 1
        output_stem = f"{counter:07d}"

        if verbose:
            print(f"\n[prepare] group: {stem} -> {output_stem}")

        # Copy to base
        s, r = copy_group(
            g, base_dir, output_stem, base_t, base_r,
            dry_run=dry_run, verbose=verbose,
        )
        counts["base"]["groups"] += 1
        counts["base"]["ref"] += r
        counts["base"]["skip_ref"] += s

        # Copy caption to base only (high uses same stem)
        src_cap = g["caption"]
        if src_cap is not None:
            dst_cap = base_dir / f"{output_stem}.txt"
            if not dst_cap.exists():
                if not dry_run:
                    dst_cap.parent.mkdir(parents=True, exist_ok=True)
                    with open(src_cap) as f_src, open(dst_cap, "w") as f_dst:
                        f_dst.write(f_src.read())
                log(f"  {dst_cap.name} (caption)", verbose=verbose)

        # Copy pre-computed text encoding if available
        src_enc = g["text_encoding"]
        if src_enc is not None:
            dst_enc = base_dir / f"{output_stem}.text.safetensors"
            if not dry_run:
                dst_enc.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_enc, dst_enc)
            log(f"  {dst_enc.name} (text encoding)", verbose=verbose)

        # Also copy to high if it qualifies
        if high_t is not None:
            if args.require_ref and high_r is None:
                log(f"  SKIP {output_stem} high/ (ref below high ref res)", verbose=verbose)
            else:
                s, r = copy_group(
                    g, high_dir, output_stem, high_t, high_r,
                    dry_run=dry_run, verbose=verbose,
                )
                counts["high"]["groups"] += 1
                counts["high"]["ref"] += r
                counts["high"]["skip_ref"] += s

    # Summary
    print()
    for name in ("high", "base"):
        c = counts[name]
        if c["groups"]:
            print(f"[prepare] {name}/: {c['groups']} group{'s' if c['groups'] != 1 else ''}, "
                  f"{c['ref']} ref images, {c['skip_ref']} filtered")
    if skipped_groups:
        print(f"[prepare] {skipped_groups} group{'s' if skipped_groups != 1 else ''} skipped (below thresholds)")
    if dry_run:
        print("[prepare] DRY RUN -- no files were written")


if __name__ == "__main__":
    main()
