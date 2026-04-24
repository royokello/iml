# image/shape.py
from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
TMP_MARKER = ".__tmp__"


def parse_ratio_token(token: str) -> tuple[int, int]:
    cleaned = token.strip().lower()
    if not cleaned:
        raise ValueError("empty ratio")
    if "x" in cleaned:
        parts = cleaned.split("x", 1)
    elif ":" in cleaned:
        parts = cleaned.split(":", 1)
    else:
        raise ValueError(f"invalid ratio '{token}' (expected NxM)")
    w = int(parts[0])
    h = int(parts[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid ratio '{token}' (must be positive)")
    g = math.gcd(w, h)
    return w // g, h // g


def parse_ratios(values: list[str]) -> list[tuple[int, int]]:
    tokens: list[str] = []
    for value in values:
        for token in re.split(r"[,\s]+", value.strip()):
            if token:
                tokens.append(token)
    if not tokens:
        raise ValueError("no ratios provided")
    parsed: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for token in tokens:
        ratio = parse_ratio_token(token)
        if ratio in seen:
            continue
        seen.add(ratio)
        parsed.append(ratio)
    return parsed


def pick_ratio_size(
    width: int,
    height: int,
    ratios: list[tuple[int, int]],
    length_multiple: int,
    min_short_side: int | None = None,
    max_long_side: int | None = None,
    mid_square_side: int | None = None,
) -> tuple[int, int]:
    best_key = None
    best_size = None
    input_ar = width / height
    for rw, rh in ratios:
        if mid_square_side is not None and rw == rh:
            square_side = min(mid_square_side, min(width, height))
            if min_short_side is not None and square_side >= min_short_side:
                square_side = max(
                    length_multiple,
                    int((square_side + length_multiple / 2) // length_multiple)
                    * length_multiple,
                )
                square_side = min(square_side, mid_square_side)
            tw = square_side
            th = square_side
        elif min_short_side is not None:
            desired_short = min_short_side
            short_ratio = min(rw, rh)
            long_ratio = max(rw, rh)
            exact_long = desired_short * long_ratio / short_ratio
            snapped_long = max(
                length_multiple,
                int(math.floor(exact_long / length_multiple)) * length_multiple,
            )
            if rw >= rh:
                tw, th = snapped_long, desired_short
            else:
                tw, th = desired_short, snapped_long
        elif max_long_side is not None:
            desired_long = max_long_side
            short_ratio = min(rw, rh)
            long_ratio = max(rw, rh)
            exact_short = desired_long * short_ratio / long_ratio
            snapped_short = max(
                length_multiple,
                int(math.floor(exact_short / length_multiple)) * length_multiple,
            )
            if rw >= rh:
                tw, th = desired_long, snapped_short
            else:
                tw, th = snapped_short, desired_long
        else:
            base_w = rw * length_multiple
            base_h = rh * length_multiple
            fit_scale = min(width / base_w, height / base_h)
            k = max(1, int(math.floor(fit_scale)))
            tw = base_w * k
            th = base_h * k

        err_size = abs(tw - width) / width + abs(th - height) / height
        err_ar = abs((tw / th) - input_ar)
        key = (err_ar, err_size)
        if best_key is None or key < best_key:
            best_key = key
            best_size = (tw, th)
    if best_size is None:
        return width, height
    return best_size


def save_image(im: Image.Image, out_path: Path) -> None:
    ext = out_path.suffix.lower()
    tmp = out_path.with_name(out_path.stem + TMP_MARKER + ext)

    if ext in {".jpg", ".jpeg"}:
        im = im.convert("RGB")
        im.save(tmp, format="JPEG", quality=95, subsampling=0, optimize=True)
    elif ext == ".png":
        im.save(tmp, format="PNG", optimize=True)
    elif ext == ".webp":
        im.save(tmp, format="WEBP", quality=95, method=6)
    elif ext in {".tif", ".tiff"}:
        im.save(tmp, format="TIFF")
    else:
        im.save(tmp)

    tmp.replace(out_path)


def reshape_file(
    path: Path,
    out_dir: Path,
    mode: str,
    side: str | None,
    size: int | None,
    ratios: list[tuple[int, int]] | None,
    length_multiple: int | None,
    min_short_side: int | None,
    max_long_side: int | None,
    mid_square_side: int | None,
) -> tuple[str, tuple[int, int], tuple[int, int]] | None:
    out_path = out_dir / path.name
    if out_path.exists() and out_path.resolve() != path.resolve():
        return "exists", (0, 0), (0, 0)
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            w, h = im.size
            if w <= 0 or h <= 0:
                return
            original_size = (w, h)

            if mode == "side":
                if side == "longest":
                    if w >= h:
                        target_w = size
                        target_h = max(1, int(round(size * (h / w))))
                    else:
                        target_h = size
                        target_w = max(1, int(round(size * (w / h))))
                elif side == "width":
                    target_w = size
                    target_h = max(1, int(round(size * (h / w))))
                else:
                    target_h = size
                    target_w = max(1, int(round(size * (w / h))))
            else:
                target_w, target_h = pick_ratio_size(
                    w,
                    h,
                    ratios or [],
                    length_multiple or 1,
                    min_short_side,
                    max_long_side,
                    mid_square_side,
                )
            target_size = (target_w, target_h)

            if (target_w, target_h) == (w, h):
                if out_path.resolve() == path.resolve():
                    return "ok", original_size, target_size
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out_path)
                return "ok", original_size, target_size

            if mode == "ratio":
                new = ImageOps.fit(im, (target_w, target_h), method=Image.LANCZOS)
            else:
                new = im.resize((target_w, target_h), Image.LANCZOS)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(new, out_path)
            return "ok", original_size, target_size
    except Exception as e:
        print(f"skip {path} ({e})")
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Resize or crop images to target shapes.")
    p.add_argument("-i", "--input", required=True, type=Path, help="Directory with images.")
    out_group = p.add_mutually_exclusive_group()
    out_group.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory. Required unless --inplace is set.",
    )
    out_group.add_argument(
        "--inplace",
        action="store_true",
        help="Rewrite images in the input directory instead of writing to output.",
    )
    p.add_argument("--mode", choices=["side", "ratio"], default="side", help="Reshape mode.")
    p.add_argument(
        "--side",
        choices=["width", "height", "longest"],
        help="Which side to set in side mode. longest uses the larger of width/height per image.",
    )
    p.add_argument("--size", type=int, help="Target pixels for chosen side in side mode.")
    p.add_argument(
        "--ratios",
        nargs="+",
        default=["1x1", "3x4", "4x3", "1x2"],
        help="Ratios for ratio mode (e.g. 1x1 3x4 or 1x1,3x4).",
    )
    p.add_argument(
        "-lm",
        "--length-multiple",
        type=int,
        default=64,
        help="Enforce width/height as multiples of this value in ratio mode.",
    )
    p.add_argument(
        "--min",
        type=int,
        default=None,
        help="Set the shortest side to this size in ratio mode.",
    )
    p.add_argument(
        "--max",
        type=int,
        default=None,
        help="Set the longest side to this size in ratio mode.",
    )
    p.add_argument(
        "--mid",
        type=int,
        default=None,
        help=(
            "Set square outputs to this side length in ratio mode. "
            "Square inputs at or above --min are snapped to the length multiple and capped at --mid. "
            "Can be paired with either --min or --max."
        ),
    )
    args = p.parse_args()

    inp: Path = args.input.expanduser().resolve()
    if not inp.is_dir():
        sys.exit(f"ERROR: {inp} is not a directory.")
    if args.inplace:
        out = inp
    else:
        if args.output is None:
            sys.exit("ERROR: --output is required unless --inplace is set.")
        out = args.output.expanduser().resolve()
        if out.exists() and not out.is_dir():
            sys.exit(f"ERROR: {out} is not a directory.")
        out.mkdir(parents=True, exist_ok=True)

    if args.mode == "side":
        if args.side is None or args.size is None:
            sys.exit("ERROR: --side and --size are required when --mode side.")
        if args.size <= 0:
            sys.exit("ERROR: --size must be a positive integer.")
        ratios = None
        length_multiple = None
    else:
        if args.length_multiple <= 0:
            sys.exit("ERROR: --length-multiple must be a positive integer.")
        if args.min is not None and args.min <= 0:
            sys.exit("ERROR: --min must be a positive integer.")
        if args.max is not None and args.max <= 0:
            sys.exit("ERROR: --max must be a positive integer.")
        if args.mid is not None and args.mid <= 0:
            sys.exit("ERROR: --mid must be a positive integer.")
        if args.min is not None and args.max is not None:
            sys.exit("ERROR: --min and --max cannot be used together. Use either one, optionally with --mid.")
        if args.min is not None and args.min % args.length_multiple != 0:
            sys.exit("ERROR: --min must be a multiple of --length-multiple.")
        if args.max is not None and args.max % args.length_multiple != 0:
            sys.exit("ERROR: --max must be a multiple of --length-multiple.")
        if args.mid is not None and args.mid % args.length_multiple != 0:
            sys.exit("ERROR: --mid must be a multiple of --length-multiple.")
        try:
            ratios = parse_ratios(args.ratios)
        except ValueError as e:
            sys.exit(f"ERROR: {e}")
        length_multiple = args.length_multiple

    files = [
        f
        for f in inp.iterdir()
        if f.is_file()
        and f.suffix.lower() in IMAGE_EXTS
        and TMP_MARKER not in f.name
    ]
    total = len(files)
    for i, f in enumerate(files, start=1):
        result = reshape_file(
            f,
            out,
            args.mode,
            args.side,
            args.size,
            ratios,
            length_multiple,
            args.min,
            args.max,
            args.mid,
        )
        if result is None:
            continue
        status, (current_w, current_h), (new_w, new_h) = result
        if status == "exists":
            print(f"{i}/{total}: {f}. skip existing output")
            continue
        print(f"{i}/{total}: {f}. {current_w}x{current_h} -> {new_w}x{new_h}")


if __name__ == "__main__":
    main()
