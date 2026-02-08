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
) -> tuple[int, int]:
    best_key = None
    best_size = None
    input_ar = width / height
    for rw, rh in ratios:
        base_w = rw * length_multiple
        base_h = rh * length_multiple
        ratio_ar = rw / rh
        k = int(math.floor(min(width / base_w, height / base_h)))
        if k < 1:
            k = 1
        tw = base_w * k
        th = base_h * k
        err_size = abs(tw - width) / width + abs(th - height) / height
        err_ar = abs(ratio_ar - input_ar)
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
        im.save(tmp, format="JPEG", quality=95, subsampling="keep", optimize=True)
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
) -> None:
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            w, h = im.size
            if w <= 0 or h <= 0:
                return

            if mode == "side":
                if side == "width":
                    target_w = size
                    target_h = max(1, int(round(size * (h / w))))
                else:
                    target_h = size
                    target_w = max(1, int(round(size * (w / h))))
            else:
                target_w, target_h = pick_ratio_size(w, h, ratios or [], length_multiple or 1)

            out_path = out_dir / path.name
            if (target_w, target_h) == (w, h):
                if out_path.resolve() == path.resolve():
                    return
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out_path)
                return

            new = im.resize((target_w, target_h), Image.LANCZOS)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(new, out_path)
    except Exception as e:
        print(f"skip {path} ({e})")


def main() -> None:
    p = argparse.ArgumentParser(description="Non-crop reshape of images.")
    p.add_argument("-i", "--input", required=True, type=Path, help="Directory with images.")
    p.add_argument("-o", "--output", required=True, type=Path, help="Output directory.")
    p.add_argument("--mode", choices=["side", "ratio"], default="side", help="Reshape mode.")
    p.add_argument("--side", choices=["width", "height"], help="Which side to set in side mode.")
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
    args = p.parse_args()

    inp: Path = args.input.expanduser().resolve()
    if not inp.is_dir():
        sys.exit(f"ERROR: {inp} is not a directory.")
    out: Path = args.output.expanduser().resolve()
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
    for f in files:
        reshape_file(
            f,
            out,
            args.mode,
            args.side,
            args.size,
            ratios,
            length_multiple,
        )


if __name__ == "__main__":
    main()
