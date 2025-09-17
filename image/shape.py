# image/shape.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def reshape_file(path: Path, side: str, size: int) -> None:
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            w, h = im.size
            if side == "width":
                if w == size:
                    return
                new = im.resize((size, h), Image.LANCZOS)
            else:
                if h == size:
                    return
                new = im.resize((w, size), Image.LANCZOS)

            ext = path.suffix.lower()
            tmp = path.with_name(path.stem + ".__tmp__" + ext)

            if ext in {".jpg", ".jpeg"}:
                new = new.convert("RGB")
                new.save(tmp, format="JPEG", quality=95, subsampling="keep", optimize=True)
            elif ext == ".png":
                new.save(tmp, format="PNG", optimize=True)
            elif ext == ".webp":
                new.save(tmp, format="WEBP", quality=95, method=6)
            elif ext in {".tif", ".tiff"}:
                new.save(tmp, format="TIFF")
            else:
                new.save(tmp)

            tmp.replace(path)
    except Exception as e:
        print(f"skip {path} ({e})")


def main() -> None:
    p = argparse.ArgumentParser(description="Non-crop reshape of images along one side.")
    p.add_argument("--input", required=True, type=Path, help="Directory with images.")
    p.add_argument("--side", required=True, choices=["width", "height"], help="Which side to set to --size.")
    p.add_argument("--size", required=True, type=int, help="Target pixels for chosen side.")
    args = p.parse_args()

    inp: Path = args.input.expanduser().resolve()
    if not inp.is_dir():
        sys.exit(f"ERROR: {inp} is not a directory.")

    for f in inp.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            reshape_file(f, args.side, args.size)


if __name__ == "__main__":
    main()
