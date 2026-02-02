# image/collect.py
"""
Collect and convert images from a directory tree into a single folder.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}


def main(
    input_dir: str,
    output_dir: str,
    width: int | None = None,
    height: int | None = None,
    square: bool = False,
    mode: str = "folder",
) -> None:
    """
    Collects all image files (and their captions, if present) from a directory and its
    subdirectories, converts them to PNG, optionally resizes them according to the
    specified width and/or height, and saves the result.

    If both width and height are provided, the image is resized to exactly width x height.
    If only one dimension is provided, the other is scaled proportionally.
    If no dimensions are provided, the original resolution is preserved.

    If --square is provided, the final output image will be placed on a white square
    background whose side length is the larger dimension of the resized image (or the
    explicitly requested dimension if both are specified).

    Modes:
        - folder: gather files in folder order as returned by os.walk
        - file: gather files sorted by filename
        - random: gather files in random order
    """
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target width: {width}")
    print(f"Target height: {height}")
    print(f"Square output: {square}")
    print(f"Mode: {mode}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files: list[str] = []
    for root, _dirs, files in os.walk(input_dir):
        for file in files:
            if Path(file).suffix.lower() in IMAGE_EXTS:
                image_files.append(os.path.join(root, file))

    if mode == "file":
        image_files.sort(key=lambda x: os.path.basename(x).lower())
    elif mode == "random":
        random.shuffle(image_files)

    file_counter = 1
    for file_path in image_files:
        try:
            with Image.open(file_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")

                original_width, original_height = img.size
                new_width, new_height = original_width, original_height

                if width is not None and height is not None:
                    new_width = width
                    new_height = height
                elif width is not None:
                    ratio = width / original_width
                    new_width = width
                    new_height = int(original_height * ratio)
                elif height is not None:
                    ratio = height / original_height
                    new_width = int(original_width * ratio)
                    new_height = height

                if (new_width, new_height) != (original_width, original_height):
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                if square:
                    side = max(new_width, new_height)
                    square_img = Image.new("RGB", (side, side), (255, 255, 255))
                    paste_x = (side - new_width) // 2
                    paste_y = (side - new_height) // 2
                    square_img.paste(img, (paste_x, paste_y))
                    final_img = square_img
                else:
                    final_img = img

                output_file_path = out_dir / f"{file_counter}.png"
                final_img.save(output_file_path, "PNG")

                caption_file_path = os.path.splitext(file_path)[0] + ".txt"
                if os.path.exists(caption_file_path):
                    with open(caption_file_path, "r", encoding="utf-8") as caption_file:
                        caption_text = caption_file.read()
                    caption_output_path = out_dir / f"{file_counter}.txt"
                    with open(caption_output_path, "w", encoding="utf-8") as output_caption_file:
                        output_caption_file.write(caption_text)

                print(f"Converted and saved: {output_file_path}")
                file_counter += 1
        except Exception as exc:
            print(f"Error processing {file_path}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and convert images to PNG format.")
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Directory where the image files are located.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the PNG files will be saved.",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Target width for the images. If only width is provided, the height is scaled proportionally.",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Target height for the images. If only height is provided, the width is scaled proportionally.",
    )
    parser.add_argument(
        "--square",
        action="store_true",
        help="If provided, output images are placed on a square white background.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["folder", "file", "random"],
        default="folder",
        help="Mode to gather files: folder, file, or random.",
    )
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        square=args.square,
        mode=args.mode,
    )


if __name__ == "__main__":
    cli()
