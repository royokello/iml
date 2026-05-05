from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
NUMBER_RE = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class ImagePair:
    image: Path
    caption: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a dataset by pairing input images with random reference captions."
    )
    p.add_argument(
        "--inputs",
        required=True,
        type=Path,
        help="Directory containing input images to copy into the output dataset.",
    )
    p.add_argument(
        "--refs",
        required=True,
        type=Path,
        help="Directory containing reference image + .txt caption pairs.",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory where dataset items will be appended.",
    )
    p.add_argument(
        "--filename-width",
        type=parse_positive_int,
        default=6,
        help="Zero-pad numeric output filenames to this width. Default: 6",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for repeatable ref selection.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned writes without copying files.",
    )
    return p.parse_args()


def parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def iter_images(root: Path) -> list[Path]:
    return sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.relative_to(root).as_posix().lower(),
    )


def scan_ref_pairs(root: Path) -> list[ImagePair]:
    pairs: list[ImagePair] = []
    for image in iter_images(root):
        caption = image.with_suffix(".txt")
        if caption.is_file():
            pairs.append(ImagePair(image=image, caption=caption))
    return pairs


def numeric_stem(path: Path) -> int | None:
    match = NUMBER_RE.search(path.stem)
    if match is None:
        return None
    return int(match.group(1))


def next_output_index(output: Path) -> int:
    highest = 0
    if not output.exists():
        return 1

    for image in output.iterdir():
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTS:
            continue
        number = numeric_stem(image)
        if number is not None:
            highest = max(highest, number)
    return highest + 1


def copy_dataset_item(
    input_image: Path,
    ref_pair: ImagePair,
    output: Path,
    output_stem: str,
    dry_run: bool,
) -> None:
    output_image = output / f"{output_stem}{input_image.suffix.lower()}"
    output_caption = output / f"{output_stem}.txt"
    ref_output_dir = output / output_stem
    ref_output_image = ref_output_dir / ref_pair.image.name

    if dry_run:
        print(f"[dataset] {input_image} -> {output_image}")
        print(f"[dataset] {ref_pair.caption} -> {output_caption}")
        print(f"[dataset] {ref_pair.image} -> {ref_output_image}")
        return

    output.mkdir(parents=True, exist_ok=True)
    ref_output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(input_image, output_image)
    shutil.copy2(ref_pair.caption, output_caption)
    shutil.copy2(ref_pair.image, ref_output_image)


def run(
    inputs: Path,
    refs: Path,
    output: Path,
    filename_width: int,
    seed: int | None,
    dry_run: bool,
) -> None:
    inputs = inputs.expanduser().resolve()
    refs = refs.expanduser().resolve()
    output = output.expanduser().resolve()

    if not inputs.is_dir():
        raise ValueError(f"--inputs is not a directory: {inputs}")
    if not refs.is_dir():
        raise ValueError(f"--refs is not a directory: {refs}")
    if output == inputs or inputs in output.parents:
        raise ValueError("--output cannot be the same directory as --inputs or a subdirectory of --inputs.")
    if output == refs or refs in output.parents:
        raise ValueError("--output cannot be the same directory as --refs or a subdirectory of --refs.")

    input_images = iter_images(inputs)
    if not input_images:
        raise ValueError("no input images found.")

    ref_pairs = scan_ref_pairs(refs)
    if not ref_pairs:
        raise ValueError("no reference image + .txt caption pairs found.")
    if len(input_images) > len(ref_pairs):
        raise ValueError(
            f"not enough reference pairs for non-repeating selection: "
            f"{len(input_images)} input image(s), {len(ref_pairs)} reference pair(s)."
        )

    rng = random.Random(seed)
    selected_ref_pairs = ref_pairs[:]
    rng.shuffle(selected_ref_pairs)
    next_index = next_output_index(output)

    copied = 0
    for offset, (input_image, ref_pair) in enumerate(zip(input_images, selected_ref_pairs)):
        output_stem = f"{next_index + offset:0{filename_width}d}"
        copy_dataset_item(
            input_image=input_image,
            ref_pair=ref_pair,
            output=output,
            output_stem=output_stem,
            dry_run=dry_run,
        )
        copied += 1

    action = "planned" if dry_run else "created"
    print(f"[dataset] {action} {copied} item(s)")
    print(f"[dataset] start index: {next_index}")
    print(f"[dataset] output: {output}")


def main() -> None:
    args = parse_args()
    try:
        run(
            inputs=args.inputs,
            refs=args.refs,
            output=args.output,
            filename_width=args.filename_width,
            seed=args.seed,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
