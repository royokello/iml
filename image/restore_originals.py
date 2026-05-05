from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace top-level dataset images with matching original-size images from stage clean folders."
        )
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        required=True,
        help='Root containing folders like "d/clean", "m/clean", etc.',
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help='Dataset root containing files like "000001.png" and folders like "000001/".',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned replacements without copying files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print skipped items as well as replacements.",
    )
    return parser.parse_args()


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def build_clean_index(stage_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: list[str] = []

    for path in stage_root.rglob("*"):
        if not is_image(path) or path.parent.name != "clean":
            continue
        key = path.name.lower()
        if key in index:
            duplicates.append(path.name)
            continue
        index[key] = path

    if duplicates:
        duplicate_text = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(f"duplicate clean image names found: {duplicate_text}")

    return index


def iter_dataset_items(dataset_root: Path) -> list[tuple[Path, Path]]:
    items: list[tuple[Path, Path]] = []

    for path in sorted(dataset_root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        subdir = dataset_root / path.stem
        if not subdir.is_dir():
            continue

        refs = [candidate for candidate in sorted(subdir.iterdir()) if is_image(candidate)]
        if len(refs) != 1:
            raise ValueError(
                f"expected exactly one reference image in {subdir}, found {len(refs)}"
            )
        items.append((path, refs[0]))

    return items


def run(stage_root: Path, dataset_root: Path, dry_run: bool, verbose: bool) -> None:
    stage_root = stage_root.expanduser().resolve()
    dataset_root = dataset_root.expanduser().resolve()

    if not stage_root.is_dir():
        raise ValueError(f"--stage-root is not a directory: {stage_root}")
    if not dataset_root.is_dir():
        raise ValueError(f"--dataset-root is not a directory: {dataset_root}")

    clean_index = build_clean_index(stage_root)
    dataset_items = iter_dataset_items(dataset_root)

    replaced = 0
    skipped = 0

    for target_image, ref_image in dataset_items:
        clean_match = clean_index.get(ref_image.name.lower())
        if clean_match is None:
            skipped += 1
            if verbose:
                print(f"[skip] no clean match for {target_image.name} via {ref_image.name}")
            continue

        if dry_run:
            print(f"[replace] {clean_match} -> {target_image}")
        else:
            shutil.copy2(clean_match, target_image)
        replaced += 1

    print(f"[restore_originals] dataset items: {len(dataset_items)}")
    print(f"[restore_originals] clean images indexed: {len(clean_index)}")
    print(f"[restore_originals] replaced: {replaced}")
    print(f"[restore_originals] skipped: {skipped}")
    print(f"[restore_originals] mode: {'dry-run' if dry_run else 'write'}")


def main() -> None:
    args = parse_args()
    try:
        run(
            stage_root=args.stage_root,
            dataset_root=args.dataset_root,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
