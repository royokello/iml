from __future__ import annotations

from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

__all__ = [
    "IMAGE_SUFFIXES",
    "load_dataset",
    "load_flux2_dataset",
]


def _is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _list_image_files(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.iterdir() if _is_supported_image(path)),
        key=lambda path: path.name,
    )


def load_flux2_dataset(dataset_dir: str | Path) -> dict[str, list[str | Path | None] | list[list[Path]]]:
    root = Path(dataset_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    target_images = _list_image_files(root)
    if not target_images:
        raise FileNotFoundError(f"No supported target images found in dataset: {root}")

    stems_seen: set[str] = set()
    duplicate_stems: list[str] = []
    for image_path in target_images:
        if image_path.stem in stems_seen:
            duplicate_stems.append(image_path.stem)
        stems_seen.add(image_path.stem)
    if duplicate_stems:
        duplicate_text = ", ".join(sorted(set(duplicate_stems)))
        raise ValueError(f"Dataset contains duplicate target image stems: {duplicate_text}")

    text_prompts: list[str | None] = []
    reference_images: list[list[Path]] = []
    for target_image in target_images:
        prompt_path = target_image.with_suffix(".txt")
        if prompt_path.is_file():
            text_prompts.append(prompt_path.read_text(encoding="utf-8").strip())
        else:
            text_prompts.append(None)

        references_dir = root / target_image.stem
        if references_dir.is_dir():
            reference_images.append(_list_image_files(references_dir))
        else:
            reference_images.append([])

    return {
        "text_prompts": text_prompts,
        "target_images": target_images,
        "reference_images": reference_images,
    }


load_dataset = load_flux2_dataset
