from __future__ import annotations

import argparse
from pathlib import Path

from flux_2_klein_4b.gen import generate_image, prepare_prompt_conditioning


def _list_epoch_checkpoints(models_dir: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for checkpoint_path in models_dir.glob("epoch_*.safetensors"):
        epoch_text = checkpoint_path.stem.removeprefix("epoch_")
        try:
            epoch_number = int(epoch_text)
        except ValueError:
            continue
        checkpoints.append((epoch_number, checkpoint_path))

    checkpoints.sort(key=lambda item: item[0])
    if not checkpoints:
        raise FileNotFoundError(f"No epoch checkpoints found in {models_dir}")
    return checkpoints


def _next_prompt_dir(sample_dir: Path) -> Path:
    next_index = 1
    for path in sample_dir.iterdir():
        if not path.is_dir():
            continue
        if not path.name.startswith("prompt_"):
            continue

        suffix = path.name.removeprefix("prompt_")
        try:
            prompt_index = int(suffix)
        except ValueError:
            continue
        next_index = max(next_index, prompt_index + 1)

    return sample_dir / f"prompt_{next_index}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one sample image per Flux 2 Klein 4B training epoch.")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root folder that contains flux_2_klein_4b/model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Training output directory that contains the models subdirectory created by flux_2_klein_4b.train.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt used for every checkpoint sample render.",
    )
    parser.add_argument(
        "--images",
        help='Optional comma-separated image paths to encode, for example "img1.png, img2.png".',
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Sample image width. Defaults to 512.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Sample image height. Defaults to 512.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output.expanduser().resolve()
    models_dir = output_dir / "models"
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Expected training models directory at {models_dir}")

    checkpoints = _list_epoch_checkpoints(models_dir)
    sample_dir = output_dir / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = _next_prompt_dir(sample_dir)
    prompt_dir.mkdir(parents=True, exist_ok=False)

    print(f"Sampling {len(checkpoints)} checkpoints from {models_dir}")
    print(f"Sample output dir: {prompt_dir}")
    conditioning = prepare_prompt_conditioning(
        args.root,
        prompt=args.prompt,
    )
    for epoch_number, checkpoint_path in checkpoints:
        sample_path = prompt_dir / f"epoch_{epoch_number}.png"
        print(f"* epoch {epoch_number}: {checkpoint_path}")
        generate_image(
            args.root,
            image=args.images,
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            conditioning=conditioning,
            loras={checkpoint_path: 1.0},
            output_path=sample_path,
        )


if __name__ == "__main__":
    main()
