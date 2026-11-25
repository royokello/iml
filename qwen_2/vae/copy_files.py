#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from qwen_2.config import resolve_model_root
from qwen_2.files import copy_files

FILE_NAMES = ("config.json", "diffusion_pytorch_model.safetensors")


def copy_vae_files(root: str | Path, input_dir: str | Path) -> list[Path]:
    """Copy the VAE config and weights into the model folder.

    Args:
        root: The root folder. The files go to `<root>/qwen_2/model/vae`.
        input_dir: The source VAE folder that holds the files.

    Returns:
        The paths of the files in the target folder.

    Raises:
        FileNotFoundError: If the source folder or a listed file is missing.
    """
    target_dir = resolve_model_root(root) / "vae"
    print(f"Copy VAE files:\n  input={input_dir}\n  target={target_dir}")
    return copy_files(input_dir, target_dir, FILE_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy the Qwen Image 2.1 VAE files.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder. The files go to <root>/qwen_2/model/vae/. The folder is created if missing.",
    )
    parser.add_argument("--input", required=True, help="Source VAE folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copy_vae_files(args.root, args.input)
    print("Done")


if __name__ == "__main__":
    main()
