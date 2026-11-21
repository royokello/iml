#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from krea_2.config import resolve_model_root
from krea_2.files import copy_files

FILE_NAMES = ("config.json",)


def copy_denoiser_files(root: str | Path, input_dir: str | Path) -> list[Path]:
    """Copy the denoiser files into the Krea 2 model folder.

    Args:
        root: The root folder. The files go to `<root>/krea_2/model/denoiser`.
        input_dir: The source folder that holds the files.

    Returns:
        The paths of the files in the target folder.

    Raises:
        FileNotFoundError: If the source folder or a listed file is missing.
    """
    target_dir = resolve_model_root(root) / "denoiser"
    print(f"Copy denoiser files:\n  input={input_dir}\n  target={target_dir}")
    return copy_files(input_dir, target_dir, FILE_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy the Krea 2 denoiser files.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder. The files go to <root>/krea_2/model/denoiser/. The folder is created if missing.",
    )
    parser.add_argument("--input", required=True, help="Source transformer folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copy_denoiser_files(args.root, args.input)
    print("Done")


if __name__ == "__main__":
    main()
