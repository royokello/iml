#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from flux3.config import resolve_model_root
from flux3.files import copy_files

FILE_NAMES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.json",
)


def copy_text_encoder_files(root: str | Path, input_dir: str | Path) -> list[Path]:
    """Copy the text encoder files into the Flux 3 model folder.

    Args:
        root: The root folder. The files go to `<root>/flux3/model/text_encoder`.
        input_dir: The source folder that holds the files.

    Returns:
        The paths of the files in the target folder.

    Raises:
        FileNotFoundError: If the source folder or a listed file is missing.
    """
    target_dir = resolve_model_root(root) / "text_encoder"
    print(f"Copy text encoder files:\n  input={input_dir}\n  target={target_dir}")
    return copy_files(input_dir, target_dir, FILE_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy the Flux 3 text encoder files.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder. The files go to <root>/flux3/model/text_encoder/. The folder is created if missing.",
    )
    parser.add_argument("--input", required=True, help="Source text_encoder folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copy_text_encoder_files(args.root, args.input)
    print("Done")


if __name__ == "__main__":
    main()
