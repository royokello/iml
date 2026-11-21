#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from krea_2.config import resolve_model_root
from krea_2.files import copy_files

CONFIG_FILE_NAMES = ("config.json",)
TOKENIZER_FILE_NAMES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def _check_source(source_dir: str | Path, names: tuple[str, ...]) -> None:
    source = Path(source_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source}")
    missing = [name for name in names if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Files not found in {source}: {', '.join(missing)}")


def copy_text_encoder_files(
    root: str | Path,
    input_dir: str | Path,
    tokenizer_dir: str | Path,
) -> list[Path]:
    """Copy the text encoder config and tokenizer files into the model folder.

    Both source folders are checked before anything is copied.

    Args:
        root: The root folder. The files go to `<root>/krea_2/model/text_encoder`.
        input_dir: The source folder that holds `config.json`.
        tokenizer_dir: The source folder that holds the tokenizer files.

    Returns:
        The paths of the files in the target folder.

    Raises:
        FileNotFoundError: If a source folder or a listed file is missing.
            No file is copied in that case.
    """
    target_dir = resolve_model_root(root) / "text_encoder"
    print(
        "Copy text encoder files:\n"
        f"  input={input_dir}\n"
        f"  tokenizer={tokenizer_dir}\n"
        f"  target={target_dir}"
    )

    _check_source(input_dir, CONFIG_FILE_NAMES)
    _check_source(tokenizer_dir, TOKENIZER_FILE_NAMES)

    config_files = copy_files(input_dir, target_dir, CONFIG_FILE_NAMES)
    tokenizer_files = copy_files(tokenizer_dir, target_dir, TOKENIZER_FILE_NAMES)
    return config_files + tokenizer_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy the Krea 2 text encoder files.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder. The files go to <root>/krea_2/model/text_encoder/. The folder is created if missing.",
    )
    parser.add_argument("--input", required=True, help="Source text_encoder folder.")
    parser.add_argument("--tokenizer", required=True, help="Source tokenizer folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copy_text_encoder_files(args.root, args.input, args.tokenizer)
    print("Done")


if __name__ == "__main__":
    main()
