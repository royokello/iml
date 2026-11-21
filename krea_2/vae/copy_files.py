#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from krea_2.config import VAE_CONFIG, resolve_model_root
from krea_2.files import copy_files

FILE_NAMES = ("diffusion_pytorch_model.safetensors",)
CONFIG_NAME = "config.json"


def copy_vae_files(root: str | Path, input_dir: str | Path) -> list[Path]:
    """Copy the VAE weights and write the VAE config into the model folder.

    The source folder has no `config.json`. The config is written from
    `VAE_CONFIG`, which holds the Qwen-Image VAE values.

    Args:
        root: The root folder. The files go to `<root>/krea_2/model/vae`.
        input_dir: The source folder that holds the weights.

    Returns:
        The paths of the files in the target folder.

    Raises:
        FileNotFoundError: If the source folder or the weight file is missing.
            No file is copied in that case.
    """
    target_dir = resolve_model_root(root) / "vae"
    print(f"Copy VAE files:\n  input={input_dir}\n  target={target_dir}")

    copied = copy_files(input_dir, target_dir, FILE_NAMES)
    config_path = target_dir / CONFIG_NAME
    config_text = json.dumps(VAE_CONFIG, indent=2) + "\n"
    if config_path.is_file() and config_path.read_text(encoding="utf-8") == config_text:
        print(f" * skip {CONFIG_NAME} (already there)")
    else:
        config_path.write_text(config_text, encoding="utf-8")
        print(f" * write {CONFIG_NAME}")
    copied.append(config_path)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy the Krea 2 VAE files.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder. The files go to <root>/krea_2/model/vae/. The folder is created if missing.",
    )
    parser.add_argument("--input", required=True, help="Source vae folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copy_vae_files(args.root, args.input)
    print("Done")


if __name__ == "__main__":
    main()
