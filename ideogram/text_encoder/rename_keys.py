#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from safetensors.torch import load_file, save_file


def rename_keys(
    source_path: Path,
    old_prefix: str = "language_model.",
    new_prefix: str = "",
) -> dict[str, object]:
    state_dict = load_file(str(source_path))
    renamed = {}
    for key, tensor in state_dict.items():
        new_key = key.removeprefix(old_prefix) if key.startswith(old_prefix) else key
        renamed[new_key] = tensor
    changed = sum(1 for k in state_dict if k.startswith(old_prefix))
    print(f"Renamed {changed}/{len(state_dict)} keys ({old_prefix!r} -> {new_prefix!r})")
    return renamed


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename keys in a safetensors file.")
    parser.add_argument("path", type=Path, help="Path to the safetensors file.")
    parser.add_argument("--output", type=Path, default=None, help="Output path (default: overwrites input).")
    parser.add_argument("--old-prefix", default="language_model.", help="Prefix to strip.")
    parser.add_argument("--new-prefix", default="", help="Replacement prefix.")
    args = parser.parse_args()

    source = args.path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Not found: {source}")

    output = (args.output or source).resolve()
    state_dict = rename_keys(source, args.old_prefix, args.new_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(output))
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
