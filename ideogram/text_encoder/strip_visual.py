#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from safetensors.torch import load_file, save_file


def strip_visual_keys(source_path: Path) -> dict[str, object]:
    state_dict = load_file(str(source_path))
    keep = {k: v for k, v in state_dict.items() if not k.startswith("visual.")}
    stripped = len(state_dict) - len(keep)
    print(f"Removed {stripped} visual keys ({len(state_dict)} -> {len(keep)})")
    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip visual.* keys from a quantized text encoder checkpoint.")
    parser.add_argument("path", type=Path, help="Path to the quantized safetensors file.")
    parser.add_argument("--output", type=Path, default=None, help="Output path (default: overwrites input).")
    args = parser.parse_args()

    source = args.path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Not found: {source}")

    output = (args.output or source).resolve()
    state_dict = strip_visual_keys(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(output))
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
