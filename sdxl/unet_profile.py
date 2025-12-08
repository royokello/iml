#!/usr/bin/env python
"""
Utility script to inspect the INT8 SDXL UNet weight layout.

It instantiates `SDXLUNet`, dumps every tensor name with shape/dtype to stdout,
and writes the same information to `sdxlunet_tensors.txt` inside a user supplied
output directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from sdxl.unet import SDXLUNet


def _format_tensor_entry(name: str, tensor) -> str:
    dims = ",".join(str(dim) for dim in tensor.shape)
    shape = f"[{dims}]" if dims else "[]"
    return f'{name},"{shape}"'


def _collect_state_lines() -> list[str]:
    unet = SDXLUNet()
    state = unet.state_dict()
    lines = ["name,shape"]
    for name, tensor in state.items():
        lines.append(_format_tensor_entry(name, tensor))
    return lines


def _write_output(lines: Iterable[str], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "sdxlunet_tensors.csv"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump the SDXL UNet state_dict tensor names to stdout and a text file."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where sdxlunet_tensors.csv will be created.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output).expanduser().resolve()
    lines = _collect_state_lines()
    print("\n".join(lines))
    out_path = _write_output(lines, output_dir)
    print(f"\n[profile] Saved tensor listing to {out_path}")


if __name__ == "__main__":
    main()
