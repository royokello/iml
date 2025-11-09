#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
from typing import List, Tuple

from safetensors.torch import load_file as load_safetensors


def _derive_output_path(model_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(model_path.rstrip(os.sep)) or "model"
    return os.path.join(out_dir, f"{base}.csv")


def _dtype_name(t) -> str:
    return str(t.dtype).split(".")[-1]


def collect_tensor_info(input_path: str) -> List[Tuple[str, str, str]]:
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)
    if not input_path.endswith(".safetensors"):
        raise ValueError("Only .safetensors files are supported.")

    state_dict = load_safetensors(input_path, device="cpu")
    rows: List[Tuple[str, str, str]] = []
    for name, tensor in state_dict.items():
        shape = "[" + ",".join(str(dim) for dim in tensor.shape) + "]"
        rows.append((name, shape, _dtype_name(tensor)))
    return sorted(rows, key=lambda x: x[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export tensor name/shape/precision data for a single SDXL .safetensors file."
    )
    parser.add_argument("--input", required=True, help="Path to a single SDXL .safetensors checkpoint")
    parser.add_argument("--output", required=True, help="Directory for the generated CSV summary")
    args = parser.parse_args()

    entries = collect_tensor_info(args.input)
    out_path = _derive_output_path(args.input, args.output)

    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "shape", "precision"])
        writer.writerows(entries)

    print(f"Wrote {len(entries)} tensors to {out_path}")


if __name__ == "__main__":
    main()
