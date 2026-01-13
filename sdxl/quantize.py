#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from safetensors.numpy import load_file, save_file

INT8_MIN = -128
INT8_MAX = 127
SCALE_MIN = 1e-8
SCALE_SUFFIX = "_scale"
BLOCK_SIZE = 32
QUANTIZE_KEYS = (
    "to_v",
    "proj_in",
    "proj_out",
    "conv_shortcut",
    "to_out.0",
)

def _bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_input_path(input_arg: str) -> Tuple[Path, Path | None]:
    path = Path(input_arg)
    if path.is_file():
        if path.suffix != ".safetensors":
            raise SystemExit("Input file must be a .safetensors checkpoint.")
        return path, None

    if not path.is_dir():
        raise SystemExit(f"Missing input path: {input_arg}")

    direct = path / "diffusion_pytorch_model.safetensors"
    nested = path / "unet" / "diffusion_pytorch_model.safetensors"
    if direct.is_file():
        return direct, path
    if nested.is_file():
        return nested, path / "unet"

    raise SystemExit(
        "Could not find diffusion_pytorch_model.safetensors in the input directory."
    )


def _copy_tree(src_dir: Path, dst_dir: Path, skip_name: str) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.name == skip_name:
            continue
        target = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _quantize_block32(
    tensor: np.ndarray,
    scale_dtype: np.dtype = np.float16,
) -> Tuple[np.ndarray, np.ndarray]:
    arr32 = np.asarray(tensor, dtype=np.float32)
    if not arr32.size:
        return arr32.astype(np.int8), np.ones((), dtype=scale_dtype)

    original_shape = arr32.shape
    flattened = arr32.flatten()

    pad_len = (BLOCK_SIZE - (flattened.size % BLOCK_SIZE)) % BLOCK_SIZE
    if pad_len > 0:
        flattened = np.pad(flattened, (0, pad_len), mode="constant")

    reshaped = flattened.reshape(-1, BLOCK_SIZE)
    max_abs = np.max(np.abs(reshaped), axis=1, keepdims=True)
    scales = np.maximum(max_abs / 127.0, SCALE_MIN)

    quantized_flat = np.clip(
        np.rint(reshaped / scales),
        INT8_MIN,
        INT8_MAX,
    ).astype(np.int8)

    if pad_len > 0:
        quantized_flat = quantized_flat.flatten()[:-pad_len]
    else:
        quantized_flat = quantized_flat.flatten()

    quantized = quantized_flat.reshape(original_shape)
    return quantized, scales.flatten().astype(scale_dtype)


def _should_quantize(name: str, tensor: np.ndarray) -> bool:
    """Return True for float weight tensors we want to quantize by name."""
    if not np.issubdtype(tensor.dtype, np.floating):
        return False
    return name.endswith(".weight") and any(key in name for key in QUANTIZE_KEYS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize SDXL UNet linear/conv weights to int8 block32."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to UNet directory or .safetensors file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the quantized UNet checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_file, input_dir = _resolve_input_path(args.input)
    input_file = input_file.resolve()
    output_dir = Path(args.output).resolve()

    if output_dir == input_file.parent:
        raise SystemExit(
            "Output directory must be different from the input directory (no inplace)."
        )

    if input_dir is not None:
        input_dir = input_dir.resolve()
        if output_dir == input_dir or _is_relative_to(output_dir, input_dir):
            raise SystemExit("Output directory must be outside the input directory.")
        _copy_tree(input_dir, output_dir, skip_name=input_file.name)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    tensors = load_file(str(input_file))
    original_bytes = sum(tensor.nbytes for tensor in tensors.values())

    output_tensors: Dict[str, np.ndarray] = {}
    quantized = 0
    copied = 0
    for name, tensor in tensors.items():
        if _should_quantize(name, tensor):
            quantized_tensor, scale = _quantize_block32(tensor)
            output_tensors[name] = quantized_tensor
            output_tensors[f"{name}{SCALE_SUFFIX}"] = scale
            print(f"[quantize] {name}")
            quantized += 1
        else:
            output_tensors[name] = tensor.astype(np.float16)
            copied += 1

    out_path = output_dir / input_file.name
    save_file(output_tensors, str(out_path))

    new_bytes = sum(tensor.nbytes for tensor in output_tensors.values())
    scale_count = sum(1 for name in output_tensors if name.endswith(SCALE_SUFFIX))

    print(f"[unet] saved {len(output_tensors)} tensors to {out_path}")
    print(
        "[summary] "
        f"quantized={quantized} copied={copied} scales={scale_count} "
        f"size={_bytes_to_mb(original_bytes):.2f}MB->{_bytes_to_mb(new_bytes):.2f}MB"
    )


if __name__ == "__main__":
    main()
