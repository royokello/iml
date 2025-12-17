#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from safetensors.numpy import load_file, save_file

INT8_MIN = -128
INT8_MAX = 127
SCALE_MIN = 1e-8
SCALE_SUFFIX = "_scale"

UNET_PREFIXES = (
    "model.diffusion_model.",
    "model_ema.diffusion_model.",
    "diffusion_model.",
    "unet.",
    "control_model.",
)


def _parse_dtype(name: str) -> np.dtype:
    if name == "fp32":
        return np.float32
    if name == "fp16":
        return np.float16
    if name == "int8":
        return np.int8
    raise ValueError(f"Unknown dtype: {name}")


def _normalize_dtype_str(s: str) -> str:
    if s == "f16": return "fp16"
    if s == "f32": return "fp32"
    if s == "i8": return "int8"
    return s

def _parse_config(value: str) -> Tuple[str, int | None, np.dtype, np.dtype | None]:
    # e.g. cast_f16, channel_i8_f16, block32_i8_f16
    parts = value.lower().split("_")
    head = parts[0]

    if head == "cast":
        # cast_f16 -> (cast, None, float16, None)
        if len(parts) != 2:
             raise argparse.ArgumentTypeError(f"Invalid cast format '{value}'. Expected cast_<dtype>.")
        return "cast", None, _parse_dtype(_normalize_dtype_str(parts[1])), None

    if head == "channel":
        # channel_i8_f16 -> (channel, None, int8, float16)
        if len(parts) != 3:
             raise argparse.ArgumentTypeError(f"Invalid channel format '{value}'. Expected channel_<w_dtype>_<s_dtype>.")
        return "channel", None, _parse_dtype(_normalize_dtype_str(parts[1])), _parse_dtype(_normalize_dtype_str(parts[2]))

    if head.startswith("block"):
        # block32_i8_f16 -> (block, 32, int8, float16)
        if len(parts) != 3:
             raise argparse.ArgumentTypeError(f"Invalid block format '{value}'. Expected block<N>_<w_dtype>_<s_dtype>.")
        try:
            block_size = int(head[5:])
        except ValueError:
             raise argparse.ArgumentTypeError(f"Invalid block size in '{head}'.")
        return "block", block_size, _parse_dtype(_normalize_dtype_str(parts[1])), _parse_dtype(_normalize_dtype_str(parts[2]))

    raise argparse.ArgumentTypeError(f"Unknown strategy in '{value}'. Must start with cast, channel, or block.")


def _derive_output_dir(input_path: str, out_dir: str) -> Path:
    os.makedirs(out_dir, exist_ok=True)
    if os.path.isdir(input_path):
        name = Path(input_path).name or "unet"
    else:
        name = Path(input_path).stem or "unet"
    target_dir = Path(out_dir) / f"{name}_unet"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _is_unet_tensor(name: str) -> bool:
    return name.startswith(UNET_PREFIXES)


def _target_precision(
    name: str,
    base: Tuple,
    to_q: Tuple,
    to_k: Tuple,
    to_v: Tuple,
    to_out: Tuple,
    proj_in: Tuple,
    proj_out: Tuple,
    conv_shortcut: Tuple,
) -> Tuple:
    if "to_q" in name:
        return to_q
    if "to_k" in name:
        return to_k
    if "to_v" in name:
        return to_v
    if "to_out" in name:
        return to_out
    if "proj_in" in name:
        return proj_in
    if "proj_out" in name:
        return proj_out
    if "conv_shortcut" in name:
        return conv_shortcut
    return base


def _quantize_int8(tensor: np.ndarray, scale_dtype: np.dtype = np.float16) -> Tuple[np.ndarray, np.ndarray]:
    arr32 = np.asarray(tensor, dtype=np.float32)
    if not arr32.size:
        scale_shape = arr32.shape[:1] if arr32.ndim else ()
        return arr32.astype(np.int8), np.ones(scale_shape, dtype=scale_dtype)

    if arr32.ndim == 0:
        max_abs = float(np.abs(arr32))
        scale = np.array(max(max_abs / 127.0, SCALE_MIN), dtype=np.float32)
        quantized = np.clip(np.rint(arr32 / scale), INT8_MIN, INT8_MAX).astype(np.int8)
        return quantized, np.asarray(scale, dtype=scale_dtype)

    if arr32.ndim == 1:
        max_abs = np.abs(arr32)
        scales = np.maximum(max_abs / 127.0, SCALE_MIN)
        quantized = np.clip(np.rint(arr32 / scales), INT8_MIN, INT8_MAX).astype(np.int8)
        return quantized, scales.astype(scale_dtype)

    reduce_axes = tuple(range(1, arr32.ndim))
    max_abs = np.max(np.abs(arr32), axis=reduce_axes, keepdims=False)
    scales = np.maximum(max_abs / 127.0, SCALE_MIN)
    reshape = (scales.shape[0],) + (1,) * (arr32.ndim - 1)
    quantized = np.clip(
        np.rint(arr32 / scales.reshape(reshape)),
        INT8_MIN,
        INT8_MAX,
    ).astype(np.int8)
    return quantized, scales.astype(scale_dtype)


def _quantize_block(
    tensor: np.ndarray,
    block_size: int,
    scale_dtype: np.dtype = np.float16
) -> Tuple[np.ndarray, np.ndarray]:
    arr32 = np.asarray(tensor, dtype=np.float32)
    if not arr32.size:
        return arr32.astype(np.int8), np.ones((), dtype=scale_dtype)

    original_shape = arr32.shape
    flattened = arr32.flatten()

    # Pad if necessary
    pad_len = (block_size - (flattened.size % block_size)) % block_size
    if pad_len > 0:
        flattened = np.pad(flattened, (0, pad_len), mode="constant")

    # Reshape to (num_blocks, block_size)
    reshaped = flattened.reshape(-1, block_size)

    max_abs = np.max(np.abs(reshaped), axis=1, keepdims=True)
    scales = np.maximum(max_abs / 127.0, SCALE_MIN)

    # Quantize
    quantized_flat = np.clip(
        np.rint(reshaped / scales), INT8_MIN, INT8_MAX
    ).astype(np.int8)

    # Remove padding and restore shape
    if pad_len > 0:
        quantized_flat = quantized_flat.flatten()[:-pad_len]
    else:
        quantized_flat = quantized_flat.flatten()

    quantized = quantized_flat.reshape(original_shape)
    return quantized, scales.flatten().astype(scale_dtype)


def _cast_tensor(
    tensor: np.ndarray,
    config: Tuple[str, int | None, np.dtype, np.dtype | None]
) -> Tuple[np.ndarray, np.ndarray | None]:
    strategy, block_size, w_dtype, s_dtype = config

    if strategy == "channel":
        # We need to pass dtypes to quantizer if we want to be strict,
        # but for now we assume int8/fp16 as before or update _quantize_int8
        # The user requested "channel and block must specify the scale precision as well"
        return _quantize_int8(tensor, scale_dtype=s_dtype or np.float16)
    
    if strategy == "block":
         assert block_size is not None
         return _quantize_block(tensor, block_size, scale_dtype=s_dtype or np.float16)

    if strategy == "cast":
         return np.asarray(tensor, dtype=w_dtype), None
    
    return np.asarray(tensor, dtype=np.float32), None






def _bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and quantize UNet tensors (int8 weights include per-channel fp16 scales)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a Diffusers UNet directory (root repo or containing diffusion_pytorch_model.safetensors).",
    )
    parser.add_argument(
        "--output",
        help="Directory where the UNet safetensors file will be written (default: required unless --inplace).",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the UNet safetensors referenced by --input instead of writing to --output.",
    )
    parser.add_argument(
        "--base",
        default="cast_f16",
        type=_parse_config,
        help="Precision for tensors that are not to_q, to_k, to_v, to_out, proj_in, or proj_out (default: cast_f16).",
    )
    parser.add_argument(
        "--to-q",
        default="channel_i8_f16",
        type=_parse_config,
        help="Precision override for to_q tensors (default: channel_i8_f16).",
    )
    parser.add_argument(
        "--to-k",
        default="channel_i8_f16",
        type=_parse_config,
        help="Precision override for to_k tensors (default: channel_i8_f16).",
    )
    parser.add_argument(
        "--to-v",
        default="block32_i8_f16",
        type=_parse_config,
        help="Precision override for to_v tensors (default: block32_i8_f16).",
    )
    parser.add_argument(
        "--to-out",
        default="cast_f16",
        type=_parse_config,
        help="Precision override for to_out tensors (default: cast_f16).",
    )
    parser.add_argument(
        "--proj-in",
        default="cast_f16",
        type=_parse_config,
        help="Precision override for proj_in tensors (default: cast_f16).",
    )
    parser.add_argument(
        "--proj-out",
        default="cast_f16",
        type=_parse_config,
        help="Precision override for proj_out tensors (default: cast_f16).",
    )
    parser.add_argument(
        "--conv-shortcut",
        default="cast_f16",
        type=_parse_config,
        help="Precision override for conv_shortcut tensors (default: cast_f16).",
    )
    
    args = parser.parse_args()
    if not args.inplace and not args.output:
        parser.error("--output is required unless --inplace is provided.")
    return args


def _log_precisions(args: argparse.Namespace) -> None:
    print(f"[precision] base={args.base}")
    print(f"[precision] to_q={args.to_q}")
    print(f"[precision] to_k={args.to_k}")
    print(f"[precision] to_v={args.to_v}")
    print(f"[precision] to_out={args.to_out}")
    print(f"[precision] proj_in={args.proj_in}")
    print(f"[precision] proj_out={args.proj_out}")
    print(f"[precision] conv_shortcut={args.conv_shortcut}")


def _resolve_input_path(input_arg: str) -> str:
    path = Path(input_arg)
    if not path.exists():
        raise SystemExit(f"Missing UNet directory: {input_arg}")
    if not path.is_dir():
        raise SystemExit("--input must be a directory that contains a Diffusers UNet.")

    direct = path / "diffusion_pytorch_model.safetensors"
    nested = path / "unet" / "diffusion_pytorch_model.safetensors"
    for candidate in (direct, nested):
        if candidate.is_file():
            return str(candidate)
    raise SystemExit(
        "Could not find UNet safetensors in directory: "
        f"{direct if direct.exists() else nested}"
    )


def main() -> None:
    args = parse_args()
    _log_precisions(args)
    resolved_input = _resolve_input_path(args.input)

    tensors = load_file(resolved_input)
    unet_tensors = {name: tensor for name, tensor in tensors.items() if _is_unet_tensor(name)}
    if not unet_tensors:
        unet_tensors = tensors
    original_bytes = sum(tensor.nbytes for tensor in unet_tensors.values())

    converted: Dict[str, np.ndarray] = {}
    scale_tensors: Dict[str, np.ndarray] = {}
    for name, tensor in unet_tensors.items():
        config = _target_precision(
            name,
            args.base,
            args.to_q,
            args.to_k,
            args.to_v,
            args.to_out,
            args.proj_in,
            args.proj_out,
            args.conv_shortcut,
        )
        cast_tensor, scale = _cast_tensor(tensor, config)
        converted[name] = cast_tensor
        if scale is not None:
            scale_tensors[f"{name}{SCALE_SUFFIX}"] = scale
    all_tensors: Dict[str, np.ndarray] = {**converted, **scale_tensors}
    if args.inplace:
        out_path = Path(resolved_input)
    else:
        out_dir = _derive_output_dir(args.input, args.output)
        out_path = out_dir / "diffusion_pytorch_model.safetensors"
    save_file(all_tensors, str(out_path))

    new_bytes = sum(tensor.nbytes for tensor in all_tensors.values())
    fp16_count = sum(1 for tensor in converted.values() if tensor.dtype == np.float16)
    int8_count = sum(1 for tensor in converted.values() if tensor.dtype == np.int8)
    scale_count = len(scale_tensors)

    print(f"[unet] saved {len(all_tensors)} tensors (including scales) to {out_path}")
    print(
        "[summary] weights="
        f"{len(converted)} scales={scale_count} fp16={fp16_count} int8={int8_count} "
        f"size={_bytes_to_mb(original_bytes):.2f}MB->{_bytes_to_mb(new_bytes):.2f}MB"
    )


if __name__ == "__main__":
    main()
