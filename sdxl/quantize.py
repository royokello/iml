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


def _parse_precision(value: str) -> str:
    value_lower = value.lower()
    if value_lower not in {"fp16", "fp32", "int8"}:
        raise argparse.ArgumentTypeError("Precision must be one of: fp16, fp32, int8.")
    return value_lower


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
    base: str,
    to_q: str,
    to_k: str,
    to_v: str,
    to_out: str,
    proj_in: str,
    proj_out: str,
    conv_shortcut: str,
) -> str:
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


def _quantize_int8(tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr32 = np.asarray(tensor, dtype=np.float32)
    if not arr32.size:
        scale_shape = arr32.shape[:1] if arr32.ndim else ()
        return arr32.astype(np.int8), np.ones(scale_shape, dtype=np.float16)

    if arr32.ndim == 0:
        max_abs = float(np.abs(arr32))
        scale = np.array(max(max_abs / 127.0, SCALE_MIN), dtype=np.float32)
        quantized = np.clip(np.rint(arr32 / scale), INT8_MIN, INT8_MAX).astype(np.int8)
        return quantized, np.asarray(scale, dtype=np.float16)

    if arr32.ndim == 1:
        max_abs = np.abs(arr32)
        scales = np.maximum(max_abs / 127.0, SCALE_MIN)
        quantized = np.clip(np.rint(arr32 / scales), INT8_MIN, INT8_MAX).astype(np.int8)
        return quantized, scales.astype(np.float16)

    reduce_axes = tuple(range(1, arr32.ndim))
    max_abs = np.max(np.abs(arr32), axis=reduce_axes, keepdims=False)
    scales = np.maximum(max_abs / 127.0, SCALE_MIN)
    reshape = (scales.shape[0],) + (1,) * (arr32.ndim - 1)
    quantized = np.clip(
        np.rint(arr32 / scales.reshape(reshape)),
        INT8_MIN,
        INT8_MAX,
    ).astype(np.int8)
    return quantized, scales.astype(np.float16)


def _cast_tensor(tensor: np.ndarray, precision: str) -> Tuple[np.ndarray, np.ndarray | None]:
    if precision == "int8":
        return _quantize_int8(tensor)
    if precision == "fp16":
        return np.asarray(tensor, dtype=np.float16), None
    return np.asarray(tensor, dtype=np.float32), None


def _convert_unet_tensors(
    tensors: Dict[str, np.ndarray],
    base_precision: str,
    to_q_precision: str,
    to_k_precision: str,
    to_v_precision: str,
    to_out_precision: str,
    proj_in_precision: str,
    proj_out_precision: str,
    conv_shortcut_precision: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    converted: Dict[str, np.ndarray] = {}
    scales: Dict[str, np.ndarray] = {}
    for name, tensor in tensors.items():
        precision = _target_precision(
            name,
            base_precision,
            to_q_precision,
            to_k_precision,
            to_v_precision,
            to_out_precision,
            proj_in_precision,
            proj_out_precision,
            conv_shortcut_precision,
        )
        cast_tensor, scale = _cast_tensor(tensor, precision)
        converted[name] = cast_tensor
        if scale is not None:
            scales[f"{name}{SCALE_SUFFIX}"] = scale
    return converted, scales


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
        "--base-precision",
        default="fp16",
        type=_parse_precision,
        help="Precision for tensors that are not to_q, to_k, to_v, to_out, proj_in, or proj_out (default: fp16).",
    )
    parser.add_argument(
        "--to-q-precision",
        default="int8",
        type=_parse_precision,
        help="Precision override for to_q tensors (default: int8).",
    )
    parser.add_argument(
        "--to-k-precision",
        default="int8",
        type=_parse_precision,
        help="Precision override for to_k tensors (default: int8).",
    )
    parser.add_argument(
        "--to-v-precision",
        default="fp16",
        type=_parse_precision,
        help="Precision override for to_v tensors (default: int8).",
    )
    parser.add_argument(
        "--to-out-precision",
        default="fp16",
        type=_parse_precision,
        help="Precision override for to_out tensors (default: int8).",
    )
    parser.add_argument(
        "--proj-in-precision",
        default="fp16",
        type=_parse_precision,
        help="Precision override for proj_in tensors (default: int8).",
    )
    parser.add_argument(
        "--proj-out-precision",
        default="fp16",
        type=_parse_precision,
        help="Precision override for proj_out tensors (default: int8).",
    )
    parser.add_argument(
        "--conv-shortcut-precision",
        default="fp16",
        type=_parse_precision,
        help="Precision override for conv_shortcut tensors (default: int8).",
    )
    
    args = parser.parse_args()
    if not args.inplace and not args.output:
        parser.error("--output is required unless --inplace is provided.")
    return args


def _log_precisions(args: argparse.Namespace) -> None:
    print(f"[precision] base={args.base_precision}")
    print(f"[precision] to_q={args.to_q_precision}")
    print(f"[precision] to_k={args.to_k_precision}")
    print(f"[precision] to_v={args.to_v_precision}")
    print(f"[precision] to_out={args.to_out_precision}")
    print(f"[precision] proj_in={args.proj_in_precision}")
    print(f"[precision] proj_out={args.proj_out_precision}")
    print(f"[precision] conv_shortcut={args.conv_shortcut_precision}")


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

    converted, scale_tensors = _convert_unet_tensors(
        unet_tensors,
        args.base_precision,
        args.to_q_precision,
        args.to_k_precision,
        args.to_v_precision,
        args.to_out_precision,
        args.proj_in_precision,
        args.proj_out_precision,
        args.conv_shortcut_precision,
    )
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
