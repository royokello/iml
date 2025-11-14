#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from safetensors.numpy import load_file, save_file

INT8_MIN = -128
INT8_MAX = 127

MODULE_PREFIXES = {
    "text_encoder": (
        "cond_stage_model.",
        "conditioner.",
        "conditioner.embedders.",
        "text_encoder.",
        "text_encoder_2.",
        "clip_l.",
        "clip_g.",
    ),
    "vae": (
        "first_stage_model.",
        "vae.",
        "decoder.",
        "encoder.",
        "quant_conv.",
        "post_quant_conv.",
        "model_ema.first_stage_model.",
        "model_ema.vae.",
    ),
    "unet": (
        "model.diffusion_model.",
        "model_ema.diffusion_model.",
        "diffusion_model.",
        "unet.",
        "control_model.",
    ),
}


@dataclass
class QuantStats:
    weights: int = 0
    biases: int = 0
    bytes_in: int = 0
    bytes_out: int = 0


def _derive_output_path(input_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(input_path.rstrip(os.sep))
    stem, ext = os.path.splitext(base or "model")
    ext = ext or ".safetensors"
    return os.path.join(out_dir, f"{stem}_int8{ext}")


def _tensor_module(name: str) -> str:
    for module, prefixes in MODULE_PREFIXES.items():
        if name.startswith(prefixes):
            return module
    return "unet"


def _quantize_int8_sym(tensor: np.ndarray) -> Tuple[np.ndarray, float, float]:
    arr32 = np.asarray(tensor, dtype=np.float32)
    amax = float(np.max(np.abs(arr32))) if arr32.size else 0.0
    if amax == 0.0:
        scale = 1.0
    else:
        scale = amax / INT8_MAX
    if arr32.size:
        q = np.clip(np.rint(arr32 / scale), INT8_MIN, INT8_MAX).astype(np.int8)
    else:
        q = arr32.astype(np.int8)
    return q, scale, amax


def _strip_suffix(name: str, suffix: str) -> str:
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def _weight_kind(name: str, tensor: np.ndarray) -> str | None:
    if _tensor_module(name) != "unet":
        return None
    if not name.endswith(".weight"):
        return None
    if tensor.ndim == 4:
        return "conv2d"
    if tensor.ndim == 2:
        return "linear"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize SDXL UNet Conv/Linear weights.")
    parser.add_argument("--input", required=True, help="Path to a single .safetensors checkpoint.")
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where the quantized checkpoint is written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.input):
        raise SystemExit(f"Missing checkpoint: {args.input}")
    if not args.input.endswith(".safetensors"):
        raise SystemExit("Only .safetensors inputs are supported.")

    tensors = load_file(args.input)
    weight_kinds: Dict[str, str] = {}
    for name, tensor in tensors.items():
        kind = _weight_kind(name, tensor)
        if kind:
            weight_kinds[name] = kind

    base_kind = { _strip_suffix(name, ".weight"): kind for name, kind in weight_kinds.items() }
    stats: Dict[str, QuantStats] = {kind: QuantStats() for kind in ("conv2d", "linear")}
    updated: Dict[str, np.ndarray] = {}

    for name, tensor in tensors.items():
        if name in weight_kinds:
            kind = weight_kinds[name]
            base = _strip_suffix(name, ".weight")
            q_weight, scale, amax = _quantize_int8_sym(tensor)
            updated[f"{base}.weight_q"] = q_weight
            updated[f"{base}.scale_w"] = np.asarray(scale, dtype=np.float16)

            stats[kind].weights += 1
            stats[kind].bytes_in += tensor.nbytes
            stats[kind].bytes_out += q_weight.nbytes + np.dtype(np.float16).itemsize

            print(
                f"[quant] {name} kind={kind} "
                f"amax={amax:.4g} scale={scale:.4g} "
                f"{tensor.dtype}->{q_weight.dtype}"
            )
            continue

        if name.endswith(".bias"):
            base = _strip_suffix(name, ".bias")
            kind = base_kind.get(base)
            if kind and tensor.dtype.kind == "f":
                updated[name] = tensor.astype(np.float16, copy=False)
                stats[kind].biases += 1
                continue

        updated[name] = tensor

    out_path = _derive_output_path(args.input, args.output)
    save_file(updated, out_path)

    for kind, kind_stats in stats.items():
        if not kind_stats.weights:
            continue
        print(
            f"[{kind}] weights={kind_stats.weights} "
            f"biases={kind_stats.biases} "
            f"bytes {kind_stats.bytes_in}->{kind_stats.bytes_out}"
        )

    print(f"[path] {out_path}")


if __name__ == "__main__":
    main()
