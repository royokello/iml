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

PRECISION_CHOICES = ("int8", "fp16", "fp32")

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
class Stats:
    tensors: int = 0
    converted: int = 0
    bytes_in: int = 0
    bytes_out: int = 0


def _derive_output_path(input_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(input_path.rstrip(os.sep))
    stem, ext = os.path.splitext(base or "model")
    ext = ext or ".safetensors"
    return os.path.join(out_dir, f"{stem}_quantized{ext}")


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


def _apply_precision(tensor: np.ndarray, precision: str) -> Tuple[np.ndarray, str, float, float | None]:
    if tensor.dtype.kind != "f":
        return tensor, "skip", 0.0, None

    if precision == "int8":
        q, scale, amax = _quantize_int8_sym(tensor)
        return q, "int8", amax, scale
    if precision == "fp16":
        return tensor.astype(np.float16, copy=False), "fp16", 0.0, None
    if precision == "fp32":
        return tensor.astype(np.float32, copy=False), "fp32", 0.0, None

    raise ValueError(f"Unsupported precision {precision}")


def identify_tensor_type(name: str) -> str:
    n = name.lower()
    if "bias" in n:
        return "bias"
    if any(k in n for k in ["attn1", "attn2", "to_q", "to_k", "to_v", "to_out"]):
        return "attn"
    if any(k in n for k in ["proj_in", "proj_out", "ff.net", "emb_layers", "proj.weight"]):
        return "linear"
    if any(k in n for k in ["in_layers", "out_layers", "skip_connection", "op.weight"]):
        return "conv2d"
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize SDXL checkpoints with per-module precision controls."
    )
    parser.add_argument("--input", required=True, help="Path to a single .safetensors checkpoint.")
    parser.add_argument("--output", required=True, help="Directory where the quantized file is stored.")
    parser.add_argument(
        "--text-encoder-precision",
        default="fp16",
        choices=PRECISION_CHOICES,
        help="Precision to use for text encoder tensors.",
    )
    parser.add_argument(
        "--vae-precision",
        default="fp16",
        choices=PRECISION_CHOICES,
        help="Precision to use for VAE tensors.",
    )
    parser.add_argument(
        "--unet-precision",
        default="fp16",
        choices=PRECISION_CHOICES,
        help="Precision to use for UNet tensors not quantized to int8.",
    )
    parser.add_argument(
        "--unet-targets",
        default="conv2d",
        help="Comma-separated UNet tensor categories to quantize to int8 (attn, linear, conv2d).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.input):
        raise SystemExit(f"Missing checkpoint: {args.input}")
    if not args.input.endswith(".safetensors"):
        raise SystemExit("Only .safetensors inputs are supported.")

    tensors = load_file(args.input)
    precision_map = {
        "text_encoder": args.text_encoder_precision,
        "vae": args.vae_precision,
    }

    stats: Dict[str, Stats] = {k: Stats() for k in ("text_encoder", "vae", "unet")}
    updated: Dict[str, np.ndarray] = {}

    requested_targets = {t.strip().lower() for t in args.unet_targets.split(",") if t.strip()}
    valid_targets = {"attn", "linear", "conv2d"}
    invalid_targets = requested_targets - valid_targets
    if invalid_targets:
        raise SystemExit(f"Unsupported --unet-targets: {', '.join(sorted(invalid_targets))}")

    for name, tensor in tensors.items():
        module = _tensor_module(name)
        if module not in stats:
            stats[module] = Stats()
        stats[module].tensors += 1

        if module == "unet":
            tensor_type = identify_tensor_type(name)
            desired_precision = "int8" if tensor_type in requested_targets else args.unet_precision
        else:
            desired_precision = precision_map[module]
        new_tensor, action, amax, scale = _apply_precision(tensor, desired_precision)
        updated[name] = new_tensor
        if action == "int8" and scale is not None:
            updated[f"{name}.scale"] = np.asarray(scale, dtype=np.float16)

        changed = action != "skip" and (
            new_tensor.dtype != tensor.dtype or action == "int8"
        )
        if changed:
            stats[module].converted += 1
            stats[module].bytes_in += tensor.nbytes
            out_bytes = new_tensor.nbytes
            if action == "int8":
                out_bytes += np.dtype(np.float16).itemsize
            stats[module].bytes_out += out_bytes

            if action == "int8":
                print(
                    f"[quant] {name} module={module} "
                    f"amax={amax:.4g} scale={scale:.4g} {tensor.dtype}->{new_tensor.dtype}"
                )

    out_path = _derive_output_path(args.input, args.output)
    save_file(updated, out_path)

    for module, module_stats in stats.items():
        if not module_stats.tensors:
            continue
        print(
            f"[module:{module}] tensors={module_stats.tensors} "
            f"converted={module_stats.converted} "
            f"bytes {module_stats.bytes_in}->{module_stats.bytes_out}"
        )

    print(f"[path] {out_path}")


if __name__ == "__main__":
    main()
