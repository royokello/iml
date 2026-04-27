#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

from safetensors.torch import save_file

from utils.quant.model import quantize_model_tensors
from utils.quant.validators import CLI_QUANT_METHODS, normalize_quant_method

_MODEL_DIRS = {
    "4b": "flux_2_klein_4b",
    "9b": "flux2_9b",
}
_DOUBLE_BLOCKS = {
    "4b": 5,
    "9b": 8,
}
_SINGLE_BLOCKS = {
    "4b": 20,
    "9b": 24,
}


def _resolve_model_dir(root: str | Path, version: str) -> Path:
    if version not in _MODEL_DIRS:
        raise ValueError(f"Unsupported version: {version!r}")
    return Path(root).expanduser().resolve() / _MODEL_DIRS[version] / "model" / "transformer"


def _build_checkpoint_files(checkpoint_dir: Path, version: str) -> list[Path]:
    if version == "4b":
        return [checkpoint_dir / "diffusion_pytorch_model.safetensors"]
    return [
        checkpoint_dir / "diffusion_pytorch_model-00001-of-00002.safetensors",
        checkpoint_dir / "diffusion_pytorch_model-00002-of-00002.safetensors",
    ]


def _build_target_tensors(version: str) -> list[str]:
    tensors: list[str] = []

    for block_idx in range(_DOUBLE_BLOCKS[version]):
        prefix = f"transformer_blocks.{block_idx}."
        tensors.extend(
            (
                f"{prefix}attn.to_q.weight",
                f"{prefix}attn.to_k.weight",
                f"{prefix}attn.to_v.weight",
                f"{prefix}attn.to_out.0.weight",
                f"{prefix}attn.add_q_proj.weight",
                f"{prefix}attn.add_k_proj.weight",
                f"{prefix}attn.add_v_proj.weight",
                f"{prefix}attn.to_add_out.weight",
                f"{prefix}ff.linear_in.weight",
                f"{prefix}ff.linear_out.weight",
                f"{prefix}ff_context.linear_in.weight",
                f"{prefix}ff_context.linear_out.weight",
            )
        )

    for block_idx in range(_SINGLE_BLOCKS[version]):
        prefix = f"single_transformer_blocks.{block_idx}."
        tensors.extend(
            (
                f"{prefix}attn.to_qkv_mlp_proj.weight",
                f"{prefix}attn.to_out.weight",
            )
        )

    return tensors


def _quantize_denoiser(
    root: str | Path,
    input_root: str | Path | None = None,
    *,
    version: str,
    method: str,
    variant: str,
) -> Path:
    method = normalize_quant_method(method)
    output_model_dir = _resolve_model_dir(root, version)
    checkpoint_dir = Path(input_root).expanduser().resolve() if input_root is not None else output_model_dir / variant
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Transformer checkpoint directory not found: {checkpoint_dir}")

    checkpoint_files = _build_checkpoint_files(checkpoint_dir, version)
    target_tensors = _build_target_tensors(version)
    output_path = output_model_dir / variant / f"{method}_quant.safetensors"

    print(f"Applying {method} quantization for Flux2 {version} denoiser ({variant}) ...")
    quantize_start = time.perf_counter()
    state_dict = quantize_model_tensors(
        files=checkpoint_files,
        targets={method: target_tensors},
    )
    quantize_seconds = time.perf_counter() - quantize_start
    print(f"Quantized in {quantize_seconds:.3f}s")

    print(f"Saving {output_path} ...")
    save_start = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        state_dict,
        str(output_path),
        metadata={
            "component": "denoiser",
            "version": version,
            "variant": variant,
            "method": method,
        },
    )
    save_seconds = time.perf_counter() - save_start
    print(f"Saved in {save_seconds:.3f}s")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize and save the Flux2 denoiser.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder where flux_2_klein_4b or flux2_9b is stored or should receive the quantized output.",
    )
    parser.add_argument(
        "--input",
        help="Full source transformer checkpoint directory. Bypasses source variant selection.",
    )
    parser.add_argument(
        "--version",
        choices=("4b", "9b"),
        required=True,
        help="Flux2 denoiser version to quantize.",
    )
    parser.add_argument(
        "--method",
        choices=CLI_QUANT_METHODS,
        required=True,
        help="Quantization method to apply.",
    )
    parser.add_argument(
        "--variant",
        choices=("base", "distill"),
        required=True,
        help="Transformer checkpoint variant to read from and save into.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = _quantize_denoiser(
        args.root,
        args.input,
        version=args.version,
        method=args.method,
        variant=args.variant,
    )
    print(f"Saved quantized denoiser to {output_path}")


if __name__ == "__main__":
    main()
