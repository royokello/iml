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
_MODEL_SHARDS = {
    "4b": 2,
    "9b": 4,
}
_NUM_HIDDEN_LAYERS = 36
_QWEN_LINEAR_WEIGHT_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def _resolve_model_dir(root: str | Path, version: str) -> Path:
    if version not in _MODEL_DIRS:
        raise ValueError(f"Unsupported version: {version!r}")
    return Path(root).expanduser().resolve() / _MODEL_DIRS[version] / "model" / "text_encoder"


def _build_checkpoint_files(model_dir: Path, version: str) -> list[Path]:
    shard_count = _MODEL_SHARDS[version]
    return [
        model_dir / f"model-{index:05d}-of-{shard_count:05d}.safetensors"
        for index in range(1, shard_count + 1)
    ]


def _build_target_tensors() -> list[str]:
    tensors: list[str] = []
    for layer_idx in range(_NUM_HIDDEN_LAYERS):
        layer_prefix = f"model.layers.{layer_idx}."
        for suffix in _QWEN_LINEAR_WEIGHT_SUFFIXES:
            tensors.append(layer_prefix + suffix)
    return tensors


def _quantize_text_encoder(
    root: str | Path,
    input_root: str | Path | None = None,
    *,
    version: str,
    method: str,
) -> Path:
    method = normalize_quant_method(method)
    output_dir = _resolve_model_dir(root, version)
    if input_root is not None:
        model_dir = Path(input_root).expanduser().resolve()
    else:
        model_dir = output_dir

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Text encoder directory not found: {model_dir}")

    checkpoint_files = _build_checkpoint_files(model_dir, version)
    target_tensors = _build_target_tensors()
    output_path = output_dir / f"{method}_quant.safetensors"

    print(f"Applying {method} quantization for Flux2 {version} text encoder ...")
    quantize_start = time.perf_counter()
    state_dict = quantize_model_tensors(
        files=checkpoint_files,
        targets={method: target_tensors},
    )
    state_dict = {
        name: tensor
        for name, tensor in state_dict.items()
        if not name.startswith("lm_head.")
    }
    quantize_seconds = time.perf_counter() - quantize_start
    print(f"Quantized in {quantize_seconds:.3f}s")

    print(f"Saving {output_path} ...")
    save_start = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        state_dict,
        str(output_path),
        metadata={
            "component": "text_encoder",
            "version": version,
            "method": method,
        },
    )
    save_seconds = time.perf_counter() - save_start
    print(f"Saved in {save_seconds:.3f}s")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize and save the Flux2 text encoder.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder where flux_2_klein_4b or flux2_9b is stored or should receive the quantized output.",
    )
    parser.add_argument(
        "--input",
        help="Full source text_encoder model directory. Bypasses the --root/--version input default.",
    )
    parser.add_argument(
        "--version",
        choices=("4b", "9b"),
        required=True,
        help="Flux2 text encoder version to quantize.",
    )
    parser.add_argument(
        "--method",
        choices=CLI_QUANT_METHODS,
        required=True,
        help="Quantization method to apply.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = _quantize_text_encoder(
        args.root,
        args.input,
        version=args.version,
        method=args.method,
    )
    print(f"Saved quantized text encoder to {output_path}")


if __name__ == "__main__":
    main()
