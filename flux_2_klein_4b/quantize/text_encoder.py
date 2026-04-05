#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

from safetensors.torch import save_file

from flux_2_klein_4b.text_encoder.loader import load_qwen3_text_encoder


def _resolve_paths(root: str | Path) -> tuple[Path, Path]:
    root_path = Path(root).expanduser().resolve()
    model_dir = root_path / "flux_2_klein_4b" / "base" / "text_encoder"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Text encoder directory not found: {model_dir}")

    output_dir = root_path / "flux_2_klein_4b" / "quant" / "text_encoder"
    output_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, output_dir


def _build_output_path(
    output_dir: Path,
    *,
    quantization_precision: str,
    scale_precision: str,
    block_size: int,
) -> Path:
    filename = f"{quantization_precision}_{scale_precision}_b{block_size}.safetensor"
    return output_dir / filename


def _prepare_state_dict(model) -> dict[str, object]:
    prepared: dict[str, object] = {}
    seen_storage_keys: set[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = set()

    for name, tensor in model.state_dict().items():
        tensor = tensor.detach().cpu()
        storage_key = (
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )
        if storage_key in seen_storage_keys:
            tensor = tensor.clone()
        elif not tensor.is_contiguous():
            tensor = tensor.contiguous()
            storage_key = (
                tensor.untyped_storage().data_ptr(),
                tensor.storage_offset(),
                tuple(tensor.shape),
                tuple(tensor.stride()),
            )

        seen_storage_keys.add(storage_key)
        prepared[name] = tensor

    return prepared


def quantize_text_encoder(
    root: str | Path,
    *,
    quantization_precision: str = "int8",
    scale_precision: str = "fp16",
    block_size: int = 128,
) -> Path:
    model_dir, output_dir = _resolve_paths(root)
    output_path = _build_output_path(
        output_dir,
        quantization_precision=quantization_precision,
        scale_precision=scale_precision,
        block_size=block_size,
    )

    print("Loading text encoder ...")
    load_start = time.perf_counter()
    model = load_qwen3_text_encoder(
        model_dir,
        quantization_precision=quantization_precision,
        scale_precision=scale_precision,
        block_size=block_size,
    )
    load_seconds = time.perf_counter() - load_start
    print(f"Loaded and quantized in {load_seconds:.3f}s")

    print("Preparing state dict ...")
    prepare_start = time.perf_counter()
    state_dict = _prepare_state_dict(model)
    prepare_seconds = time.perf_counter() - prepare_start
    print(f"Prepared state dict in {prepare_seconds:.3f}s")

    print(f"Saving {output_path} ...")
    save_start = time.perf_counter()
    save_file(
        state_dict,
        str(output_path),
        metadata={
            "source": str(model_dir),
            "quantization_precision": quantization_precision,
            "scale_precision": scale_precision,
            "block_size": str(block_size),
        },
    )
    save_seconds = time.perf_counter() - save_start
    print(f"Saved in {save_seconds:.3f}s")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize and save the Flux 2 Klein 4B text encoder.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder that contains flux_2_klein_4b/base/text_encoder.",
    )
    parser.add_argument(
        "--quantization-precision",
        choices=("fp16", "int8", "int4"),
        default="int8",
        help="Weight quantization mode.",
    )
    parser.add_argument(
        "--scale-precision",
        choices=("fp32", "fp16", "e8m0"),
        default="fp16",
        help="Per-block scale storage mode.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        choices=(32, 64, 128),
        default=128,
        help="Block size used for quantized linear layers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = quantize_text_encoder(
        args.root,
        quantization_precision=args.quantization_precision,
        scale_precision=args.scale_precision,
        block_size=args.block_size,
    )
    print(f"Saved quantized text encoder to {output_path}")


if __name__ == "__main__":
    main()
