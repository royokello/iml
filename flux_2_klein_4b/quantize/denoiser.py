#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

from safetensors.torch import save_file

from flux_2_klein_4b.denoiser.loader import DEFAULT_TARGET_LINEAR_NAMES, load_qwen3_denoiser


def _resolve_model_dir(root: str | Path) -> Path:
    root_path = Path(root).expanduser().resolve()
    model_dir = root_path / "flux_2_klein_4b" / "base" / "transformer"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Denoiser directory not found: {model_dir}")
    return model_dir


def _resolve_output_dir(root: str | Path) -> Path:
    root_path = Path(root).expanduser().resolve()
    output_dir = root_path / "flux_2_klein_4b" / "quant" / "transformer"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_input_checkpoint(model_dir: Path, input_path: str | Path | None) -> Path:
    if input_path is None:
        checkpoint_path = model_dir / "diffusion_pytorch_model.safetensors"
    else:
        checkpoint_path = Path(input_path).expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Denoiser checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _parse_target_linear_names(value: str | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_TARGET_LINEAR_NAMES
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    return items or DEFAULT_TARGET_LINEAR_NAMES


def _target_suffix(target_linear_names: tuple[str, ...]) -> str:
    if tuple(target_linear_names) == tuple(DEFAULT_TARGET_LINEAR_NAMES):
        return ""
    digest = hashlib.sha1(",".join(target_linear_names).encode("utf-8")).hexdigest()[:8]
    return f"__targets_{digest}"


def _build_output_path(
    output_dir: Path,
    *,
    quantization_precision: str,
    scale_precision: str,
    block_size: int,
    target_linear_names: tuple[str, ...],
) -> Path:
    filename = (
        f"{quantization_precision}_{scale_precision}_b{block_size}"
        f"{_target_suffix(target_linear_names)}.safetensor"
    )
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


def quantize_denoiser(
    root: str | Path,
    *,
    input_path: str | Path | None = None,
    quantization_precision: str = "int8",
    scale_precision: str = "fp16",
    block_size: int = 128,
    target_linear_names: tuple[str, ...] = DEFAULT_TARGET_LINEAR_NAMES,
) -> Path:
    model_dir = _resolve_model_dir(root)
    input_checkpoint_path = _resolve_input_checkpoint(model_dir, input_path)
    output_dir = _resolve_output_dir(root)
    output_path = _build_output_path(
        output_dir,
        quantization_precision=quantization_precision,
        scale_precision=scale_precision,
        block_size=block_size,
        target_linear_names=target_linear_names,
    )

    print("Loading denoiser ...")
    load_start = time.perf_counter()
    model = load_qwen3_denoiser(
        model_dir,
        checkpoint_path=input_checkpoint_path,
        quantization_precision=quantization_precision,
        scale_precision=scale_precision,
        block_size=block_size,
        target_linear_names=target_linear_names,
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
            "source": str(input_checkpoint_path),
            "model_dir": str(model_dir),
            "component": "transformer",
            "quantization_precision": quantization_precision,
            "scale_precision": scale_precision,
            "block_size": str(block_size),
            "target_linear_names": ",".join(target_linear_names),
        },
    )
    save_seconds = time.perf_counter() - save_start
    print(f"Saved in {save_seconds:.3f}s")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize and save the Flux 2 Klein 4B denoiser.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder that contains flux_2_klein_4b/base/transformer and receives quantized output.",
    )
    parser.add_argument(
        "--input",
        help=(
            "Optional path to the source denoiser checkpoint file. "
            "Defaults to <root>/flux_2_klein_4b/base/transformer/diffusion_pytorch_model.safetensors."
        ),
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
    parser.add_argument(
        "--target-linear-names",
        default=",".join(DEFAULT_TARGET_LINEAR_NAMES),
        help="Comma-separated linear module names to quantize.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = quantize_denoiser(
        args.root,
        input_path=args.input,
        quantization_precision=args.quantization_precision,
        scale_precision=args.scale_precision,
        block_size=args.block_size,
        target_linear_names=_parse_target_linear_names(args.target_linear_names),
    )
    print(f"Saved quantized denoiser to {output_path}")


if __name__ == "__main__":
    main()
