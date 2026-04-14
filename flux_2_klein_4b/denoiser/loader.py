from __future__ import annotations

import gc
import inspect
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file

from flux_2_klein_4b.quantize.linear import QuantizedLinear

from .transformer import Flux2Transformer2DModel

DEFAULT_TARGET_LINEAR_NAMES = (
    "to_q",
    "to_k",
    "to_v",
    "to_out",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "to_qkv_mlp_proj",
    "linear_in",
    "linear_out",
    "x_embedder",
    "context_embedder",
)


def _resolve_model_dir(path: str | Path) -> Path:
    model_path = Path(path)
    if model_path.is_file():
        return model_path.parent
    return model_path


def _parse_target_linear_names(value: str | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_TARGET_LINEAR_NAMES
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    return items or DEFAULT_TARGET_LINEAR_NAMES


def _load_local_single_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    shard_start = time.perf_counter()
    state_dict = safe_load_file(str(checkpoint_path))
    load_seconds = time.perf_counter() - shard_start
    apply_start = time.perf_counter()
    incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
    apply_seconds = time.perf_counter() - apply_start
    total_seconds = time.perf_counter() - shard_start
    del state_dict
    gc.collect()
    print(
        f"    [1/1] "
        f"load={load_seconds:.3f}s "
        f"apply={apply_seconds:.3f}s "
        f"total={total_seconds:.3f}s"
    )
    if incompatible.unexpected_keys:
        raise RuntimeError("Unexpected keys in denoiser checkpoint: " + ", ".join(sorted(incompatible.unexpected_keys)))


def _materialize_meta_tensors(model: torch.nn.Module) -> None:
    unresolved_parameters = [name for name, parameter in model.named_parameters() if getattr(parameter, "is_meta", False)]
    if unresolved_parameters:
        raise RuntimeError("Checkpoint load left parameters on meta: " + ", ".join(unresolved_parameters))

    unresolved_buffers = [name for name, buffer in model.named_buffers() if getattr(buffer, "is_meta", False)]
    if unresolved_buffers:
        raise RuntimeError("Checkpoint load left buffers on meta: " + ", ".join(unresolved_buffers))


def _build_model_from_config(model_dir: Path) -> Flux2Transformer2DModel:
    config_path = model_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    init_parameters = inspect.signature(Flux2Transformer2DModel.__init__).parameters
    init_kwargs = {
        key: value
        for key, value in config.items()
        if key in init_parameters and key != "self"
    }

    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        with torch.device("meta"):
            model = Flux2Transformer2DModel(**init_kwargs)
    finally:
        torch.set_default_dtype(default_dtype)
    return model


def _list_checkpoint_keys(checkpoint_path: Path) -> set[str]:
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        return set(handle.keys())


def _replace_targeted_linear_modules(
    module: torch.nn.Module,
    *,
    quantization_precision: str,
    scale_precision: str,
    block_size: int,
    target_linear_names: tuple[str, ...],
    quantize_weights: bool = True,
    path: tuple[str, ...] = (),
    checkpoint_keys: set[str] | None = None,
) -> None:
    for name, child in list(module.named_children()):
        child_path = (*path, name)
        if isinstance(child, torch.nn.Linear) and _is_targeted_linear(
            child_path,
            target_linear_names,
            checkpoint_keys=checkpoint_keys,
        ):
            setattr(
                module,
                name,
                _build_quantized_linear(
                    child,
                    quantization_precision=quantization_precision,
                    scale_precision=scale_precision,
                    block_size=block_size,
                    quantize_weights=quantize_weights,
                ),
            )
            continue
        _replace_targeted_linear_modules(
            child,
            quantization_precision=quantization_precision,
            scale_precision=scale_precision,
            block_size=block_size,
            target_linear_names=target_linear_names,
            quantize_weights=quantize_weights,
            path=child_path,
            checkpoint_keys=checkpoint_keys,
        )


def _is_targeted_linear(
    module_path: tuple[str, ...],
    target_linear_names: tuple[str, ...],
    *,
    checkpoint_keys: set[str] | None = None,
) -> bool:
    if not module_path:
        return False

    full_name = ".".join(module_path)
    is_target = (
        full_name in target_linear_names
        or module_path[-1] in target_linear_names
        or (module_path[-1].isdigit() and len(module_path) > 1 and module_path[-2] in target_linear_names)
    )
    if not is_target:
        return False

    if checkpoint_keys is None:
        return True

    return f"{full_name}.scales" in checkpoint_keys


def _build_quantized_linear(
    linear: torch.nn.Linear,
    *,
    quantization_precision: str,
    scale_precision: str,
    block_size: int,
    quantize_weights: bool,
) -> torch.nn.Module:
    return QuantizedLinear(
        linear,
        quantization_precision=quantization_precision,
        scale_precision=scale_precision,
        block_size=block_size,
        quantize_weights=quantize_weights,
    )


def load_qwen3_denoiser(
    path: str | Path,
    quantization_precision: str | None = "int8",
    scale_precision: str | None = "fp16",
    block_size: int | None = 128,
    target_linear_names: tuple[str, ...] = DEFAULT_TARGET_LINEAR_NAMES,
    checkpoint_path: str | Path | None = None,
    quantized_state_path: str | Path | None = None,
) -> Flux2Transformer2DModel:
    model_dir = _resolve_model_dir(path)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Denoiser directory not found: {model_dir}")
    if checkpoint_path is not None and quantized_state_path is not None:
        raise ValueError("checkpoint_path and quantized_state_path are mutually exclusive.")

    if quantization_precision is None:
        if scale_precision is not None or block_size is not None:
            raise ValueError("scale_precision and block_size require quantization_precision.")
    else:
        quantization_precision = quantization_precision.lower()
        if quantization_precision not in {"fp16", "int8", "int4"}:
            raise ValueError('quantization_precision must be "fp16", "int8", or "int4".')
        scale_precision = (scale_precision or "fp32").lower()
        if scale_precision not in {"fp32", "fp16", "e8m0"}:
            raise ValueError('scale_precision must be "fp32", "fp16", or "e8m0".')
        block_size = 64 if block_size is None else block_size
        if block_size not in {32, 64, 128}:
            raise ValueError("block_size must be one of: 32, 64, 128.")

    model = _build_model_from_config(model_dir)
    if quantized_state_path is None:
        checkpoint_path = (
            Path(checkpoint_path).expanduser().resolve()
            if checkpoint_path is not None
            else model_dir / "diffusion_pytorch_model.safetensors"
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Denoiser checkpoint not found: {checkpoint_path}")
        _load_local_single_checkpoint(model, checkpoint_path)
        _materialize_meta_tensors(model)
    else:
        quantized_state_path = Path(quantized_state_path)
        if quantization_precision is None:
            raise ValueError("quantized_state_path requires quantization_precision.")
        if not quantized_state_path.is_file():
            raise FileNotFoundError(f"Quantized denoiser checkpoint not found: {quantized_state_path}")
        checkpoint_keys = _list_checkpoint_keys(quantized_state_path)
        _replace_targeted_linear_modules(
            model,
            quantization_precision=quantization_precision,
            scale_precision=scale_precision,
            block_size=block_size,
            target_linear_names=target_linear_names,
            quantize_weights=False,
            checkpoint_keys=checkpoint_keys,
        )
        _load_local_single_checkpoint(model, quantized_state_path)
        _materialize_meta_tensors(model)
    if quantization_precision is not None and quantized_state_path is None:
        _replace_targeted_linear_modules(
            model,
            quantization_precision=quantization_precision,
            scale_precision=scale_precision,
            block_size=block_size,
            target_linear_names=target_linear_names,
        )
    model = model.to(dtype=torch.float16)
    model.eval()
    return model
