 
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safe_load_file

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from flux_2_klein_4b.quantize.linear import QuantizedLinear
from utils.dequantize import dequantize_from_block
from utils.quantize import quantize_to_block

from .embedding import Qwen3RotaryEmbedding
from .model import Qwen3TextEncoder


def _resolve_model_dir(path: str | Path) -> Path:
    model_path = Path(path)
    if model_path.is_file():
        return model_path.parent
    return model_path


def _load_local_sharded_checkpoint(model: nn.Module, folder: Path) -> None:
    index_file = folder / "model.safetensors.index.json"
    with index_file.open("r", encoding="utf-8") as handle:
        index = json.load(handle)

    shard_files = sorted(set(index["weight_map"].values()))

    total_shards = len(shard_files)
    for shard_index, shard_file in enumerate(shard_files, start=1):
        shard_start = time.perf_counter()
        state_dict = safe_load_file(str(folder / shard_file))
        load_seconds = time.perf_counter() - shard_start
        apply_start = time.perf_counter()
        model.load_state_dict(state_dict, strict=False, assign=True)
        apply_seconds = time.perf_counter() - apply_start
        del state_dict
        gc.collect()
        total_seconds = time.perf_counter() - shard_start
        print(
            f"    [{shard_index}/{total_shards}] "
            f"load={load_seconds:.3f}s "
            f"apply={apply_seconds:.3f}s "
            f"total={total_seconds:.3f}s"
        )


def _load_local_single_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    shard_start = time.perf_counter()
    state_dict = safe_load_file(str(checkpoint_path))
    load_seconds = time.perf_counter() - shard_start
    apply_start = time.perf_counter()
    model.load_state_dict(state_dict, strict=False, assign=True)
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


def _replace_linear_modules(
    module: nn.Module,
    *,
    quantization_precision: str,
    scale_precision: str,
    block_size: int,
    quantize_weights: bool = True,
) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                QuantizedLinear(
                    child,
                    quantization_precision=quantization_precision,
                    scale_precision=scale_precision,
                    block_size=block_size,
                    quantize_weights=quantize_weights,
                ),
            )
            continue
        _replace_linear_modules(
            child,
            quantization_precision=quantization_precision,
            scale_precision=scale_precision,
            block_size=block_size,
            quantize_weights=quantize_weights,
        )


def _materialize_meta_tensors(model: nn.Module, *, validate: bool = True) -> None:
    for module in model.modules():
        if isinstance(module, Qwen3RotaryEmbedding) and getattr(module.inv_freq, "is_meta", False):
            inv_freq, attention_scaling = module.rope_init_fn(module.config, device="cpu")
            module.register_buffer("inv_freq", inv_freq, persistent=False)
            module.original_inv_freq = module.inv_freq
            module.attention_scaling = attention_scaling

    if not validate:
        return

    unresolved_parameters = [name for name, parameter in model.named_parameters() if getattr(parameter, "is_meta", False)]
    if unresolved_parameters:
        raise RuntimeError(
            "Checkpoint load left parameters on meta: " + ", ".join(unresolved_parameters)
        )

    unresolved_buffers = [name for name, buffer in model.named_buffers() if getattr(buffer, "is_meta", False)]
    if unresolved_buffers:
        raise RuntimeError(
            "Checkpoint load left buffers on meta: " + ", ".join(unresolved_buffers)
        )


def load_qwen3_text_encoder(
    path: str | Path,
    quantization_precision: str | None = None,
    scale_precision: str | None = None,
    block_size: int | None = None,
    quantized_state_path: str | Path | None = None,
) -> Qwen3TextEncoder:
    """Load the local Qwen3 text encoder from a checkpoint directory.

    Args:
        path: Directory containing the local Qwen3 text encoder checkpoint.
        quantization_precision: Optional quantization mode. Use `"fp16"` to
            store linear weights in fp16 blocks, `"int8"` for blockwise int8
            weights, or `"int4"` for packed blockwise int4 weights.
        scale_precision: Storage format for block scales. Used when
            `quantization_precision` is set. Defaults to `"fp32"`.
        block_size: Block size for weight quantization. Used when
            `quantization_precision` is set. Defaults to `64`.
        quantized_state_path: Optional path to a previously saved single-file
            quantized checkpoint. When set, the base config is still loaded
            from `path`, but linear modules are instantiated in quantized form
            and populated from this file instead of re-quantizing the original
            dense shards.

    Returns:
        A ready-to-use `Qwen3TextEncoder` instance.
    """
    model_dir = _resolve_model_dir(path)
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

    config = Qwen3Config.from_pretrained(str(model_dir), local_files_only=True)
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        config.dtype = torch.float16
        with torch.device("meta"):
            model = Qwen3TextEncoder(config)
    finally:
        torch.set_default_dtype(default_dtype)

    if quantized_state_path is None:
        _load_local_sharded_checkpoint(model, model_dir)
        model.tie_weights()
        _materialize_meta_tensors(model)
    else:
        quantized_state_path = Path(quantized_state_path)
        if quantization_precision is None:
            raise ValueError("quantized_state_path requires quantization_precision.")
        if not quantized_state_path.is_file():
            raise FileNotFoundError(f"Quantized checkpoint not found: {quantized_state_path}")
        _replace_linear_modules(
            model,
            quantization_precision=quantization_precision,
            scale_precision=scale_precision or "fp32",
            block_size=block_size or 64,
            quantize_weights=False,
        )
        _load_local_single_checkpoint(model, quantized_state_path)
        _materialize_meta_tensors(model)
    if quantization_precision is not None and quantized_state_path is None:
        _replace_linear_modules(
            model,
            quantization_precision=quantization_precision,
            scale_precision=scale_precision or "fp32",
            block_size=block_size or 64,
        )
    model.eval()
    return model
