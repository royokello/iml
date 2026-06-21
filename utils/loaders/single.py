from __future__ import annotations

import gc
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file


def load_local_single_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    *,
    key_transform: Callable[[str], str] | None = None,
):
    shard_start = time.perf_counter()
    state_dict = safe_load_file(str(checkpoint_path))
    if key_transform is not None:
        state_dict = {key_transform(name): tensor for name, tensor in state_dict.items()}
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
    return {
        "unexpected_keys": list(incompatible.unexpected_keys),
        "missing_keys": list(incompatible.missing_keys),
    }


def _resolve_owner(model: nn.Module, key: str) -> tuple[nn.Module, str] | None:
    if "." not in key:
        return model, key
    parent_path, _, local = key.rpartition(".")
    try:
        parent = model.get_submodule(parent_path)
    except AttributeError:
        return None
    return parent, local


def _apply_to_parameter(
    parameter: nn.Parameter,
    tensor: torch.Tensor,
    *,
    target_device: torch.device | None,
    pin_memory: bool,
) -> None:
    if parameter.is_floating_point() and tensor.is_floating_point() and tensor.dtype != parameter.dtype:
        tensor = tensor.to(dtype=parameter.dtype)
    elif not tensor.is_floating_point() and parameter.is_floating_point():
        tensor = tensor.to(dtype=parameter.dtype)
    if pin_memory and tensor.device.type == "cpu" and target_device is not None and target_device.type == "cuda":
        tensor = tensor.pin_memory()
    if target_device is not None:
        tensor = tensor.to(target_device, non_blocking=True)
    parameter.data = tensor


def _apply_to_buffer(
    buffer: torch.Tensor,
    tensor: torch.Tensor,
    *,
    target_device: torch.device | None,
    pin_memory: bool,
) -> torch.Tensor:
    if buffer.is_floating_point() and tensor.is_floating_point() and tensor.dtype != buffer.dtype:
        tensor = tensor.to(dtype=buffer.dtype)
    if pin_memory and tensor.device.type == "cpu" and target_device is not None and target_device.type == "cuda":
        tensor = tensor.pin_memory()
    if target_device is not None:
        tensor = tensor.to(target_device, non_blocking=True)
    return tensor


def stream_local_single_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    *,
    key_transform: Callable[[str], str] | None = None,
    target_device: torch.device | str | None = None,
    pin_memory: bool = False,
) -> dict[str, list[str]]:
    """Stream tensors from a safetensors file directly into the model.

    Avoids the full state_dict roundtrip on CPU by reading one tensor at a
    time and assigning it to the matching parameter/buffer. Returns a dict
    listing unexpected keys and missing keys, in the same shape as
    `nn.Module.load_state_dict` return value's `incompatible_keys`.
    """
    checkpoint_path = Path(checkpoint_path)
    target_device = torch.device(target_device) if target_device is not None else None

    named_parameters = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())

    seen: set[str] = set()
    unexpected: list[str] = []
    load_seconds = 0.0
    apply_seconds = 0.0

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for key in keys:
            load_start = time.perf_counter()
            tensor = handle.get_tensor(key)
            load_seconds += time.perf_counter() - load_start

            mapped_key = key_transform(key) if key_transform is not None else key
            seen.add(mapped_key)

            apply_start = time.perf_counter()
            if mapped_key in named_parameters:
                _apply_to_parameter(
                    named_parameters[mapped_key],
                    tensor,
                    target_device=target_device,
                    pin_memory=pin_memory,
                )
            elif mapped_key in named_buffers:
                owner = _resolve_owner(model, mapped_key)
                if owner is None:
                    unexpected.append(mapped_key)
                else:
                    parent_module, local_name = owner
                    new_buffer = _apply_to_buffer(
                        named_buffers[mapped_key],
                        tensor,
                        target_device=target_device,
                        pin_memory=pin_memory,
                    )
                    parent_module._buffers[local_name] = new_buffer
            else:
                unexpected.append(mapped_key)
            apply_seconds += time.perf_counter() - apply_start

    missing: list[str] = []
    for key in named_parameters:
        if key not in seen:
            missing.append(key)

    gc.collect()
    total_seconds = load_seconds + apply_seconds
    print(
        f"    [1/1 stream] "
        f"load={load_seconds:.3f}s "
        f"apply={apply_seconds:.3f}s "
        f"total={total_seconds:.3f}s "
        f"unexpected={len(unexpected)} missing={len(missing)}"
    )
    return {"unexpected_keys": unexpected, "missing_keys": missing}
