from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file


def _resolve_shard_files(
    folder: Path,
    *,
    index_filename: str | None,
    shard_pattern: str | None,
) -> list[str]:
    shard_files: list[str]
    if index_filename is not None and (folder / index_filename).is_file():
        with (folder / index_filename).open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        shard_files = sorted(set(index["weight_map"].values()))
    elif shard_pattern is not None:
        shard_files = sorted(path.name for path in folder.glob(shard_pattern))
    else:
        raise FileNotFoundError(f"Sharded checkpoint index not found: {folder / index_filename}")

    if not shard_files:
        detail = shard_pattern if shard_pattern is not None else index_filename
        raise FileNotFoundError(f"No shard files found in {folder} using {detail}")
    return shard_files


def load_local_sharded_checkpoint(
    model: nn.Module,
    folder: Path,
    *,
    index_filename: str | None = "model.safetensors.index.json",
    shard_pattern: str | None = None,
    key_transform: Callable[[str], str] | None = None,
) -> None:
    shard_files = _resolve_shard_files(folder, index_filename=index_filename, shard_pattern=shard_pattern)

    total_shards = len(shard_files)
    for shard_index, shard_file in enumerate(shard_files, start=1):
        shard_start = time.perf_counter()
        state_dict = safe_load_file(str(folder / shard_file))
        if key_transform is not None:
            state_dict = {key_transform(name): tensor for name, tensor in state_dict.items()}
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
    tensor = tensor.to(dtype=parameter.dtype)
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


def stream_local_sharded_checkpoint(
    model: nn.Module,
    folder: Path,
    *,
    index_filename: str | None = "model.safetensors.index.json",
    shard_pattern: str | None = None,
    key_transform: Callable[[str], str] | None = None,
    target_device: torch.device | str | None = None,
    offloading: bool = False,
) -> dict[str, list[str]]:
    """Stream tensors from a sharded safetensors checkpoint into the model.

    Like `load_local_sharded_checkpoint` but reads one tensor at a time from
    each shard and assigns it directly to the matching parameter or buffer,
    skipping the full state_dict roundtrip on CPU.
    """
    folder = Path(folder)
    shard_files = _resolve_shard_files(folder, index_filename=index_filename, shard_pattern=shard_pattern)
    target_device = torch.device(target_device) if target_device is not None else None

    named_parameters = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())

    seen: set[str] = set()
    unexpected: list[str] = []
    total_load_seconds = 0.0
    total_apply_seconds = 0.0

    total_shards = len(shard_files)
    for shard_index, shard_file in enumerate(shard_files, start=1):
        shard_start = time.perf_counter()
        shard_load_seconds = 0.0
        shard_apply_seconds = 0.0

        with safe_open(str(folder / shard_file), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                load_start = time.perf_counter()
                tensor = handle.get_tensor(key)
                shard_load_seconds += time.perf_counter() - load_start

                mapped_key = key_transform(key) if key_transform is not None else key
                seen.add(mapped_key)

                apply_start = time.perf_counter()
                if mapped_key in named_parameters:
                    _apply_to_parameter(
                        named_parameters[mapped_key],
                        tensor,
                        target_device=target_device,
                        pin_memory=offloading,
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
                            pin_memory=offloading,
                        )
                        parent_module._buffers[local_name] = new_buffer
                else:
                    unexpected.append(mapped_key)
                shard_apply_seconds += time.perf_counter() - apply_start

        total_load_seconds += shard_load_seconds
        total_apply_seconds += shard_apply_seconds
        gc.collect()
        total_seconds = time.perf_counter() - shard_start
        print(
            f"    [{shard_index}/{total_shards} stream] "
            f"load={shard_load_seconds:.3f}s "
            f"apply={shard_apply_seconds:.3f}s "
            f"total={total_seconds:.3f}s"
        )

    missing: list[str] = []
    for key in named_parameters:
        if key not in seen:
            missing.append(key)

    print(
        f"    [stream total] "
        f"load={total_load_seconds:.3f}s "
        f"apply={total_apply_seconds:.3f}s "
        f"unexpected={len(unexpected)} missing={len(missing)}"
    )
    return {"unexpected_keys": unexpected, "missing_keys": missing}
