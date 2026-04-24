from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from pathlib import Path

import torch.nn as nn
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
