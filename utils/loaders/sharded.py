from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch.nn as nn
from safetensors.torch import load_file as safe_load_file


def load_local_sharded_checkpoint(model: nn.Module, folder: Path) -> None:
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
