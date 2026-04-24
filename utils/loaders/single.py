from __future__ import annotations

import gc
import time
from collections.abc import Callable
from pathlib import Path

import torch.nn as nn
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
    return incompatible
