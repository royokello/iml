from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load_file

from .model import TrainableLoraLinear


def load_checkpoint(transformer: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint_state = safe_load_file(str(checkpoint_path), device="cpu")
    for module_name, child in transformer.named_modules():
        if not isinstance(child, TrainableLoraLinear):
            continue

        lora_a_key = f"{module_name}.lora_A.weight"
        lora_b_key = f"{module_name}.lora_B.weight"
        if lora_a_key not in checkpoint_state and lora_b_key not in checkpoint_state:
            child.lora_A.data.zero_()
            child.lora_B.data.zero_()
            continue
        if lora_a_key not in checkpoint_state or lora_b_key not in checkpoint_state:
            raise KeyError(f"Incomplete LoRA weights for {module_name} in checkpoint {checkpoint_path}")

        child.lora_A.data.copy_(
            checkpoint_state[lora_a_key].to(device=child.lora_A.device, dtype=child.lora_A.dtype)
        )
        child.lora_B.data.copy_(
            checkpoint_state[lora_b_key].to(device=child.lora_B.device, dtype=child.lora_B.dtype)
        )


__all__ = ["load_checkpoint"]
