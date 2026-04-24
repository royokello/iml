from __future__ import annotations

import torch

from utils.quant.linear import QuantizedLinear

__all__ = [
    "TrainableLoraLinear",
    "build_lora_state_dict",
    "inject_trainable_lora_modules",
]


class TrainableLoraLinear(torch.nn.Module):
    def __init__(self, base_module: torch.nn.Module, *, rank: int, alpha: int) -> None:
        super().__init__()
        self.base_module = base_module
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.in_features = base_module.in_features
        self.out_features = base_module.out_features

        for parameter in self.base_module.parameters():
            parameter.requires_grad = False

        parameter_device = None
        for tensor in list(self.base_module.parameters()) + list(self.base_module.buffers()):
            if tensor.is_floating_point():
                parameter_device = tensor.device
                break
        if parameter_device is None:
            parameter_device = torch.device("cpu")

        self.lora_A = torch.nn.Parameter(
            torch.empty((rank, self.in_features), device=parameter_device, dtype=torch.float32)
        )
        self.lora_B = torch.nn.Parameter(
            torch.zeros((self.out_features, rank), device=parameter_device, dtype=torch.float32)
        )
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        base_output = self.base_module(input)
        lora_hidden = torch.nn.functional.linear(input, self.lora_A.to(dtype=input.dtype))
        lora_output = torch.nn.functional.linear(lora_hidden, self.lora_B.to(dtype=input.dtype))
        return base_output + lora_output * self.scaling


def inject_trainable_lora_modules(
    module: torch.nn.Module,
    *,
    target_linear_names: tuple[str, ...],
    rank: int,
    alpha: int,
) -> list[str]:
    injected_module_names: list[str] = []

    def _inject(parent: torch.nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if child_name in target_linear_names and isinstance(child, (torch.nn.Linear, QuantizedLinear)):
                setattr(parent, child_name, TrainableLoraLinear(child, rank=rank, alpha=alpha))
                injected_module_names.append(full_name)
                continue
            _inject(child, full_name)

    _inject(module)
    return injected_module_names


def build_lora_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    lora_state_dict: dict[str, torch.Tensor] = {}
    for module_name, child in module.named_modules():
        if not isinstance(child, TrainableLoraLinear):
            continue
        lora_state_dict[f"{module_name}.lora_A.weight"] = child.lora_A.detach().cpu()
        lora_state_dict[f"{module_name}.lora_B.weight"] = child.lora_B.detach().cpu()
        lora_state_dict[f"{module_name}.alpha"] = torch.tensor(float(child.alpha), dtype=torch.float32)
    return lora_state_dict
