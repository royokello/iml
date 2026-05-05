from __future__ import annotations

import torch
import torch.nn.functional as F
from utils.quant.fro.intermediate import dequantize_from_intermediate
from utils.quant.linear import QuantizedLinear
from utils.quant.to.intermediate import quantize_to_intermediate


__all__ = [
    "TrainableLoraLinear",
    "build_lora_state_dict",
    "inject_trainable_lora_modules",
]


class QuantSavedLoraBranchFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lora_A, lora_B, scaling):
        qinput, input_sub_sums, input_super_scales = quantize_to_intermediate(input)

        ctx.input_shape = input.shape
        ctx.input_dtype = input.dtype
        ctx.scaling = scaling
        ctx.save_for_backward(qinput, input_super_scales, lora_A, lora_B)

        lora_hidden = F.linear(input, lora_A.to(dtype=input.dtype))
        lora_output = F.linear(lora_hidden, lora_B.to(dtype=input.dtype))
        return lora_output * scaling

    @staticmethod
    def backward(ctx, grad_output):
        qinput, input_super_scales, lora_A, lora_B = ctx.saved_tensors
        scaling = ctx.scaling

        input = dequantize_from_intermediate(
            qinput,
            input_super_scales,
            ctx.input_shape,
        ).to(dtype=grad_output.dtype)

        lora_A_f = lora_A.to(dtype=grad_output.dtype)
        lora_B_f = lora_B.to(dtype=grad_output.dtype)

        scaled_grad = grad_output * scaling

        lora_hidden = F.linear(input, lora_A_f)

        grad_lora_B = scaled_grad.reshape(-1, scaled_grad.shape[-1]).T.matmul(
            lora_hidden.reshape(-1, lora_hidden.shape[-1])
        )

        grad_lora_hidden = scaled_grad.matmul(lora_B_f)

        grad_lora_A = grad_lora_hidden.reshape(-1, grad_lora_hidden.shape[-1]).T.matmul(
            input.reshape(-1, input.shape[-1])
        )

        grad_input = grad_lora_hidden.matmul(lora_A_f)

        return grad_input, grad_lora_A.to(dtype=lora_A.dtype), grad_lora_B.to(dtype=lora_B.dtype), None

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

        lora_output = QuantSavedLoraBranchFn.apply(
            input,
            self.lora_A,
            self.lora_B,
            float(self.scaling),
        )

        return base_output + lora_output
    
    @staticmethod
    def backward(ctx, grad_output):
        qinput, input_super_scales, lora_A, lora_B = ctx.saved_tensors

        input = dequantize_from_intermediate(
            qinput,
            input_super_scales,
            ctx.input_shape,
        ).to(dtype=grad_output.dtype)

        lora_A_f = lora_A.to(dtype=grad_output.dtype)
        lora_B_f = lora_B.to(dtype=grad_output.dtype)

        scaled_grad = grad_output * ctx.scaling

        lora_hidden = F.linear(input, lora_A_f)

        grad_lora_B = scaled_grad.reshape(-1, scaled_grad.shape[-1]).T.matmul(
            lora_hidden.reshape(-1, lora_hidden.shape[-1])
        )

        grad_lora_hidden = scaled_grad.matmul(lora_B_f)

        grad_lora_A = grad_lora_hidden.reshape(-1, grad_lora_hidden.shape[-1]).T.matmul(
            input.reshape(-1, input.shape[-1])
        )

        grad_input = grad_lora_hidden.matmul(lora_A_f)

        return (
            grad_input,
            grad_lora_A.to(dtype=lora_A.dtype),
            grad_lora_B.to(dtype=lora_B.dtype),
            None,
        )


def inject_trainable_lora_modules(
    module: torch.nn.Module,
    *,
    target_linear_names: tuple[str, ...],
    rank: int,
    alpha: int,
) -> list[str]:
    injected_module_names: list[str] = []

    def _matches_target(full_name: str, child_name: str) -> bool:
        normalized_full_name = ".".join(part for part in full_name.split(".") if not part.isdigit())
        for target_name in target_linear_names:
            if "." not in target_name:
                if child_name == target_name:
                    return True
                continue
            if full_name == target_name or normalized_full_name == target_name:
                return True
            if full_name.endswith(f".{target_name}") or normalized_full_name.endswith(f".{target_name}"):
                return True
        return False

    def _inject(parent: torch.nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if _matches_target(full_name, child_name) and isinstance(child, (torch.nn.Linear, QuantizedLinear)):
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
