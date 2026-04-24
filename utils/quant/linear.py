import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.quant.double import dequantize_from_double_block, quantize_to_double_block
from utils.quant.double.to import SUB_BLOCKS_PER_SUPER, SUPER_BLOCK_SIZE
from utils.quant.single import BLOCK_SIZE as SINGLE_BLOCK_SIZE, dequantize_from_single_block, quantize_to_single_block


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        method: str,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.method = method.strip().lower()
        if self.method not in {"single", "double"}:
            raise ValueError(f"Unsupported quantization method: {method!r}")

        if self.method == "single":
            qweight, scales = quantize_to_single_block(linear.weight.detach())
            self.register_buffer("weight", qweight)
            self.register_buffer("scales", scales)
            self.super_scales = None
        else:
            qweight, scales, super_scales = quantize_to_double_block(linear.weight.detach())
            self.register_buffer("weight", qweight)
            self.register_buffer("scales", scales)
            self.register_buffer("super_scales", super_scales)

        if linear.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", linear.bias.detach().clone())

    @classmethod
    def from_prequantized(cls, linear: nn.Linear, method: str) -> "QuantizedLinear":
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        module.in_features = linear.in_features
        module.out_features = linear.out_features
        module.method = method.strip().lower()
        if module.method not in {"single", "double"}:
            raise ValueError(f"Unsupported quantization method: {method!r}")

        if module.method == "single":
            qweight = torch.empty_like(linear.weight, device=linear.weight.device, dtype=torch.int8)
            num_blocks = (linear.weight.numel() + SINGLE_BLOCK_SIZE - 1) // SINGLE_BLOCK_SIZE
            scales = torch.empty((num_blocks,), device=linear.weight.device, dtype=torch.float16)
            module.register_buffer("weight", qweight)
            module.register_buffer("scales", scales)
            module.super_scales = None
        else:
            numel = linear.weight.numel()
            num_super_blocks = (numel + SUPER_BLOCK_SIZE - 1) // SUPER_BLOCK_SIZE
            packed_numel = (num_super_blocks * SUPER_BLOCK_SIZE) // 2
            qweight = torch.empty((packed_numel,), device=linear.weight.device, dtype=torch.int8)
            scales = torch.empty(
                (num_super_blocks, SUB_BLOCKS_PER_SUPER),
                device=linear.weight.device,
                dtype=torch.int8,
            )
            super_scales = torch.empty((num_super_blocks,), device=linear.weight.device, dtype=torch.float16)
            module.register_buffer("weight", qweight)
            module.register_buffer("scales", scales)
            module.register_buffer("super_scales", super_scales)

        if linear.bias is None:
            module.bias = None
        else:
            bias = torch.empty(linear.bias.shape, device=linear.bias.device, dtype=linear.bias.dtype)
            module.register_buffer("bias", bias)

        return module

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.method == "single":
            weight = dequantize_from_single_block(self.weight, self.scales)
        else:
            weight = dequantize_from_double_block(
                self.weight,
                self.scales,
                self.super_scales,
                original_numel=self.out_features * self.in_features,
            )
        weight = weight.view(self.out_features, self.in_features).to(dtype=input.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=input.dtype)
        return F.linear(input, weight, bias)
