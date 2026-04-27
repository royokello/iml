import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.quant.cuda.affine_high import dequantize_from_affine_high as dequantize_from_affine_high_cuda
from utils.quant.cuda.affine_low import dequantize_from_affine_low as dequantize_from_affine_low_cuda
from utils.quant.cuda.symmetric_high import dequantize_from_symmetric_high as dequantize_from_symmetric_high_cuda
from utils.quant.cuda.symmetric_low import dequantize_from_symmetric_low as dequantize_from_symmetric_low_cuda
from utils.quant.to.affine import SUB_BLOCK_SIZE as AFFINE_SUB_BLOCK_SIZE, SUPER_BLOCK_SIZE as AFFINE_SUPER_BLOCK_SIZE, quantize_to_affine
from utils.quant.to.symmetric import SUB_BLOCK_SIZE as SYMMETRIC_SUB_BLOCK_SIZE, SUPER_BLOCK_SIZE as SYMMETRIC_SUPER_BLOCK_SIZE, quantize_to_symmetric
from utils.quant.validators import normalize_quant_method, quant_method_family, quant_method_mode

AFFINE_HIGH_PACKED_WORDS_PER_SUB_BLOCK = 5
SYMMETRIC_LOW_PACKED_WORDS_PER_SUB_BLOCK = 3


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        method: str,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.method = normalize_quant_method(method)

        if quant_method_family(self.method) == "symmetric":
            qweight, sub_scales, super_scales = quantize_to_symmetric(
                linear.weight.detach(),
                mode=quant_method_mode(self.method),
            )
            self.register_buffer("weight", qweight)
            self.register_buffer("sub_scales", sub_scales)
            self.register_buffer("super_scales", super_scales)
            self.sub_mins = None
            self.super_mins = None
        else:
            qweight, sub_scales, sub_mins, super_scales, super_mins = quantize_to_affine(
                linear.weight.detach(),
                mode=quant_method_mode(self.method),
            )
            self.register_buffer("weight", qweight)
            self.register_buffer("sub_scales", sub_scales)
            self.register_buffer("sub_mins", sub_mins)
            self.register_buffer("super_scales", super_scales)
            self.register_buffer("super_mins", super_mins)

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
        module.method = normalize_quant_method(method)

        if quant_method_family(module.method) == "symmetric":
            num_super_blocks, super_block_size = _quantized_2d_shape(
                linear.weight.shape,
                super_block_size=SYMMETRIC_SUPER_BLOCK_SIZE,
                sub_block_size=SYMMETRIC_SUB_BLOCK_SIZE,
            )
            sub_blocks_per_super = super_block_size // SYMMETRIC_SUB_BLOCK_SIZE
            if quant_method_mode(module.method) == "high":
                qweight_shape = (num_super_blocks, super_block_size)
                qweight_dtype = torch.int8
            else:
                qweight_shape = (
                    num_super_blocks,
                    sub_blocks_per_super * SYMMETRIC_LOW_PACKED_WORDS_PER_SUB_BLOCK,
                )
                qweight_dtype = torch.int32
            qweight = torch.empty(qweight_shape, device=linear.weight.device, dtype=qweight_dtype)
            sub_scales = torch.empty(
                (num_super_blocks, sub_blocks_per_super),
                device=linear.weight.device,
                dtype=torch.int8,
            )
            super_scales = torch.empty((num_super_blocks,), device=linear.weight.device, dtype=torch.float16)
            module.register_buffer("weight", qweight)
            module.register_buffer("sub_scales", sub_scales)
            module.register_buffer("super_scales", super_scales)
            module.sub_mins = None
            module.super_mins = None
        else:
            num_super_blocks, super_block_size = _quantized_2d_shape(
                linear.weight.shape,
                super_block_size=AFFINE_SUPER_BLOCK_SIZE,
                sub_block_size=AFFINE_SUB_BLOCK_SIZE,
            )
            sub_blocks_per_super = super_block_size // AFFINE_SUB_BLOCK_SIZE
            if quant_method_mode(module.method) == "low":
                qweight_shape = (num_super_blocks, super_block_size // 2)
                qweight_dtype = torch.uint8
            else:
                qweight_shape = (
                    num_super_blocks,
                    sub_blocks_per_super * AFFINE_HIGH_PACKED_WORDS_PER_SUB_BLOCK,
                )
                qweight_dtype = torch.int32
            qweight = torch.empty(qweight_shape, device=linear.weight.device, dtype=qweight_dtype)
            sub_scales = torch.empty(
                (num_super_blocks, sub_blocks_per_super),
                device=linear.weight.device,
                dtype=torch.int8,
            )
            sub_mins = torch.empty_like(sub_scales)
            super_scales = torch.empty((num_super_blocks,), device=linear.weight.device, dtype=torch.float16)
            super_mins = torch.empty_like(super_scales)
            module.register_buffer("weight", qweight)
            module.register_buffer("sub_scales", sub_scales)
            module.register_buffer("sub_mins", sub_mins)
            module.register_buffer("super_scales", super_scales)
            module.register_buffer("super_mins", super_mins)

        if linear.bias is None:
            module.bias = None
        else:
            bias = torch.empty(linear.bias.shape, device=linear.bias.device, dtype=linear.bias.dtype)
            module.register_buffer("bias", bias)

        return module

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if quant_method_family(self.method) == "symmetric":
            dequantize = (
                dequantize_from_symmetric_high_cuda
                if quant_method_mode(self.method) == "high"
                else dequantize_from_symmetric_low_cuda
            )
            weight = dequantize(
                self.weight,
                self.sub_scales,
                self.super_scales,
                original_shape=(self.out_features, self.in_features),
            )
        else:
            dequantize = (
                dequantize_from_affine_high_cuda
                if quant_method_mode(self.method) == "high"
                else dequantize_from_affine_low_cuda
            )
            weight = dequantize(
                self.weight,
                self.sub_scales,
                self.sub_mins,
                self.super_scales,
                self.super_mins,
                original_shape=(self.out_features, self.in_features),
            )
        weight = weight.view(self.out_features, self.in_features).to(dtype=input.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=input.dtype)
        return F.linear(input, weight, bias)


def _quantized_2d_shape(
    shape: torch.Size,
    *,
    super_block_size: int,
    sub_block_size: int,
) -> tuple[int, int]:
    out_features, in_features = (int(dim) for dim in shape)
    block_size = super_block_size if in_features % super_block_size == 0 else in_features
    if block_size % sub_block_size != 0:
        raise ValueError(
            "Unsupported linear weight shape for quantization: "
            f"{tuple(shape)}. Input features must be divisible by {sub_block_size}."
        )
    blocks_per_row = (in_features + block_size - 1) // block_size
    return out_features * blocks_per_row, block_size
