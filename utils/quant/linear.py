import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.quant.cuda.affine_high import dequantize_from_affine_high as dequantize_from_affine_high_cuda
from utils.quant.cuda.affine_low import dequantize_from_affine_low as dequantize_from_affine_low_cuda
from utils.quant.cuda.affine_med import dequantize_from_affine_med as dequantize_from_affine_med_cuda
from utils.quant.cuda.symmetric_high import dequantize_from_symmetric_high as dequantize_from_symmetric_high_cuda
from utils.quant.cuda.symmetric_low import dequantize_from_symmetric_low as dequantize_from_symmetric_low_cuda
from utils.quant.cuda.symmetric_med import dequantize_from_symmetric_med as dequantize_from_symmetric_med_cuda
from utils.quant.to.affine import AFFINE_MODES, HALF_SUPER_BLOCK_SIZE as AFFINE_HALF_SUPER_BLOCK_SIZE, SUPER_BLOCK_SIZE as AFFINE_SUPER_BLOCK_SIZE, quantize_to_affine
from utils.quant.to.symmetric import HIGH_BLOCK_SIZE as SYMMETRIC_HIGH_BLOCK_SIZE, SUB_BLOCK_SIZE as SYMMETRIC_SUB_BLOCK_SIZE, SUPER_BLOCK_SIZE as SYMMETRIC_SUPER_BLOCK_SIZE, quantize_to_symmetric
from utils.quant.validators import quant_method_family, quant_method_mode

SYMMETRIC_MED_PACKED_WORDS_PER_SUB_BLOCK = 3
SYMMETRIC_LOW_PACKED_WORDS_PER_SUB_BLOCK = 2
SYMMETRIC_LOW_SUB_SCALE_BITS = 6


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        method: str,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.method = method

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
        module.method = method

        if quant_method_family(module.method) == "symmetric":
            mode = quant_method_mode(module.method)
            if mode == "high":
                out_features, in_features = (int(dim) for dim in linear.weight.shape)
                blocks_per_row = (in_features + SYMMETRIC_HIGH_BLOCK_SIZE - 1) // SYMMETRIC_HIGH_BLOCK_SIZE
                num_blocks = out_features * blocks_per_row
                qweight_shape = (num_blocks, SYMMETRIC_HIGH_BLOCK_SIZE)
                qweight_dtype = torch.int8
                sub_scales_shape = (num_blocks,)
                sub_scales_dtype = torch.float16
                super_scales_shape = None
            else:
                num_super_blocks, super_block_size = _quantized_2d_shape(
                    linear.weight.shape,
                    super_block_size=SYMMETRIC_SUPER_BLOCK_SIZE,
                    sub_block_size=SYMMETRIC_SUB_BLOCK_SIZE,
                )
                sub_blocks_per_super = super_block_size // SYMMETRIC_SUB_BLOCK_SIZE
                if mode == "med":
                    qweight_shape = (
                        num_super_blocks,
                        sub_blocks_per_super * SYMMETRIC_MED_PACKED_WORDS_PER_SUB_BLOCK,
                    )
                    qweight_dtype = torch.int32
                    sub_scales_shape = (num_super_blocks, sub_blocks_per_super)
                    sub_scales_dtype = torch.int8
                else:
                    qweight_shape = (
                        num_super_blocks,
                        sub_blocks_per_super * SYMMETRIC_LOW_PACKED_WORDS_PER_SUB_BLOCK,
                    )
                    qweight_dtype = torch.int32
                    sub_scale_words = (sub_blocks_per_super * SYMMETRIC_LOW_SUB_SCALE_BITS + 31) // 32
                    sub_scales_shape = (num_super_blocks, sub_scale_words)
                    sub_scales_dtype = torch.int32
                super_scales_shape = (num_super_blocks,)
            qweight = torch.empty(qweight_shape, device=linear.weight.device, dtype=qweight_dtype)
            sub_scales = torch.empty(
                sub_scales_shape,
                device=linear.weight.device,
                dtype=sub_scales_dtype,
            )
            super_scales = (
                None
                if super_scales_shape is None
                else torch.empty(super_scales_shape, device=linear.weight.device, dtype=torch.float16)
            )
            module.register_buffer("weight", qweight)
            module.register_buffer("sub_scales", sub_scales)
            module.register_buffer("super_scales", super_scales)
            module.sub_mins = None
            module.super_mins = None
        else:
            mode = quant_method_mode(module.method)
            num_super_blocks, super_block_size, sub_block_size = _affine_quantized_2d_shape(
                linear.weight.shape,
                mode=mode,
            )
            sub_blocks_per_super = super_block_size // sub_block_size
            weight_bits = int(AFFINE_MODES[mode]["weight_bits"])
            qweight_shape = (
                num_super_blocks,
                sub_blocks_per_super * _packed_words_for_values(sub_block_size, weight_bits),
            )
            qweight_dtype = torch.int32
            qweight = torch.empty(qweight_shape, device=linear.weight.device, dtype=qweight_dtype)
            meta_bits = int(AFFINE_MODES[mode]["meta_bits"])
            sub_metadata_words = _packed_words_for_values(sub_blocks_per_super, meta_bits)
            sub_scales = torch.empty(
                (num_super_blocks, sub_metadata_words),
                device=linear.weight.device,
                dtype=torch.int32,
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
            mode = quant_method_mode(self.method)
            if mode == "high":
                dequantize = dequantize_from_symmetric_high_cuda
            elif mode == "med":
                dequantize = dequantize_from_symmetric_med_cuda
            else:
                dequantize = dequantize_from_symmetric_low_cuda
            if mode == "high":
                weight = dequantize(
                    self.weight,
                    self.sub_scales,
                    original_shape=(self.out_features, self.in_features),
                )
            else:
                weight = dequantize(
                    self.weight,
                    self.sub_scales,
                    self.super_scales,
                    original_shape=(self.out_features, self.in_features),
                )
        else:
            mode = quant_method_mode(self.method)
            if mode == "high":
                dequantize = dequantize_from_affine_high_cuda
            elif mode == "med":
                dequantize = dequantize_from_affine_med_cuda
            else:
                dequantize = dequantize_from_affine_low_cuda
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


def _packed_words_for_values(value_count: int, bits: int) -> int:
    return (value_count * bits + 31) // 32


def _select_affine_super_block_size(row_size: int, sub_block_size: int) -> int:
    for super_block_size in (AFFINE_SUPER_BLOCK_SIZE, AFFINE_HALF_SUPER_BLOCK_SIZE):
        if row_size % super_block_size == 0:
            return super_block_size
    raise ValueError(
        "Unsupported linear weight shape for affine quantization: "
        f"in_features={row_size}. Input features must be divisible by "
        f"{AFFINE_HALF_SUPER_BLOCK_SIZE} or {AFFINE_SUPER_BLOCK_SIZE}, and by sub-block size {sub_block_size}."
    )


def _affine_quantized_2d_shape(shape: torch.Size, *, mode: str) -> tuple[int, int, int]:
    out_features, in_features = (int(dim) for dim in shape)
    sub_block_size = int(AFFINE_MODES[mode]["sub_block_size"])
    super_block_size = _select_affine_super_block_size(in_features, sub_block_size)
    blocks_per_row = in_features // super_block_size
    return out_features * blocks_per_row, super_block_size, sub_block_size
