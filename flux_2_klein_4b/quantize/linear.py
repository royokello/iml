import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dequantize import dequantize_from_block
from utils.quantize import quantize_to_block


def _weight_dtype(quantization_precision: str) -> torch.dtype:
    if quantization_precision == "fp16":
        return torch.float16
    if quantization_precision == "int8":
        return torch.int8
    return torch.uint8


def _scale_dtype(quantization_precision: str, scale_precision: str) -> torch.dtype:
    if quantization_precision == "fp16":
        return torch.float32 if scale_precision == "fp32" else torch.float16 if scale_precision == "fp16" else torch.int8
    if scale_precision == "fp32":
        return torch.float32
    if scale_precision == "fp16":
        return torch.float16
    return torch.int8


def _scale_shape(numel: int, quantization_precision: str, block_size: int) -> tuple[int, ...]:
    if quantization_precision == "fp16":
        return ()
    num_blocks = (numel + block_size - 1) // block_size
    return (num_blocks,)


def _weight_shape(dense_shape: torch.Size, quantization_precision: str) -> tuple[int, ...]:
    if quantization_precision != "int4":
        return tuple(dense_shape)
    numel = 1
    for dim in dense_shape:
        numel *= dim
    return ((numel + 1) // 2,)


class QuantizedLinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        quantization_precision: str,
        scale_precision: str,
        block_size: int,
        *,
        quantize_weights: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.quantization_precision = quantization_precision
        self.scale_precision = scale_precision
        self.block_size = block_size

        if quantize_weights:
            qweight, scales = quantize_to_block(
                linear.weight.detach(),
                block_size=block_size,
                quantization_precision=quantization_precision,
                scaling_precision=scale_precision,
            )
        else:
            weight_dtype = _weight_dtype(quantization_precision)
            qweight = torch.empty(
                _weight_shape(linear.weight.shape, quantization_precision),
                device=linear.weight.device,
                dtype=weight_dtype,
            )
            scales = torch.empty(
                _scale_shape(linear.weight.numel(), quantization_precision, block_size),
                device=linear.weight.device,
                dtype=_scale_dtype(quantization_precision, scale_precision),
            )
        self.register_buffer("weight", qweight)
        self.register_buffer("scales", scales)
        if linear.bias is None:
            self.bias = None
        else:
            if quantize_weights:
                bias = linear.bias.detach().clone()
            else:
                bias = torch.empty(linear.bias.shape, device=linear.bias.device, dtype=linear.bias.dtype)
            self.register_buffer("bias", bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = dequantize_from_block(
            self.weight,
            self.scales,
            block_size=self.block_size,
            quantization_precision=self.quantization_precision,
            scaling_precision=self.scale_precision,
            output_dtype=input.dtype,
            output_shape=(self.out_features, self.in_features),
        )
        bias = None if self.bias is None else self.bias.to(dtype=input.dtype)
        return F.linear(input, weight, bias)
