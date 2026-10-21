from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from flux2.lora.model import TrainableLoraLinear


def linear_fp32(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply a linear layer in fp32.

    The text stream holds values above the fp16 limit. A plain fp16 matmul
    overflows on them. This function returns an fp32 result for these layers:
    `nn.Linear`, `QuantizedLinear`, and `TrainableLoraLinear`.

    Args:
        layer: The linear layer. Its weight can be fp16 or quantized.
        x: The input. It is cast to fp32.

    Returns:
        The fp32 output.
    """
    x = x.float()
    if isinstance(layer, nn.Linear):
        bias = None if layer.bias is None else layer.bias.float()
        return F.linear(x, layer.weight.float(), bias)
    if isinstance(layer, TrainableLoraLinear):
        lora = F.linear(F.linear(x, layer.lora_A.float()), layer.lora_B.float()) * layer.scaling
        return linear_fp32(layer.base_module, x) + lora
    # A quantized layer dequantizes to the dtype of its input.
    return layer(x)


def feed_forward_fp32_out(ff: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run a feed-forward block with an fp32 output projection.

    The input projection and the activation run in the dtype of `x`. Only
    `linear_out` runs in fp32, because it is the layer that overflows.

    Args:
        ff: The `Flux2FeedForward` block.
        x: The input in fp16.

    Returns:
        The fp32 output.
    """
    return linear_fp32(ff.linear_out, ff.act_fn(ff.linear_in(x)))
