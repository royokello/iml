import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantConv1x1(nn.Module):
    """
    Block-quantized int8 1x1 conv with fp16 dequantization and compute.
    Expects:
      - weight: int8 tensor shaped [out_channels, in_channels, 1, 1]
      - weight_scale: fp16 tensor shaped [ceil(out_channels * in_channels / 32)]
      - bias: fp16 parameter (optional)
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        total_elems = out_channels * in_channels
        self._pad_len = (32 - (total_elems % 32)) % 32

        self.register_buffer(
            "weight",
            torch.zeros(out_channels, in_channels, 1, 1, dtype=torch.int8),
        )
        num_blocks = math.ceil((out_channels * in_channels) / 32)
        self.register_buffer(
            "weight_scale",
            torch.ones(num_blocks, dtype=torch.float16),
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float16))
        else:
            self.bias = None

    def load_quantized_weights(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> None:
        """
        Load int8 weights + fp16 per-block scales (block32).

        Args:
            weight: [out_channels, in_channels, 1, 1], dtype int8
            weight_scale: [num_blocks], dtype fp16 or fp32
        """
        scale_flat = weight_scale.flatten()

        self.weight.copy_(weight.to(torch.int8))
        self.weight_scale.copy_(scale_flat.to(torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float16:
            x = x.to(torch.float16)

        w_flat = self.weight.flatten()
        pad_len = self._pad_len

        if pad_len > 0:
            w_flat = F.pad(w_flat, (0, pad_len))

        w_reshaped = w_flat.view(-1, 32)
        scale_reshaped = self.weight_scale.view(-1, 1)

        w_dequant = w_reshaped * scale_reshaped
        w_dequant_flat = w_dequant.flatten()

        if pad_len > 0:
            w_dequant_flat = w_dequant_flat[:-pad_len]

        weight_fp16 = w_dequant_flat.view(self.weight.shape)
        bias_fp16 = self.bias

        return F.conv2d(
            x,
            weight_fp16,
            bias_fp16,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )
