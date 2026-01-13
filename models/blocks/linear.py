import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantLinear(nn.Module):
    """
    Block-quantized int8 Linear with fp16 dequantization and compute.
    Expects:
      - weight: int8 tensor shaped [out_features, in_features]
      - weight_scale: fp16 tensor shaped [ceil(out_features * in_features / 32)]
      - bias: fp16 parameter (optional)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.zeros(out_features, in_features, dtype=torch.int8),
        )
        num_blocks = math.ceil((out_features * in_features) / 32)
        self.register_buffer(
            "weight_scale",
            torch.ones(num_blocks, dtype=torch.float16),
        )
        self._weights_loaded = False

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
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
            weight: [out_features, in_features], dtype int8
            weight_scale: [num_blocks], dtype fp16 or fp32
        """
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"weight shape {weight.shape} != "
                f"({self.out_features}, {self.in_features})"
            )
        expected_blocks = math.ceil((self.out_features * self.in_features) / 32)
        scale_flat = weight_scale.flatten()
        if scale_flat.shape[0] != expected_blocks:
            raise ValueError(
                f"weight_scale items {scale_flat.shape[0]} != expected {expected_blocks}"
            )

        self.weight.copy_(weight.to(torch.int8))
        self.weight_scale.copy_(scale_flat.to(torch.float16))
        self._weights_loaded = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, got {x.shape[-1]}"
            )
        if not self._weights_loaded:
            raise RuntimeError(
                "Block-quantized int8 weights must be loaded before calling forward."
            )

        x_fp16 = x.to(torch.float16)
        bias_fp16 = self.bias.to(torch.float16) if self.bias is not None else None

        w_fp16 = self.weight.to(torch.float16)
        w_flat = w_fp16.flatten()
        target_len = w_flat.numel()
        pad_len = (32 - (target_len % 32)) % 32

        if pad_len > 0:
            w_flat = F.pad(w_flat, (0, pad_len))

        w_reshaped = w_flat.view(-1, 32)
        scale_reshaped = self.weight_scale.to(torch.float16).view(-1, 1)

        w_dequant = w_reshaped * scale_reshaped
        w_dequant_flat = w_dequant.flatten()

        if pad_len > 0:
            w_dequant_flat = w_dequant_flat[:-pad_len]

        weight_fp16 = w_dequant_flat.view(self.out_features, self.in_features)

        y_fp16 = F.linear(x_fp16, weight_fp16, bias_fp16)
        return y_fp16.to(torch.float16)
