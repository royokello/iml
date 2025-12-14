import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearFP16(nn.Linear):
    """
    Standard Linear layer using fp16 parameters and compute.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        self.to(torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.to(torch.float16))


class LinearInt8(nn.Module):
    """
    Linear layer that stores int8 weights + fp16 per-channel scales.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.zeros(out_features, in_features, dtype=torch.int8),
        )
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features, dtype=torch.float16),
        )
        self._weights_loaded = False

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def enable_int8(self, weight: torch.Tensor, weight_scale: torch.Tensor) -> None:
        """
        Set per-channel int8 weights + fp16 scales.

        weight: [out_features, in_features], dtype int8
        weight_scale: [out_features], dtype fp16 or fp32
        """
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"weight shape {weight.shape} != "
                f"({self.out_features}, {self.in_features})"
            )
        if weight_scale.shape != (self.out_features,):
            raise ValueError(
                f"weight_scale shape {weight_scale.shape} != ({self.out_features},)"
            )

        self.weight.copy_(weight.to(torch.int8))
        self.weight_scale.copy_(weight_scale.to(torch.float16))
        self._weights_loaded = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, got {x.shape[-1]}"
            )

        if not self._weights_loaded:
            raise RuntimeError(
                "Per-channel scaled int8 weights must be loaded before calling forward."
            )

        x_fp16 = x.to(torch.float16)
        bias_fp16 = self.bias.to(torch.float16) if self.bias is not None else None

        w_int8 = self.weight.to(torch.float16)
        scale = self.weight_scale.to(torch.float16).unsqueeze(1)
        weight_fp16 = w_int8 * scale

        y_fp16 = F.linear(x_fp16, weight_fp16, bias_fp16)
        return y_fp16.to(torch.float16)
