import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    """
    FP16 weight linear layer that promotes inputs to FP32 for computation.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.float16)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: fp16/fp32 activations shaped [..., in_features]

        Returns:
            fp16 tensor shaped [..., out_features]
        """
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, got {x.shape[-1]}"
            )

        weight_fp32 = self.weight.to(torch.float32)
        bias_fp32 = self.bias.to(torch.float32) if self.bias is not None else None
        y_fp32 = F.linear(x.to(torch.float32), weight_fp32, bias_fp32)
        return y_fp32.to(torch.float16)
