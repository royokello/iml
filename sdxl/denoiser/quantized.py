from __future__ import annotations

import torch
import torch.nn as nn

from utils.quant.linear import QuantizedLinear
from utils.quant.qlinear import QuantizedLinearFrozenFn


class RecomputeQuantizedLinear(QuantizedLinear):
    """A `QuantizedLinear` that rebuilds its weight in `backward`.

    `utils.quant.linear.QuantizedLinear` dequantizes the weight into a tensor
    that `F.linear` saves for backward. That is an fp16 copy of every weight
    per forward pass, about 4.5 GB for the SDXL UNet.

    This class runs the same forward and, when the input needs a gradient,
    routes through `QuantizedLinearFrozenFn`. That function re-dequantizes the
    weight in `backward` instead of saving it.

    It subclasses `QuantizedLinear`, so every `isinstance` check in the repo
    still matches it. For training that also streams the packed weights from
    host memory, use `utils.quant.streaming.StreamedQuantizedLinear`.
    """

    def _normal_forward_body(
        self,
        input: torch.Tensor,
        bsums: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the plain quantized forward.

        Args:
            input: The input tensor.
            bsums: Unused. Accepted for signature parity with `qlinear`.

        Returns:
            The linear output.
        """
        return super().forward(input)

    def forward(
        self,
        input: torch.Tensor,
        bsums: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the linear, saving the input instead of the weight.

        Args:
            input: The input tensor.
            bsums: Unused. Accepted for signature parity with `qlinear`.

        Returns:
            The linear output.
        """
        if torch.is_grad_enabled() and input.requires_grad:
            return QuantizedLinearFrozenFn.apply(input, self)
        return super().forward(input)


__all__ = ["RecomputeQuantizedLinear"]
