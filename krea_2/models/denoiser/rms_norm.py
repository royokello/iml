from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Krea2RMSNorm(nn.Module):
    """RMSNorm with a zero-centered scale.

    The checkpoint stores the scale offset from 1, so the effective multiplier
    is `weight + 1`. The normalization always runs in float32 and the result is
    cast back to the input dtype.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        weight = self.weight.float() + 1.0
        hidden_states = F.rms_norm(hidden_states.float(), (self.dim,), weight=weight, eps=self.eps)
        return hidden_states.to(dtype)


__all__ = ["Krea2RMSNorm"]
