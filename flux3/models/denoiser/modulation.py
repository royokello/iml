from __future__ import annotations

import torch
import torch.nn as nn


class Flux3Modulation(nn.Module):
    """One set of per-token shift, scale and gate values.

    The input is an embedding that the caller already passed through SiLU.
    The shape is `(batch_size, seq_len, dim)`. The output has the shape
    `(batch_size, seq_len, dim * 3)`.
    """

    def __init__(self, dim: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 3, bias=bias)

    def forward(self, temb: torch.Tensor) -> torch.Tensor:
        return self.linear(temb)


class Flux3AdaLayerNorm(nn.Module):
    """Output layer norm with a per-token scale and shift.

    The `temb` input has the shape `(batch_size, seq_len, dim)`.
    """

    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.act_fn = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2, bias=bias)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.act_fn(temb))
        scale, shift = torch.chunk(emb, 2, dim=-1)
        return self.norm(hidden_states) * (1 + scale) + shift
