import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks.linear import Linear


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        mult: float = 4.0,
        dropout: float = 0.0,
        use_gelu: bool = True,
    ):
        super().__init__()

        hidden_dim = int(dim * mult)

        self.fc1 = Linear(
            in_features=dim,
            out_features=hidden_dim * 2,
            bias=True,
        )
        self.fc2 = Linear(
            in_features=hidden_dim,
            out_features=dim,
            bias=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.use_gelu = use_gelu
        self.act = F.gelu if use_gelu else F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        value, gate = x.chunk(2, dim=-1)
        x = value * self.act(gate)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
