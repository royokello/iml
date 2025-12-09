import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.linear import LinearFP16


class GatedActivation(nn.Module):
    """
    Matches Diffusers' GEGLU/SILU block so weights live under `.proj`.
    """

    def __init__(self, in_features: int, hidden_dim: int, use_gelu: bool = True):
        super().__init__()
        self.proj = LinearFP16(
            in_features=in_features,
            out_features=hidden_dim * 2,
            bias=True,
        )
        self.use_gelu = use_gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        value, gate = x.chunk(2, dim=-1)
        act_fn = F.gelu if self.use_gelu else F.silu
        return value * act_fn(gate)


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

        self.net = nn.Sequential(
            GatedActivation(dim, hidden_dim, use_gelu=use_gelu),
            nn.Dropout(dropout),
            LinearFP16(
                in_features=hidden_dim,
                out_features=dim,
                bias=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
