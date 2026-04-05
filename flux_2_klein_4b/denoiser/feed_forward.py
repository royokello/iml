import torch
import torch.nn as nn

from .swi_glu import Flux2SwiGLU


class Flux2FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: float = 3.0,
        inner_dim: int | None = None,
        bias: bool = False,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        # Flux2SwiGLU will reduce the dimension by half
        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.act_fn = Flux2SwiGLU()
        self.linear_out = nn.Linear(inner_dim, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        x = self.act_fn(x)
        x = self.linear_out(x)
        return x
