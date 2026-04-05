import torch
import torch.nn as nn


class Flux2SwiGLU(nn.Module):
    """
    Flux 2 uses a SwiGLU-style activation in the transformer feedforward sub-blocks, but with the linear projection
    layer fused into the first linear layer of the FF sub-block. Thus, this module has no trainable parameters.
    """

    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        x = self.gate_fn(x1) * x2
        return x
