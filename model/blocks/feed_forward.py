import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks.linear import Linear
from model.utils.activations import quantize_input_and_attach_scale

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,          # input and output embedding dimension
        mult: float = 4.0, # expansion factor for hidden dimension (dim → dim*mult)
        dropout: float = 0.0,
        use_gelu: bool = True,  # True: GELU, False: SiLU (both common in UNets/transformers)
    ):
        super().__init__()

        # hidden dimension is typically 4x the input dim
        hidden_dim = int(dim * mult)

        # first linear expands dim → hidden_dim
        self.fc1 = Linear(
            in_features=dim,
            out_features=hidden_dim,
            bias=True,
        )

        # second linear projects hidden_dim → dim
        self.fc2 = Linear(
            in_features=hidden_dim,
            out_features=dim,
            bias=True,
        )

        # dropout for regularization / smoothing
        self.dropout = nn.Dropout(dropout)

        # choose activation
        self.use_gelu = use_gelu
        self.act = F.gelu if use_gelu else F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, dim] tokens from attention

        # project to higher-dimensional hidden space
        tensor_q = quantize_input_and_attach_scale(self.fc1, x)
        x = self.fc1(tensor_q)  # [B, N, hidden_dim]

        # apply non-linear activation token-wise
        x = self.act(x)          # [B, N, hidden_dim]

        # optional dropout in the hidden space
        x = self.dropout(x)      # [B, N, hidden_dim]

        # project back down to original dim
        tensor_q = quantize_input_and_attach_scale(self.fc2, x)
        x = self.fc2(tensor_q)  # [B, N, dim]

        # another dropout on the output (standard transformer style)
        x = self.dropout(x)      # [B, N, dim]

        return x
