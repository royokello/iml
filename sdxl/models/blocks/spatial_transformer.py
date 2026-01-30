import torch
import torch.nn as nn

from ..blocks.basic_transformer_block import BasicTransformerBlock
from ..blocks.linear import QuantLinear


class SpatialTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        head_dim: int,
        depth: int = 1,
        cross_attention_dim: int | None = None,
        num_groups: int = 32,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.inner_dim = num_heads * head_dim

        self.norm = nn.GroupNorm(num_groups, in_channels, eps=1e-5, affine=True)
        self.proj_in = QuantLinear(
            in_features=in_channels,
            out_features=self.inner_dim,
            bias=True,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=self.inner_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for _ in range(depth)
            ]
        )

        self.proj_out = QuantLinear(
            in_features=self.inner_dim,
            out_features=in_channels,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, _, h, w = x.shape
        residual = x

        x = self.norm(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.proj_in(x)
        x = x.view(b, h * w, self.inner_dim)

        for block in self.transformer_blocks:
            x = block(
                hidden_states=x,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )

        x = x.view(b, h, w, self.inner_dim)
        x = self.proj_out(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x + residual
        return x
