import torch
import torch.nn as nn

from models.blocks.resnet_block_2d import ResnetBlock2D
from models.blocks.spatial_transformer import SpatialTransformer

class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_attention_heads: int,
        head_dim: int,
        cross_attention_dim: int,
        num_layers: int = 1,
        transformer_depth: int = 1,
        num_groups: int = 32,
    ):
        super().__init__()

        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    num_groups=num_groups,
                ),
                ResnetBlock2D(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    num_groups=num_groups,
                ),
            ]
        )

        self.attentions = nn.ModuleList(
            [
                SpatialTransformer(
                    in_channels=out_channels,
                    num_heads=num_attention_heads,
                    head_dim=head_dim,
                    depth=transformer_depth,
                    cross_attention_dim=cross_attention_dim,
                    num_groups=num_groups,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        debug: bool = False,
    ) -> torch.Tensor:
        x = self.resnets[0](x, temb, debug=debug)

        for attn in self.attentions:
            x = attn(
                x,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )

        x = self.resnets[1](x, temb, debug=debug)
        return x
