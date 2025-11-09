import torch
import torch.nn as nn

from model.blocks.resnet_block_2d import ResnetBlock2D
from model.blocks.spatial_transformer import SpatialTransformer
from model.blocks.up_sample_2d import UpSample2D

class CrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int,
        num_attention_heads: int,
        head_dim: int,
        cross_attention_dim: int,
        add_upsample: bool = True,
        num_groups: int = 32,
        transformer_depth: int = 1,
        use_conv_up: bool = True,
    ):
        super().__init__()

        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        curr_in_channels = in_channels

        for i in range(num_layers):
            # we concat skip features with current x, so resnet input channels are:
            resnet_in = curr_in_channels + out_channels

            self.resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    num_groups=num_groups,
                )
            )

            self.attentions.append(
                SpatialTransformer(
                    in_channels=out_channels,
                    num_heads=num_attention_heads,
                    head_dim=head_dim,
                    depth=transformer_depth,
                    cross_attention_dim=cross_attention_dim,
                    num_groups=num_groups,
                )
            )

            curr_in_channels = out_channels

        if add_upsample:
            self.upsamplers.append(
                UpSample2D(out_channels, use_conv=use_conv_up)
            )

    def forward(
        self,
        x: torch.Tensor,                         # [B, C, H, W]
        temb: torch.Tensor,                      # [B, temb_channels]
        res_hidden_states_list: list[torch.Tensor],  # skip connections from down path
        encoder_hidden_states: torch.Tensor,     # [B, T, cross_attention_dim]
        attention_mask: torch.Tensor | None = None,
    ):
        for resnet, attn in zip(self.resnets, self.attentions):
            # take last skip, concatenate along channels
            res_hidden = res_hidden_states_list.pop()
            x = torch.cat([x, res_hidden], dim=1)

            x = resnet(x, temb)
            x = attn(
                x,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )

        for upsampler in self.upsamplers:
            x = upsampler(x)

        return x
