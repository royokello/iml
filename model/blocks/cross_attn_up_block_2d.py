from typing import Sequence

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
        layer_transformer_depths: Sequence[int] | None = None,
        skip_channels_per_layer: Sequence[int] | None = None,
        use_conv_up: bool = True,
    ):
        super().__init__()

        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        if layer_transformer_depths is None:
            layer_transformer_depths = (transformer_depth,) * num_layers
        else:
            layer_transformer_depths = tuple(layer_transformer_depths)
            if len(layer_transformer_depths) != num_layers:
                raise ValueError(
                    "layer_transformer_depths must match num_layers "
                    f"(got {len(layer_transformer_depths)} vs {num_layers})"
                )

        if skip_channels_per_layer is None:
            skip_channels_per_layer = (out_channels,) * num_layers
        else:
            skip_channels_per_layer = tuple(skip_channels_per_layer)
            if len(skip_channels_per_layer) != num_layers:
                raise ValueError(
                    "skip_channels_per_layer must match num_layers "
                    f"(got {len(skip_channels_per_layer)} vs {num_layers})"
                )

        curr_in_channels = in_channels

        for i in range(num_layers):
            # we concat skip features with current x, so resnet input channels are:
            resnet_in = curr_in_channels + skip_channels_per_layer[i]

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
                    depth=layer_transformer_depths[i],
                    cross_attention_dim=cross_attention_dim,
                    num_groups=num_groups,
                )
            )

            curr_in_channels = out_channels

        if add_upsample:
            self.upsamplers.append(
                UpSample2D(out_channels, use_conv=use_conv_up)
            )

    def _pop_skip(
        self,
        res_hidden_states_list: list[torch.Tensor],
        target_shape: torch.Size,
    ) -> torch.Tensor:
        while res_hidden_states_list:
            res_hidden = res_hidden_states_list.pop()
            if res_hidden.shape[2:] == target_shape[2:]:
                return res_hidden
        raise RuntimeError("No matching skip tensor for CrossAttnUpBlock2D.")

    def forward(
        self,
        x: torch.Tensor,                         # [B, C, H, W]
        temb: torch.Tensor,                      # [B, temb_channels]
        res_hidden_states_list: list[torch.Tensor],  # skip connections from down path
        encoder_hidden_states: torch.Tensor,     # [B, T, cross_attention_dim]
        attention_mask: torch.Tensor | None = None,
    ):
        for idx, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
            res_hidden = self._pop_skip(res_hidden_states_list, x.shape)
            x = torch.cat([x, res_hidden], dim=1)

            print(f"[debug] up block resnet {idx} input stats:", float(x.min()), float(x.max()))
            x = resnet(x, temb)
            print(f"[debug] up block resnet {idx} output stats:", float(x.min()), float(x.max()))
            x = attn(
                x,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )
            print(f"[debug] up block attn {idx} output stats:", float(x.min()), float(x.max()))

        for upsampler in self.upsamplers:
            print("[debug] upsampler input stats:", float(x.min()), float(x.max()))
            x = upsampler(x)
            print("[debug] upsampler output stats:", float(x.min()), float(x.max()))

        return x
