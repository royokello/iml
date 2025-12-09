from typing import Sequence

import torch
import torch.nn as nn

from models.blocks.resnet_block_2d import ResnetBlock2D
from models.blocks.up_sample_2d import UpSample2D

class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,        # channels from previous block (current x)
        out_channels: int,       # channels we want to output from this block
        temb_channels: int,      # time embedding dimension
        num_layers: int,         # how many ResNet+skip layers in this block
        add_upsample: bool = True,  # whether to upsample at the end of the block
        num_groups: int = 32,    # GroupNorm groups for ResnetBlock2D
        skip_channels_per_layer: Sequence[int] | None = None,
        use_conv_up: bool = True # whether UpSample2D uses conv after interpolate
    ):
        super().__init__()

        self.resnets = nn.ModuleList()   # holds the ResNet layers in this up block
        self.upsamplers = nn.ModuleList()# optional learned upsampling at the end

        curr_in_channels = in_channels   # channels coming into the first layer

        if skip_channels_per_layer is None:
            skip_channels_per_layer = (out_channels,) * num_layers
        else:
            skip_channels_per_layer = tuple(skip_channels_per_layer)
            if len(skip_channels_per_layer) != num_layers:
                raise ValueError(
                    "skip_channels_per_layer must match num_layers "
                    f"(got {len(skip_channels_per_layer)} vs {num_layers})"
                )

        for i in range(num_layers):
            # each layer concatenates a skip tensor [B, out_channels, H, W]
            # with current x [B, curr_in_channels, H, W]
            # so ResNet input channels = curr_in_channels + out_channels
            resnet_in = curr_in_channels + skip_channels_per_layer[i]

            # ResNet maps concatenated channels down to out_channels
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    num_groups=num_groups,
                )
            )

            # after the first layer, the block's running channels become out_channels
            curr_in_channels = out_channels

        # optional upsampling stage at the end of the block
        if add_upsample:
            self.upsamplers.append(
                UpSample2D(out_channels, use_conv=use_conv_up)
            )

    def forward(
        self,
        x: torch.Tensor,                         # [B, C_in, H, W] current features from previous block
        temb: torch.Tensor,                      # [B, temb_channels] time embedding
        res_hidden_states_list: list[torch.Tensor],  # list/stack of skip connections from down path
        debug: bool = False,
    ) -> torch.Tensor:
        def pop_matching_skip(target_shape: torch.Size) -> torch.Tensor:
            while res_hidden_states_list:
                res_hidden = res_hidden_states_list.pop()
                if res_hidden.shape[2:] == target_shape[2:]:
                    return res_hidden
            raise RuntimeError("No matching skip tensor for UpBlock2D.")

        for resnet in self.resnets:
            res_hidden = pop_matching_skip(x.shape)
            x = torch.cat([x, res_hidden], dim=1)   # [B, C_in + C_skip, H, W]

            # run through ResNet to fuse skip + current and inject time embedding
            x = resnet(x, temb, debug=debug)                     # [B, out_channels, H, W]

        # optionally upsample spatial resolution at the end of the block
        for upsampler in self.upsamplers:
            x = upsampler(x, debug=debug)                        # [B, out_channels, 2H, 2W]

        # return updated features to feed into the next up block
        return x
