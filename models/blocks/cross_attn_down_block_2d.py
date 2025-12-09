import torch
import torch.nn as nn

from models.blocks.down_sample_2d import Downsample2D
from models.blocks.resnet_block_2d import ResnetBlock2D
from models.blocks.spatial_transformer import SpatialTransformer

class CrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,            # channels in from previous block
        out_channels: int,           # channels this block outputs
        temb_channels: int,          # time embedding dim
        num_layers: int,             # how many resnet+attn pairs
        num_attention_heads: int,    # transformer heads
        head_dim: int,               # per-head dim, so inner_dim = heads * head_dim
        cross_attention_dim: int,    # text/context embedding dim
        add_downsample: bool = True, # add downsample at end of block
        num_groups: int = 32,        # GroupNorm groups
        transformer_depth: int = 1,  # BasicTransformerBlocks per SpatialTransformer
        use_conv_down: bool = True,  # conv stride=2 vs avgpool
    ):
        super().__init__()

        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        curr_in_channels = in_channels

        for i in range(num_layers):
            # ResNet maps curr_in_channels -> out_channels
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=curr_in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    num_groups=num_groups,
                )
            )

            # Attention works on out_channels (resnet output)
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

        if add_downsample:
            self.downsamplers.append(
                Downsample2D(out_channels, use_conv=use_conv_down)
            )

    def forward(
        self,
        x: torch.Tensor,                    # [B, C, H, W]
        temb: torch.Tensor,                 # [B, temb_channels]
        encoder_hidden_states: torch.Tensor,# [B, T, cross_attention_dim]
        attention_mask: torch.Tensor | None = None,
        debug: bool = False,
    ):
        """
        Returns:
            hidden_states: final output of this block
            res_samples: tuple of intermediate outputs for skip connections
        """
        res_samples = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            x = resnet(x, temb, debug=debug)  # local conv + time embedding
            x = attn(
                x,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )                    # spatial + cross attention
            res_samples += (x,)

        for downsampler in self.downsamplers:
            x = downsampler(x, debug=debug)   # optional H,W downsample
            res_samples += (x,)

        return x, res_samples
