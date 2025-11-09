import torch
import torch.nn as nn

from model.blocks.down_sample_2d import Downsample2D
from model.blocks.resnet_block_2d import ResnetBlock2D

class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,      # number of input channels
        out_channels: int,     # number of output channels for this block
        temb_channels: int,    # time embedding dimension (for conditioning)
        num_layers: int = 2,   # number of ResNet layers in this block
        add_downsample: bool = True,  # whether to add a downsampling layer
        use_conv_down: bool = True,   # whether downsample is Conv(stride=2)
    ):
        super().__init__()

        self.resnets = nn.ModuleList()     # holds sequential ResnetBlock2D layers
        self.downsamplers = nn.ModuleList() # optional downsampling stage at end

        # Build ResNet layers
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                )
            )

        # Add downsampling operator if requested
        if add_downsample:
            self.downsamplers.append(
                Downsample2D(out_channels, use_conv=use_conv_down)
            )

    def forward(self, x: torch.Tensor, temb: torch.Tensor):
        """
        Forward pass for DownBlock2D.

        Args:
            x: input feature map [B, C, H, W]
            temb: time embedding [B, temb_channels]
        Returns:
            output: downsampled features [B, C_out, H/2, W/2]
            residuals: tuple of intermediate outputs for skip connections
        """
        residuals = ()  # collect intermediate outputs for skip connections

        # Pass through ResNet layers
        for resnet in self.resnets:
            x = resnet(x, temb)
            residuals += (x,)

        # Optional downsample step
        for downsampler in self.downsamplers:
            x = downsampler(x)
            residuals += (x,)

        return x, residuals
