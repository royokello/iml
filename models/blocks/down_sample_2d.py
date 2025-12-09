import torch
import torch.nn as nn

from models.blocks.conv_2d import Conv2d


class Downsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = True,
    ):
        super().__init__()

        self.channels = channels
        self.use_conv = use_conv

        if use_conv:
            self.conv = Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )
            self.pool = None
        else:
            self.conv = None
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        if self.use_conv and self.conv is not None:
            return self.conv(x, debug=debug)
        return self.pool(x)
