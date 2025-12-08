import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks.conv_2d import Conv2d


class UpSample2D(nn.Module):
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
                padding=1,
            )
        else:
            self.conv = None

    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        if self.conv is not None:
            if debug:
                print("[debug] upsample conv input stats:", float(x.min()), float(x.max()))
            x = self.conv(x, debug=debug)
            if debug:
                print("[debug] upsample conv output stats:", float(x.min()), float(x.max()))

        return x
