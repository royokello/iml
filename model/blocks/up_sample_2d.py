import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks.conv_2d import Conv2d
from model.utils.quantization import quantize_input_and_attach_scale

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # upsample spatial dims by factor 2 (H,W â†’ 2H,2W)
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        if self.conv is not None:
            tensor_q = quantize_input_and_attach_scale(self.conv, x)
            x = self.conv(tensor_q)

        return x
