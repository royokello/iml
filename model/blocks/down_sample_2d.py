import torch
import torch.nn as nn
import torch.nn.functional as F

class Downsample2D(nn.Module):
    def __init__(
        self,
        channels: int,       # number of feature channels coming into the block
        use_conv: bool = True,  # whether to do learned downsampling (conv stride 2) or plain pooling
    ):
        super().__init__()   # initialize nn.Module internals

        self.channels = channels   # store channel count for reference
        self.use_conv = use_conv   # remember if we should use a conv or not

        if use_conv:
            # learned downsampling: conv with stride=2 halves H and W
            self.op = nn.Conv2d(
                in_channels=channels,   # same number of channels in
                out_channels=channels,  # and out; SD changes channels in ResBlocks, not here
                kernel_size=3,          # 3x3 kernel captures local neighborhood
                stride=2,               # stride 2 → downsample by factor of 2
                padding=1,              # padding 1 keeps things centered when shrinking
            )
        else:
            # non-learned downsampling: average pooling with kernel_size=2, stride=2
            self.op = nn.AvgPool2d(
                kernel_size=2,          # pool over 2x2 patches
                stride=2,               # move 2 pixels each step → halves H and W
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply the chosen downsampling operator (conv or pooling) to reduce spatial size
        return self.op(x)
