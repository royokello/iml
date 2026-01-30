import torch
import torch.nn as nn

from ..blocks.conv_2d import QuantConv1x1


def _linear_fp16(in_features: int, out_features: int, bias: bool = True) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, bias=bias)
    return layer.to(torch.float16)


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_groups: int = 32,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temb_channels = temb_channels

        self.norm1 = nn.GroupNorm(num_groups, in_channels, eps=1e-5, affine=True)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        ).to(torch.float16)

        self.time_emb_proj = _linear_fp16(
            in_features=temb_channels,
            out_features=out_channels,
            bias=True,
        )

        self.norm2 = nn.GroupNorm(num_groups, out_channels, eps=1e-5, affine=True)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        ).to(torch.float16)

        if in_channels != out_channels:
            self.conv_shortcut = QuantConv1x1(in_channels, out_channels, bias=True)
        else:
            self.conv_shortcut = None

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, temb: torch.Tensor, debug: bool = False) -> torch.Tensor:
        residual = x

        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        if temb is not None:
            temb_act = self.act(temb)
            temb_proj = self.time_emb_proj(temb_act)
            temb_proj = temb_proj[:, :, None, None]
            h = h + temb_proj

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        out_fp32 = residual.to(torch.float32) + h.to(torch.float32)
        max_fp16 = torch.tensor(65504.0, dtype=out_fp32.dtype, device=out_fp32.device)
        return out_fp32.clamp(-max_fp16, max_fp16).to(torch.float16)
