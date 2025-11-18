import torch
import torch.nn as nn
from model.blocks.conv_2d import Conv2d
from model.blocks.linear import Linear
from model.utils.quantization import quantize_input_and_attach_scale

class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,   # number of channels coming in from previous layer
        out_channels: int,  # number of channels this block should output
        temb_channels: int, # size of the time embedding vector (timestep embedding)
        num_groups: int = 32, # groups for GroupNorm (typical value in UNets)
    ):
        super().__init__()  # initialize nn.Module internals

        self.in_channels = in_channels   # store input channel count for reference
        self.out_channels = out_channels # store output channel count for reference
        self.temb_channels = temb_channels # store time-embedding size

        # first normalization on the input feature map, stabilizes scale across channels
        self.norm1 = nn.GroupNorm(num_groups, in_channels, eps=1e-5, affine=True)

        # first conv: mixes spatial info and maps in_channels â†’ out_channels
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # linear layer to project the time embedding to match out_channels
        self.time_emb_proj = Linear(
            in_features=temb_channels,
            out_features=out_channels,
            bias=True,
        )

        # second normalization on the intermediate features (now out_channels wide)
        self.norm2 = nn.GroupNorm(num_groups, out_channels, eps=1e-5, affine=True)

        # second conv: refines features, keeps same channel count out_channels â†’ out_channels
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # if in_channels != out_channels we need a 1x1 conv to match shapes for residual add
        if in_channels != out_channels:
            self.conv_shortcut = Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv_shortcut = None  # no shortcut conv needed when channels already match

        # shared activation used after norms and time embedding projection
        self.act = nn.SiLU()  # x * sigmoid(x), used in SD/SDXL

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        # save original input for the residual/skip connection
        residual = x

        # normalize input features to have stable statistics across channels
        h = self.norm1(x)

        # apply non-linearity to increase representational power
        h = self.act(h)

        # first conv: spatial feature extraction and channel mapping
        original_dtype = h.dtype
        tensor_q = quantize_input_and_attach_scale(self.conv1, h, channel_dim=1)
        h = self.conv1(tensor_q).to(original_dtype)

        # if a time embedding is provided, inject it as a per-channel bias
        if temb is not None:
            # project time embedding to out_channels
            temb_act = self.act(temb)
            tensor_q = quantize_input_and_attach_scale(
                self.time_emb_proj, temb_act, channel_dim=temb_act.ndim - 1
            )
            temb_proj = self.time_emb_proj(tensor_q)  # apply activation then linear

            # reshape temb_proj to broadcast across spatial dims (H, W)
            temb_proj = temb_proj[:, :, None, None]

            # add time-dependent bias to feature map
            h = h + temb_proj

        # second normalization keeps post-temporal features well-scaled
        h = self.norm2(h)

        # another non-linearity before the second conv
        h = self.act(h)

        # second conv: refine features without changing channel count
        original_dtype = h.dtype
        tensor_q = quantize_input_and_attach_scale(self.conv2, h, channel_dim=1)
        h = self.conv2(tensor_q).to(original_dtype)

        # if needed, transform residual so its channels match h for addition
        if self.conv_shortcut is not None:
            original_dtype = residual.dtype
            tensor_q = quantize_input_and_attach_scale(self.conv_shortcut, residual, channel_dim=1)
            residual = self.conv_shortcut(tensor_q).to(original_dtype)

        # final residual add: output = transformed_input + shortcut(input)
        return residual + h

