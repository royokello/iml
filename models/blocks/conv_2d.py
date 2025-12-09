import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        weight_shape = (
            out_channels,
            in_channels // groups,
            kernel_size[0],
            kernel_size[1],
        )
        self.weight = nn.Parameter(torch.zeros(weight_shape, dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        Args:
            x: fp16/fp32 activations [N, C_in, H_in, W_in]

        Returns:
            fp16 output [N, C_out, H_out, W_out]
        """

        if x.dim() != 4:
            raise ValueError("Conv2d expects NCHW input tensors")
        if self.groups != 1:
            raise NotImplementedError("Only groups=1 is supported")

        weight_fp32 = self.weight.to(torch.float32)
        bias_fp32 = self.bias.to(torch.float32) if self.bias is not None else None
        x_fp32 = x.to(torch.float32)

        if debug:
            print(
                "[debug][conv2d] input stats:",
                float(x_fp32.min()),
                float(x_fp32.max()),
                "weight stats:",
                float(weight_fp32.min()),
                float(weight_fp32.max()),
            )
            if bias_fp32 is not None:
                print(
                    "[debug][conv2d] bias stats:",
                    float(bias_fp32.min()),
                    float(bias_fp32.max()),
                )

        y_fp32 = F.conv2d(
            x_fp32,
            weight_fp32,
            bias_fp32,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return y_fp32.to(torch.float16)
