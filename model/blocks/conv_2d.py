import torch
import torch.nn as nn
import torch.nn.functional as F


# This will be your C++/CUDA extension entry point
# from your_extension import int8_conv2d_dp4a_forward


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

        # int8 weights: [C_out, C_in/groups, K_h, K_w]
        # You will load these from your safetensors (already quantized)
        self.register_buffer(
            "weight_q",
            torch.zeros(
                out_channels,
                in_channels // groups,
                kernel_size[0],
                kernel_size[1],
                dtype=torch.int8,
            ),
        )

        # per-tensor fp16 scale for weights (scalar)
        self.register_buffer(
            "scale_w",
            torch.ones((), dtype=torch.float16),
        )

        # per-tensor fp16 scale for activations (scalar)
        # (set by the previous layer / quantization logic)
        self.register_buffer(
            "scale_x",
            torch.ones((), dtype=torch.float16),
        )

        # fp16 bias, one per output channel (optional)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float16))
        else:
            self.bias = None

    def load_quantized_weights(self, w_q: torch.Tensor, scale_w: torch.Tensor):
        """
        Load pre-quantized int8 weights and per-tensor fp16 scale.

        w_q: int8 tensor [C_out, C_in/groups, K_h, K_w]
        scale_w: scalar fp16 tensor
        """
        assert w_q.shape == self.weight_q.shape
        assert w_q.dtype == torch.int8
        assert scale_w.numel() == 1
        self.weight_q.copy_(w_q)
        self.scale_w.copy_(scale_w.to(torch.float16))

    def set_input_scale(self, scale_x: torch.Tensor):
        """
        Set the per-tensor fp16 activation scale (from previous quantization step).

        scale_x: scalar fp16 tensor
        """
        assert scale_x.numel() == 1
        self.scale_x.copy_(scale_x.to(torch.float16))

    def forward(self, x_q: torch.Tensor) -> torch.Tensor:
        """
        x_q: int8 activations [N, C_in, H_in, W_in]
        Returns:
            y: fp16 output [N, C_out, H_out, W_out]
        """

        assert x_q.dtype == torch.int8, "Input to Int8Conv2dDP4A must be int8"

        # Call the custom kernel (to be implemented in C++/CUDA)
        # y_fp16 = int8_conv2d_dp4a_forward(
        #     x_q,
        #     self.weight_q,
        #     self.scale_x,
        #     self.scale_w,
        #     self.bias,
        #     stride=self.stride,
        #     padding=self.padding,
        #     dilation=self.dilation,
        #     groups=self.groups,
        # )

        # For now, fallback to a fake float implementation so shapes work while developing:
        # Dequantize to float, do normal conv, then cast to fp16.
        x_f = x_q.float() * self.scale_x.float()
        w_f = self.weight_q.float() * self.scale_w.float()
        b_f = self.bias.float() if self.bias is not None else None

        y_f = F.conv2d(
            x_f,
            w_f,
            b_f,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        return y_f.to(torch.float16)
