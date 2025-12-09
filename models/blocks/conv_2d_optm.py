import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load


_INT8_CONV2D_MODULE = None


def _load_int8_conv2d_extension():
    global _INT8_CONV2D_MODULE
    if _INT8_CONV2D_MODULE is not None:
        return _INT8_CONV2D_MODULE

    repo_root = Path(__file__).resolve().parents[2]
    sources = [
        repo_root / "cuda" / "int8_conv2d_1x1.cu",
        repo_root / "cuda" / "int8_conv2d_3x3_im2col.cu",
        repo_root / "cuda" / "int8_conv2d_bindings.cpp",
    ]

    extra_ldflags: list[str] = []
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if os.name == "nt":
        if cuda_home:
            lib_dir = Path(cuda_home) / "lib" / "x64"
            extra_ldflags.append(f"/LIBPATH:{lib_dir}")
        extra_ldflags += ["cublasLt.lib", "cublas.lib"]
    else:
        extra_ldflags += ["-lcublasLt", "-lcublas"]

    _INT8_CONV2D_MODULE = load(
        name="int8_conv2d_im2col",
        sources=[str(src) for src in sources],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        extra_ldflags=extra_ldflags,
    )
    return _INT8_CONV2D_MODULE

# Optimized Conv2d

class Conv2dOptm(nn.Module):
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

        # per-channel fp16 scale for weights, one per output channel
        self.register_buffer(
            "scale_w",
            torch.ones(out_channels, dtype=torch.float16),
        )

        # per-input-channel fp16 scales for activations
        # (set by the previous layer / quantization logic)
        self.register_buffer(
            "scale_x",
            torch.ones(in_channels, dtype=torch.float16),
        )

        # fp16 bias, one per output channel (optional)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float16))
        else:
            self.bias = None

    def load_quantized_weights(self, w_q: torch.Tensor, scale_w: torch.Tensor):
        """
        Load pre-quantized int8 weights and per-channel fp16 scales.

        w_q: int8 tensor [C_out, C_in/groups, K_h, K_w]
        scale_w: tensor with shape [C_out] or [C_out, 1, 1]
        """
        if w_q.shape != self.weight_q.shape:
            raise ValueError(
                f"Expected weight shape {tuple(self.weight_q.shape)}, got {tuple(w_q.shape)}"
            )
        if w_q.dtype is not torch.int8:
            raise TypeError("Quantized weights must be int8")

        expected_shapes = (
            torch.Size([self.out_channels]),
            torch.Size([self.out_channels, 1, 1]),
        )
        if scale_w.shape not in expected_shapes:
            raise ValueError(
                f"scale_w must be shape [{self.out_channels}] or "
                f"[{self.out_channels}, 1, 1], got {tuple(scale_w.shape)}"
            )
        scale_w = scale_w.to(torch.float16).view(-1)
        if scale_w.numel() != self.out_channels:
            raise ValueError(
                f"scale_w must provide {self.out_channels} values, got {scale_w.numel()}"
            )

        self.weight_q.copy_(w_q)
        self.scale_w.copy_(scale_w)

    def set_input_scale(self, scale_x: torch.Tensor):
        """
        Set the per-tensor fp16 activation scale (from previous quantization step).

        scale_x: scalar fp16 tensor
        """
        if scale_x.dim() == 4:
            if scale_x.shape != (1, self.in_channels, 1, 1):
                raise ValueError(
                    "scale_x must be shape [1, C_in, 1, 1] when 4-dimensional"
                )
            scale_x = scale_x.view(-1)
        elif scale_x.dim() != 1:
            raise ValueError("scale_x must be 1D or 4D [1, C_in, 1, 1]")
        if scale_x.numel() != self.in_channels:
            raise ValueError(
                f"activation scale must match in_channels ({self.in_channels})"
            )
        self.scale_x.copy_(scale_x.to(torch.float16).contiguous())

    def forward(self, x_q: torch.Tensor) -> torch.Tensor:
        """
        x_q: int8 activations [N, C_in, H_in, W_in]
        Returns:
            y: fp16 output [N, C_out, H_out, W_out]
        """

        assert x_q.dtype == torch.int8, "Input to Int8Conv2dDP4A must be int8"
        if not x_q.is_cuda:
            raise RuntimeError("INT8 conv kernels require CUDA tensors")
        if self.groups != 1:
            raise NotImplementedError("Only groups=1 is supported by the custom kernels")

        module = _load_int8_conv2d_extension()

        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dilation_h, dilation_w = self.dilation

        is_1x1_kernel = (
            self.kernel_size == (1, 1)
            and self.stride == (1, 1)
            and self.padding == (0, 0)
            and self.dilation == (1, 1)
        )

        if is_1x1_kernel:
            conv_fn = module.int8_conv2d_1x1
        elif self.kernel_size == (3, 3):
            conv_fn = module.int8_conv2d_3x3_im2col
        else:
            raise NotImplementedError(
                f"No custom kernel available for kernel_size={self.kernel_size}"
            )

        if self.weight_q.device != x_q.device:
            raise RuntimeError("Call .to(device) on the module before running the forward pass")

        bias = self.bias
        if bias is None:
            bias = torch.zeros(self.out_channels, dtype=torch.float16, device=x_q.device)

        bias = bias.contiguous()
        x_q = x_q.contiguous()
        weight_q = self.weight_q.contiguous()

        scale_x_fp32 = self.scale_x.to(torch.float32)
        scale_w_fp32 = self.scale_w.to(torch.float32)
        scale = (scale_w_fp32[:, None] * scale_x_fp32[None, :]).contiguous()

        y_fp32 = conv_fn(
            x_q,
            weight_q,
            bias,
            scale,
            True,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            self.groups,
        )

        return y_fp32.to(torch.float16)
