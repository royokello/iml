import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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

        weight_fp16 = self.weight.to(torch.float16)
        bias_fp16 = self.bias.to(torch.float16) if self.bias is not None else None
        x_fp16 = x.to(torch.float16)

        y_fp16 = F.conv2d(
            x_fp16,
            weight_fp16,
            bias_fp16,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return y_fp16

class Conv2dBlock32U8F16(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True
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

        # match your checkpoint: int8 tensor stored as `weight`
        self.register_buffer(
            "weight",
            torch.zeros(weight_shape, dtype=torch.int8),
        )

        # per-block scale (block size 32)
        # total elements = out * (in//groups) * kH * kW
        total_elements = out_channels * (in_channels // groups) * kernel_size[0] * kernel_size[1]
        num_blocks = math.ceil(total_elements / 32)
        self.register_buffer(
            "weight_scale",
            torch.ones(num_blocks, dtype=torch.float16),
        )
        self._weights_loaded = False

        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float16)) if bias else None

    def enable_int8(self, weight: torch.Tensor, weight_scale: torch.Tensor) -> None:
        """
        Set int8 weights + fp16 per-block scales (block32).
        
        weight: [out_channels, in_channels // groups, kH, kW], dtype int8
        weight_scale: [num_blocks], dtype fp16 or fp32
        """
        if weight.shape != self.weight.shape:
             raise ValueError(
                f"weight shape {weight.shape} != "
                f"{self.weight.shape}"
            )
        
        expected_blocks = self.weight_scale.shape[0]
        scale_flat = weight_scale.flatten()
        if scale_flat.shape[0] != expected_blocks:
            raise ValueError(
                f"weight_scale items {scale_flat.shape[0]} != expected {expected_blocks}"
            )

        self.weight.copy_(weight.to(torch.int8))
        self.weight_scale.copy_(scale_flat.to(torch.float16))
        self._weights_loaded = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float16:
            x = x.to(torch.float16)

        if not self._weights_loaded:
             # If weights haven't been loaded via enable_int8, we can't properly dequantize
             # because we expect the scale to be setup for block32 dequantization.
             # However, for initialization compatibility or if user wants to run without loading (zeros), 
             # checking _weights_loaded is better.
             # If forced to run, we might raise or warn. Let's raise to be safe like LinearBlock32.
             raise RuntimeError(
                "Block scaled int8 weights must be loaded before calling forward."
            )

        # dequantize to fp16
        # Logic matches LinearBlock32U8F16 but adapted for 4D weight tensor (out, in, kH, kW)
        
        # 1. Access weight as fp16
        w_fp16 = self.weight.to(torch.float16) # (out, in, kH, kW)
        
        # 2. Flatten to 1D to apply block-wise scaling
        w_flat = w_fp16.flatten()
        target_len = w_flat.numel()
        pad_len = (32 - (target_len % 32)) % 32
        
        if pad_len > 0:
            w_flat = F.pad(w_flat, (0, pad_len))
            
        # 3. Reshape to apply scales
        w_reshaped = w_flat.view(-1, 32)
        scale_reshaped = self.weight_scale.to(torch.float16).view(-1, 1)
        
        # 4. Apply scale
        w_dequant = w_reshaped * scale_reshaped
        w_dequant_flat = w_dequant.flatten()
        
        # 5. Remove padding
        if pad_len > 0:
            w_dequant_flat = w_dequant_flat[:-pad_len]
            
        # 6. Reshape back to original conv weight shape
        w = w_dequant_flat.view(self.weight.shape)
        
        b = self.bias if self.bias is not None else None

        return F.conv2d(
            x, 
            w, 
            b, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation, 
            groups=self.groups
        )