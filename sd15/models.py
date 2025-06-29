import math
from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from diffusers import UNet2DConditionModel


# -----------------------------------------------------------------------------
#  Low‑level Quantised layers
# -----------------------------------------------------------------------------

class QuantConv2d(nn.Module):
    """Row‑wise INT8 Conv2d with runtime de‑quant."""

    def __init__(
        self,
        weight_int8: torch.Tensor,
        scale: torch.Tensor,
        bias: Union[torch.Tensor, None],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        groups: int,
    ) -> None:
        super().__init__()
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("scale", scale.half())  # store fp16 row‑scale
        if bias is not None:
            self.register_buffer("bias", bias.half())
        else:
            self.bias = None

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        # INT8 → FP16 just‑in‑time
        w = self.weight_int8.to(dtype=torch.float16) * self.scale.view(-1, 1, 1, 1)
        x = x.to(dtype=torch.float16)
        return F.conv2d(
            x,
            w,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class QuantConvTranspose2d(nn.Module):
    """Row‑wise INT8 ConvTranspose2d with runtime de‑quant."""

    def __init__(
        self,
        weight_int8: torch.Tensor,
        scale: torch.Tensor,
        bias: Union[torch.Tensor, None],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        output_padding: Tuple[int, int],
        dilation: Tuple[int, int],
        groups: int,
    ) -> None:
        super().__init__()
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("scale", scale.half())
        if bias is not None:
            self.register_buffer("bias", bias.half())
        else:
            self.bias = None

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        w = self.weight_int8.to(dtype=torch.float16) * self.scale.view(-1, 1, 1, 1)
        x = x.to(dtype=torch.float16)
        return F.conv_transpose2d(
            x,
            w,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )


class QuantLinear(nn.Module):
    """Row‑wise INT8 Linear with runtime de‑quant."""

    def __init__(
        self,
        weight_int8: torch.Tensor,
        scale: torch.Tensor,
        bias: Union[torch.Tensor, None],
    ) -> None:
        super().__init__()
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("scale", scale.half())
        if bias is not None:
            self.register_buffer("bias", bias.half())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        w = self.weight_int8.to(dtype=torch.float16) * self.scale.view(-1, 1)
        x = x.to(dtype=torch.float16)
        return F.linear(x, w, self.bias)


# -----------------------------------------------------------------------------
#  Helper – replace FP layers with Quant layers using saved INT8 tensors
# -----------------------------------------------------------------------------

def _replace_module(parent: nn.Module, child_name: str, new_module: nn.Module) -> None:
    """Utility to swap *child_name* inside *parent* with *new_module*."""
    parent._modules[child_name] = new_module  # type: ignore[attr-defined]


def _build_quant_layer(
    mod: nn.Module,
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    bias: Union[torch.Tensor, None],
) -> nn.Module:
    """Return a Quant* layer matching the original module *mod*."""
    if isinstance(mod, nn.Conv2d):
        return QuantConv2d(
            weight_int8,
            scale,
            bias,
            mod.stride,
            mod.padding,
            mod.dilation,
            mod.groups,
        )
    if isinstance(mod, nn.ConvTranspose2d):
        return QuantConvTranspose2d(
            weight_int8,
            scale,
            bias,
            mod.stride,
            mod.padding,
            mod.output_padding,
            mod.dilation,
            mod.groups,
        )
    if isinstance(mod, nn.Linear):
        return QuantLinear(weight_int8, scale, bias)

    raise TypeError(f"Unsupported module type for quantisation: {type(mod)}")


# -----------------------------------------------------------------------------
#  Int8UNet – main class
# -----------------------------------------------------------------------------

class Int8UNet(UNet2DConditionModel):
    """UNet2DConditionModel whose heavy weights live in row‑wise INT8 on GPU."""

    @classmethod
    def from_quantised(
        cls,
        model_dir: Union[str, Path],
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> "Int8UNet":
        """Instantiate and patch an INT8‑weight UNet.

        Parameters
        ----------
        model_dir
            Directory containing the original diffusers checkpoints (for VAE,
            scheduler, config, etc.).  Only the UNet config is read; weights
            come from *weight_file*.
        weight_file
            `.safetensors` file produced by `quant.py` containing INT8 weights
            and row scales.
        device
            'cuda' or 'cpu' placement for *parameters* and buffers.
        dtype
            dtype for **activations** and **temporary de‑quant weights**.
        """
        model_dir = Path(model_dir)
        weight_file = model_dir / "unet" / "diffusion_pytorch_model.safetensors"

        # 1. Build a *structure‑only* UNet (random FP16)
        cfg = UNet2DConditionModel.load_config(model_dir, subfolder="unet")
        unet = cls.from_config(cfg, torch_dtype=dtype, low_cpu_mem_usage=True)

        # 2. Load INT8 tensors (CPU)
        state = load_file(weight_file, device="cpu")  # Dict[str,Tensor]

        # 3. Recursively replace each heavy module with Quant* variant
        for module_name, module in unet.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                key_prefix = f"{module_name}.weight"
                int8_key = f"{key_prefix}_int8"
                scale_key = f"{key_prefix}_scale"
                if int8_key not in state:
                    # leave original FP16 weight (e.g. conv_out) untouched
                    continue
                weight_int8 = state[int8_key].to(device)
                scale = state[scale_key].to(device)
                bias_key = f"{module_name}.bias"
                bias = state.get(bias_key, None)
                if bias is not None:
                    bias = bias.to(device)

                # Build replacement layer
                parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
                parent_module = unet.get_submodule(parent_name) if parent_name else unet
                quant_layer = _build_quant_layer(module, weight_int8, scale, bias)
                _replace_module(parent_module, child_name, quant_layer)

        # 4. Move entire model to *device*
        unet.to(device)
        unet.eval()
        return unet
