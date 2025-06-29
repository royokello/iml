import os, re, math, types
import torch, torch.nn as nn, torch.nn.functional as F
from safetensors.torch import load_file
from diffusers import UNet2DConditionModel

import int8_gemm

###############################################################################
# helpers
###############################################################################

def fp8_to_int8(t_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    tmp = t_fp8.float()
    return torch.clamp((tmp / scale).round(), -127, 127).to(torch.int8)

class Int8Linear(nn.Module):
    def __init__(self, w_int8: torch.Tensor, scale_w: float, bias: torch.Tensor = None):
        super().__init__()
        self.register_buffer("w_int8", w_int8)           # (out,in)
        self.scale_w = scale_w
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x …, in
        shape = x.shape
        x = x.reshape(-1, shape[-1])                     # (B,in)
        s_x = x.abs().max() / 127.0                      # dynamic act-scale
        x_q = fp8_to_int8(x, s_x)
        y_i32 = int8_gemm.int8_gemm(x_q, self.w_int8.t())# (B,out)
        y = y_i32.float() * (s_x * self.scale_w)
        if self.bias is not None: y += self.bias
        return y.to(torch.float16).reshape(*shape[:-1], -1)

# Conv → unfold + int8 GEMM
class Int8Conv2d(nn.Module):
    def __init__(self, weight_int8: torch.Tensor, scale_w: float,
                 bias: torch.Tensor, stride, padding, dilation):
        super().__init__()
        self.register_buffer("w_int8", weight_int8)      # (out, in*k*k)
        self.scale_w = scale_w
        self.bias = nn.Parameter(bias) if bias is not None else None
        self.stride, self.padding, self.dilation = stride, padding, dilation
        C_out, flat = self.w_int8.shape
        self.k = int(math.sqrt(flat // (weight_int8.size(1) // 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,C,H,W = x.shape
        k = self.k
        cols = F.unfold(x, k, dilation=self.dilation, padding=self.padding, stride=self.stride)  # B, in*k*k, L
        L = cols.size(-1)
        cols = cols.transpose(1,2).contiguous().view(-1, cols.size(1))                           # (B*L, K)
        s_x = cols.abs().max() / 127.0
        cols_q = fp8_to_int8(cols, s_x)
        y_i32 = int8_gemm.int8_gemm(cols_q, self.w_int8.t())                                     # (B*L, C_out)
        y = y_i32.float() * (s_x * self.scale_w)
        if self.bias is not None: y += self.bias
        y = y.to(torch.float16).view(B, L, -1).transpose(1,2)
        out_h = (H + 2*self.padding - self.dilation*(k-1) - 1)//self.stride + 1
        out_w = (W + 2*self.padding - self.dilation*(k-1) - 1)//self.stride + 1
        return F.fold(y, (out_h, out_w), 1)                 # 1×1 kernels after GEMM

###############################################################################
# loader
###############################################################################

def load_quant_unet(root: str, device="cuda"):
    w = load_file(os.path.join(root, "unet", "diffusion_pytorch_model.safetensors"))
    fp8 = {k: v for k,v in w.items() if not k.endswith(".scale")}
    scales = {k[:-6]: v.item() for k,v in w.items() if k.endswith(".scale")}
    unet = UNet2DConditionModel.from_config(os.path.join(root, "unet"))
    unet.eval()

    # walk modules; swap conv/linear
    for name, mod in list(unet.named_modules())[::-1]:
        if isinstance(mod, nn.Linear):
            wk = f"{name}.weight"
            bk = f"{name}.bias"
            w_int8 = fp8_to_int8(fp8[wk], scales[wk]).to(device)
            bias = fp8[bk].to(torch.float16).to(device) if bk in fp8 else None
            repl = Int8Linear(w_int8, scales[wk], bias)
            _assign(unet, name, repl)
        elif isinstance(mod, nn.Conv2d) and mod.kernel_size==(3,3):
            wk = f"{name}.weight"
            bk = f"{name}.bias"
            weight_int8 = fp8_to_int8(fp8[wk].flatten(1), scales[wk]).to(device)
            bias = fp8[bk].to(torch.float16).to(device) if bk in fp8 else None
            repl = Int8Conv2d(weight_int8, scales[wk], bias,
                              mod.stride[0], mod.padding[0], mod.dilation[0])
            _assign(unet, name, repl)
        else:
            # keep norms & others in FP16
            for p in mod.parameters(recurse=False):
                p.data = p.data.to(torch.float16).to(device)
    unet.to(device)
    return unet

def _assign(root: nn.Module, dotted: str, sub: nn.Module):
    parts = dotted.split(".")
    for p in parts[:-1]:
        root = getattr(root, p)
    setattr(root, parts[-1], sub)
