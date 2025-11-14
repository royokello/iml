import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load


_INT8_LINEAR_MODULE = None


def _load_int8_linear_extension():
    global _INT8_LINEAR_MODULE
    if _INT8_LINEAR_MODULE is not None:
        return _INT8_LINEAR_MODULE

    repo_root = Path(__file__).resolve().parents[2]
    sources = [
        repo_root / "cuda" / "int8_linear.cu",
        repo_root / "cuda" / "int8_linear_bindings.cpp",
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

    _INT8_LINEAR_MODULE = load(
        name="int8_linear_ops",
        sources=[str(src) for src in sources],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        extra_ldflags=extra_ldflags,
    )
    return _INT8_LINEAR_MODULE


class Linear(nn.Module):
    """
    The layer expects int8 activations as input. Activations and weights are
    scaled back to floating point inside the forward pass using the stored
    scale factors (set via `set_input_scale` and `load_quantized_weights`).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight_q",
            torch.zeros(out_features, in_features, dtype=torch.int8),
        )
        self.register_buffer("scale_w", torch.ones((), dtype=torch.float16))
        self.register_buffer("scale_x", torch.ones((), dtype=torch.float16))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def load_quantized_weights(
        self,
        w_q: torch.Tensor,
        scale_w: torch.Tensor,
    ) -> None:
        """
        Load pre-quantized int8 weights and their (scalar) fp16 scale.

        Args:
            w_q: int8 tensor with shape [out_features, in_features]
            scale_w: scalar tensor with dtype float16/float32
        """
        if w_q.shape != self.weight_q.shape:
            raise ValueError(
                f"Expected weight shape {tuple(self.weight_q.shape)}, got {tuple(w_q.shape)}"
            )
        if w_q.dtype is not torch.int8:
            raise TypeError("Quantized weights must be int8")
        if scale_w.numel() != 1:
            raise ValueError("scale_w must be a scalar tensor")

        self.weight_q.copy_(w_q)
        self.scale_w.copy_(scale_w.to(torch.float16))

    def set_input_scale(self, scale_x: torch.Tensor) -> None:
        """
        Persist the activation scale produced by the previous quantized op.
        """
        if scale_x.numel() != 1:
            raise ValueError("scale_x must be a scalar tensor")
        self.scale_x.copy_(scale_x.to(torch.float16))

    def forward(self, x_q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_q: int8 activations shaped [..., in_features]

        Returns:
            fp16 tensor shaped [..., out_features]
        """
        if x_q.dtype is not torch.int8:
            raise TypeError("Linear expects int8 inputs")
        if x_q.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, got {x_q.shape[-1]}"
            )

        if not x_q.is_cuda:
            raise RuntimeError("INT8 Linear requires CUDA tensors")
        if self.weight_q.device != x_q.device:
            raise RuntimeError("Call .to(device) on the module before running the forward pass")

        module = _load_int8_linear_extension()

        bias = self.bias
        if bias is None:
            bias = torch.zeros(self.out_features, dtype=torch.float16, device=x_q.device)

        scale = float(self.scale_x.item() * self.scale_w.item())

        y_fp32 = module.int8_linear(
            x_q.contiguous(),
            self.weight_q.contiguous(),
            bias.contiguous(),
            scale,
            True,
        )

        return y_fp32.to(dtype=torch.float16)
