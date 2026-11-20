from __future__ import annotations

import torch
import torch.nn as nn

from utils.quant.linear import QuantizedLinear
from utils.quant.qlinear import QuantizedLinearFrozenFn, _dequantize_quantized_linear_weight


class RecomputeQuantizedLinear(QuantizedLinear):
    """A `QuantizedLinear` that rebuilds its weight in `backward`.

    `utils.quant.linear.QuantizedLinear` dequantizes the weight into a tensor
    that `F.linear` saves for backward. That is an fp16 copy of every weight
    per forward pass, about 4.5 GB for the SDXL UNet.

    This class runs the same forward and, when the input needs a gradient,
    routes through `QuantizedLinearFrozenFn`. That function re-dequantizes the
    weight in `backward` instead of saving it.

    It subclasses `QuantizedLinear`, so every `isinstance` check in the repo
    still matches it.
    """

    def _normal_forward_body(
        self,
        input: torch.Tensor,
        bsums: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the plain quantized forward.

        Args:
            input: The input tensor.
            bsums: Unused. Accepted for signature parity with `qlinear`.

        Returns:
            The linear output.
        """
        return super().forward(input)

    def forward(
        self,
        input: torch.Tensor,
        bsums: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the linear, saving the input instead of the weight.

        Args:
            input: The input tensor.
            bsums: Unused. Accepted for signature parity with `qlinear`.

        Returns:
            The linear output.
        """
        if torch.is_grad_enabled() and input.requires_grad:
            return QuantizedLinearFrozenFn.apply(input, self)
        return super().forward(input)


class _StreamedQuantizedLinearFrozenFn(torch.autograd.Function):
    """Rebuild the weight for `backward`, pulling it onto the GPU first.

    `QuantizedLinearFrozenFn` reads the module buffers where `forward` left
    them. A streamed module leaves them in host memory, so `backward` pulls
    first and releases afterwards.
    """

    @staticmethod
    def forward(ctx, input, module):
        ctx.module = module
        ctx.input_dtype = input.dtype
        return module._normal_forward_body(input)

    @staticmethod
    def backward(ctx, grad_output):
        module = ctx.module
        module._pull(grad_output.device)
        weight = _dequantize_quantized_linear_weight(module).to(dtype=grad_output.dtype)
        grad_input = grad_output.matmul(weight)
        module._push()
        return grad_input, None


class StreamedQuantizedLinear(RecomputeQuantizedLinear):
    """A quantized linear that keeps its weights in host memory between calls.

    The packed weights stay pinned on the CPU. `forward` copies them to the
    input device, runs the multiply, and releases the GPU copy. `backward`
    pulls them again, because gradient checkpointing may have discarded the
    graph and because the release happens before the backward runs.

    The buffers are swapped, not reassigned through the module tree, so the
    host tensors remain the source of truth and `state_dict` still reads them.
    Only the frozen weights move. No parameter changes device.
    """

    def _ensure_host(self) -> None:
        """Capture the host tensors on first use."""
        if getattr(self, "_host_buffers", None) is None:
            self._host_buffers = dict(self._buffers)
            self._device_buffers = {}

    def _pull(self, device: torch.device) -> None:
        """Copy the weights to `device`.

        Args:
            device: The device that will run the multiply.
        """
        self._ensure_host()
        self._device_buffers = {}
        for name, host in self._host_buffers.items():
            if host is None:
                continue
            device_buffer = host.to(device=device, non_blocking=True)
            self._buffers[name] = device_buffer
            self._device_buffers[name] = device_buffer

    def _push(self) -> None:
        """Release the GPU copy and point the buffers back at the host."""
        self._ensure_host()
        self._device_buffers = {}
        for name, host in self._host_buffers.items():
            self._buffers[name] = host

    def forward(
        self,
        input: torch.Tensor,
        bsums: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the linear with the weight copied in for this call.

        Args:
            input: The input tensor.
            bsums: Unused. Accepted for signature parity with `qlinear`.

        Returns:
            The linear output.
        """
        self._pull(input.device)
        if torch.is_grad_enabled() and input.requires_grad:
            output = _StreamedQuantizedLinearFrozenFn.apply(input, self)
        else:
            output = super().forward(input, bsums)
        self._push()
        return output


def move_unstreamed_modules_to(model: torch.nn.Module, device: torch.device) -> None:
    """Move every module to `device` except the streamed quantized linears.

    `Module.to` would move the streamed weights as well, which defeats the
    streaming and usually runs out of memory. Parameters keep their identity,
    so an optimizer built afterwards holds the right tensors.

    Args:
        model: The UNet, with the LoRA wrappers already injected.
        device: The device that runs the compute.
    """
    for module in model.modules():
        if isinstance(module, StreamedQuantizedLinear):
            continue
        for _, parameter in module.named_parameters(recurse=False):
            parameter.data = parameter.data.to(device)
        for name, buffer in list(module._buffers.items()):
            if buffer is not None:
                module._buffers[name] = buffer.to(device)


__all__ = [
    "RecomputeQuantizedLinear",
    "StreamedQuantizedLinear",
    "move_unstreamed_modules_to",
]
