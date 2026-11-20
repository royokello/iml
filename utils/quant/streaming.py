"""Stream quantized weights from pinned host memory during training.

A model that streams keeps the packed weights on the CPU and copies one layer's
weights onto the GPU for its own multiply. The LoRA parameters never move, so
the optimizer is unaffected and gradient checkpointing can stay on.

The frozen backward matters as much as the copy. A plain `QuantizedLinear`
dequantizes inside `F.linear`, which makes autograd save the fp16 weight for the
backward. `StreamedQuantizedLinear` dequantizes again in `backward` instead, so
the fp16 copy is never held.
"""

from __future__ import annotations

import torch

from utils.quant.linear import QuantizedLinear
from utils.quant.qlinear import _dequantize_quantized_linear_weight


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
        return QuantizedLinear.forward(module, input)

    @staticmethod
    def backward(ctx, grad_output):
        module = ctx.module
        module._pull(grad_output.device)
        weight = _dequantize_quantized_linear_weight(module).to(dtype=grad_output.dtype)
        grad_input = grad_output.matmul(weight)
        module._push()
        return grad_input, None


class StreamedQuantizedLinear(QuantizedLinear):
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the linear with the weight copied in for this call.

        Args:
            input: The input tensor.

        Returns:
            The linear output.
        """
        self._pull(input.device)
        if torch.is_grad_enabled() and input.requires_grad:
            output = _StreamedQuantizedLinearFrozenFn.apply(input, self)
        else:
            output = QuantizedLinear.forward(self, input)
        self._push()
        return output


def pin_quantized_buffers(model: torch.nn.Module) -> None:
    """Move the quantized weights into pinned host memory.

    A streamed layer copies its weights to the GPU on every forward and
    backward. Pinned host memory makes that copy asynchronous; pageable memory
    makes it synchronous and about three times slower.

    Args:
        model: The model, already loaded and on the CPU.

    Raises:
        RuntimeError: If the host cannot pin the weights.
    """
    tensor_count = 0
    total_bytes = 0
    try:
        for module in model.modules():
            if not isinstance(module, QuantizedLinear):
                continue
            for name, buffer in list(module._buffers.items()):
                if buffer is None or buffer.device.type != "cpu" or buffer.is_pinned():
                    continue
                pinned = buffer.pin_memory()
                module._buffers[name] = pinned
                tensor_count += 1
                total_bytes += pinned.numel() * pinned.element_size()
    except RuntimeError as error:
        raise RuntimeError(
            "Could not pin the quantized weights in host memory. "
            f"Pinned {total_bytes / 2**30:.2f} GiB before the failure. "
            "Free host memory or run without --stream-denoiser."
        ) from error
    print(
        f"    pinned {tensor_count} quantized tensors, {total_bytes / 2**30:.2f} GiB",
        flush=True,
    )


def enable_streaming(model: torch.nn.Module) -> int:
    """Turn a quantized model that sits on the CPU into a streamed one.

    A model loaded without the stream flag stores its packed weights in
    `QuantizedLinear` buffers. Those buffers live on the CPU already, so the
    model can start streaming without a reload: only the class changes. The
    buffers keep their identity, so no second copy is made.

    Args:
        model: A quantized model the caller already holds.

    Returns:
        The number of layers converted. Zero means the model was already
        streamed.

    Raises:
        RuntimeError: If a packed weight is not on the CPU. A resident model
            has already lost its host copy, so streaming it would stream from
            the GPU and save nothing.
    """
    converted = 0
    for module in model.modules():
        if type(module) is not QuantizedLinear:
            continue
        for buffer in module._buffers.values():
            if buffer is not None and buffer.device.type != "cpu":
                raise RuntimeError(
                    f"Cannot stream weights that are on {buffer.device}. "
                    "Load the model with the stream flag, or run it resident."
                )
        module.__class__ = StreamedQuantizedLinear
        converted += 1
    return converted


def has_streamed_layers(model: torch.nn.Module) -> bool:
    """Return True when the model keeps any quantized weights in host memory.

    Args:
        model: The model to inspect.

    Returns:
        True if at least one layer streams its weights.
    """
    return any(isinstance(module, StreamedQuantizedLinear) for module in model.modules())


def move_unstreamed_modules_to(model: torch.nn.Module, device: torch.device) -> None:
    """Move every module to `device` except the streamed quantized linears.

    `Module.to` would move the streamed weights as well, which defeats the
    streaming and usually runs out of memory. Parameters keep their identity,
    so an optimizer built afterwards holds the right tensors.

    Args:
        model: The model, with the LoRA wrappers already injected.
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
    "StreamedQuantizedLinear",
    "enable_streaming",
    "has_streamed_layers",
    "move_unstreamed_modules_to",
    "pin_quantized_buffers",
]
