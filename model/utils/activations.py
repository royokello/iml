import torch

from model.blocks.quantization import quantize_activation_to_int8


def quantize_input_and_attach_scale(module, tensor: torch.Tensor) -> torch.Tensor:
    """Quantize activations and set the module's input scale."""
    tensor_q, scale = quantize_activation_to_int8(tensor)
    module.set_input_scale(scale)
    return tensor_q
