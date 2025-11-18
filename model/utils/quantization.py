import torch



def quantize_to_int8(tensor: torch.Tensor, channel_dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute live per-channel fp16 scales from activations and
    return (int8_quantized, fp16_scale_vector).

    This matches the contract: activations are quantized to int8 and the
    per-channel fp16 scale vector is provided to the consuming op via
    set_input_scale.
    """
    x = tensor.to(torch.float32)
    if channel_dim < 0:
        channel_dim += x.ndim
    assert 0 <= channel_dim < x.ndim, "channel_dim out of range"
    reduce_dims = [d for d in range(x.ndim) if d != channel_dim]
    amax = x.abs().amax(dim=reduce_dims)
    scale_fp32 = (amax / 127.0).clamp_min(1e-8)
    view_shape = [1] * x.ndim
    view_shape[channel_dim] = scale_fp32.shape[0]
    scale_broadcast = scale_fp32.view(*view_shape)
    x_q = torch.clamp(torch.round(x / scale_broadcast), -128, 127).to(torch.int8)
    scale_fp16 = scale_fp32.to(torch.float16)
    return x_q, scale_fp16


def quantize_input_and_attach_scale(
    module, tensor: torch.Tensor, channel_dim: int
) -> torch.Tensor:
    """Quantize activations and set the module's input scale."""
    tensor_q, scale = quantize_to_int8(tensor, channel_dim=channel_dim)
    module.set_input_scale(scale)
    return tensor_q
