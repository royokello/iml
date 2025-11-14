import torch



def quantize_activation_to_int8(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute a live per-tensor fp16 scale from the fp16 activations and
    return (int8_quantized, fp16_scale).

    This matches the contract: activations are quantized to int8 and the
    fp16 scale is provided to the consuming op via set_input_scale.
    """
    # Compute scale in fp32 for numerical stability, then cast to fp16.
    amax = tensor.abs().amax()
    scale_fp32 = (amax / 127.0).clamp_min(1e-8).to(dtype=torch.float32)
    x_q = torch.clamp(torch.round(tensor.float() / scale_fp32), -128, 127).to(torch.int8)
    scale_fp16 = scale_fp32.to(dtype=torch.float16)
    return x_q, scale_fp16

