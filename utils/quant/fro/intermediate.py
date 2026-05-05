import torch


def dequantize_from_intermediate(
    qweight: torch.Tensor,
    super_scales: torch.Tensor,
    original_shape: tuple[int, ...] | torch.Size,
) -> torch.Tensor:
    """
    Dequantize Q8_K-style intermediate weights to float.

    Format:
        weights: int8 (signed)
        super scales: fp32 d
        blocks: 256 weights per super-block

    Expected shapes:
        qweight:
            [num_super_blocks, 256]

        super_scales:
            [num_super_blocks]

    Reconstruction:
        q ∈ [-128, 127]

        weight = q * super_scale

    Notes:
        - This is symmetric quantization (no min/offset term).
        - `bsums` from quantization is NOT required for dequantization.
        - `bsums` is only used in fused matmul kernels to handle affine corrections
          when interacting with lower-bit formats (e.g., Q4_K/Q5_K).

    Behavior:
        - Expands super scales across each 256-weight block
        - Applies symmetric reconstruction
        - Removes padding introduced during quantization
        - Reshapes to original_shape

    Args:
        qweight: int8 quantized weights
        super_scales: fp32 super scale factors
        original_shape: original tensor shape before quantization

    Returns:
        Dequantized float16 tensor with original_shape
    """
    if qweight.dtype != torch.int8:
        raise TypeError("qweight must be int8.")
    if not torch.is_floating_point(super_scales):
        raise TypeError("super_scales must be floating point.")

    original_shape = tuple(original_shape)
    original_numel = 1
    for dim in original_shape:
        original_numel *= int(dim)

    if original_numel == 0:
        return torch.empty(original_shape, dtype=torch.float16, device=qweight.device)

    if qweight.ndim != 2:
        raise ValueError("qweight must have shape [num_super_blocks, super_block_size].")

    num_super_blocks, super_block_size = qweight.shape
    padded_numel = int(num_super_blocks) * int(super_block_size)

    if original_numel > padded_numel:
        raise ValueError(
            "original_shape contains more values than the quantized blocks: "
            f"expected at most {padded_numel}, got {original_numel}."
        )
    if super_scales.numel() != num_super_blocks:
        raise ValueError("super_scales must contain one value per super-block.")

    dequantized = qweight.to(torch.float32) * super_scales.to(torch.float32).view(-1, 1)
    flattened = dequantized.reshape(-1)[:original_numel].to(torch.float16)
    return flattened.view(original_shape)
