import torch

# from utils.quant.cuda.symmetric_high.dequant import dequantize_from_symmetric_high
# from utils.quant.to.symmetric import quantize_to_symmetric


class Flux2KVLayerCache:
    """Per-layer KV cache for reference image tokens in the Flux2 Klein KV model.

    Stores the K and V projections (post-RoPE) for reference tokens extracted during the first denoising step. Tensor
    format: (batch_size, num_ref_tokens, num_heads, head_dim).
    """

    def __init__(self):
        self.k_ref: torch.Tensor | None = None
        self.v_ref: torch.Tensor | None = None

    def store(self, k_ref: torch.Tensor, v_ref: torch.Tensor):
        """Store reference token K/V."""
        self.k_ref = k_ref
        self.v_ref = v_ref

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve cached reference token K/V."""
        if self.k_ref is None:
            raise RuntimeError("KV cache has not been populated yet.")
        return self.k_ref, self.v_ref

    def clear(self):
        self.k_ref = None
        self.v_ref = None


# class Flux2KVLayerCache:
#     def __init__(self):
#         self.k_ref_q: torch.Tensor | None = None
#         self.k_ref_scale: torch.Tensor | None = None
#         self.k_shape: tuple[int, ...] | None = None
#         self.v_ref: torch.Tensor | None = None

#     def store(self, k_ref: torch.Tensor, v_ref: torch.Tensor):
#         self.k_shape = tuple(k_ref.shape)
#         self.k_ref_q, self.k_ref_scale, _ = quantize_to_symmetric(k_ref, mode="high")
#         self.v_ref = v_ref

#     def get(self) -> tuple[torch.Tensor, torch.Tensor]:
#         if self.k_ref_q is None or self.k_ref_scale is None or self.k_shape is None or self.v_ref is None:
#             raise RuntimeError("KV cache has not been populated yet.")

#         k_ref = dequantize_from_symmetric_high(
#             self.k_ref_q,
#             self.k_ref_scale,
#             original_shape=self.k_shape,
#         ).to(dtype=self.v_ref.dtype)

#         return k_ref, self.v_ref

#     def clear(self):
#         self.k_ref_q = None
#         self.k_ref_scale = None
#         self.k_shape = None
#         self.v_ref = None