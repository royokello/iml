import torch


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
