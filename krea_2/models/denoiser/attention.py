from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rms_norm import Krea2RMSNorm


def apply_rotary_emb(x: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Apply interleaved rotary embeddings to a query or key tensor.

    Args:
        x: The tensor with shape `(B, S, H, D)`.
        rotary_emb: The cosine and sine tensors from `Flux2PosEmbed`, each with
            shape `(S, D)`.

    Returns:
        The rotated tensor with the shape and dtype of `x`.
    """
    cos, sin = rotary_emb
    cos = cos[None, :, None, :].to(x.device)
    sin = sin[None, :, None, :].to(x.device)
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)
    return (x.float() * cos + x_rotated * sin).to(x.dtype)


class Krea2Attention(nn.Module):
    """Self-attention with grouped-query projections, q/k norm, RoPE and a gate."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads

        self.to_q = nn.Linear(hidden_size, self.head_dim * self.num_heads, bias=False)
        self.to_k = nn.Linear(hidden_size, self.head_dim * self.num_kv_heads, bias=False)
        self.to_v = nn.Linear(hidden_size, self.head_dim * self.num_kv_heads, bias=False)
        self.to_gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_q = Krea2RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Krea2RMSNorm(self.head_dim, eps=eps)
        self.to_out = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False), nn.Dropout(0.0)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        query = self.to_q(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        key = self.to_k(hidden_states).unflatten(-1, (self.num_kv_heads, self.head_dim))
        value = self.to_v(hidden_states).unflatten(-1, (self.num_kv_heads, self.head_dim))
        gate = self.to_gate(hidden_states)

        query = self.norm_q(query)
        key = self.norm_k(key)

        if rotary_emb is not None:
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        num_key_value_groups = self.num_heads // self.num_kv_heads
        if num_key_value_groups > 1:
            key = key.repeat_interleave(num_key_value_groups, dim=1)
            value = value.repeat_interleave(num_key_value_groups, dim=1)

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states * torch.sigmoid(gate)
        return self.to_out[0](hidden_states)


__all__ = ["Krea2Attention", "apply_rotary_emb"]
