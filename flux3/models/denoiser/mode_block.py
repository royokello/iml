from __future__ import annotations

import torch
import torch.nn as nn

from flux2.models.denoiser.feed_forward import Flux2FeedForward

from .attention import Flux3Attention


class Flux3ModeTransformerBlock(nn.Module):
    """Attention and feed-forward in parallel, inside one token mode.

    Each mode (image, image_cond, text) attends to itself only.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Flux3Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            eps=eps,
        )
        self.ff = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb_mod: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # The split is here, not in the modulation, so that no tuple enters a checkpoint.
        mod_shift, mod_scale, mod_gate = torch.chunk(temb_mod, 3, dim=-1)

        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

        attn_output = self.attn(hidden_states=norm_hidden_states, image_rotary_emb=image_rotary_emb)
        ff_output = self.ff(norm_hidden_states)

        return hidden_states + mod_gate * (attn_output + ff_output)
