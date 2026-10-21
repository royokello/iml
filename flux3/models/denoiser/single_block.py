from __future__ import annotations

import torch
import torch.nn as nn

from flux2.models.denoiser.feed_forward import Flux2FeedForward

from .attention import Flux3Attention


class Flux3SingleTransformerBlock(nn.Module):
    """Joint block: text and content tokens share one sequence.

    The text tokens and the content tokens each have their own modulation.
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
        encoder_hidden_states: torch.Tensor,
        temb_mod: torch.Tensor,
        temb_mod_context: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mod_shift, mod_scale, mod_gate = torch.chunk(temb_mod, 3, dim=-1)
        c_mod_shift, c_mod_scale, c_mod_gate = torch.chunk(temb_mod_context, 3, dim=-1)

        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

        norm_encoder_hidden_states = self.norm(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_mod_scale) * norm_encoder_hidden_states + c_mod_shift

        combined = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)

        attn_output = self.attn(hidden_states=combined, image_rotary_emb=image_rotary_emb)
        ff_output = self.ff(combined)
        output = attn_output + ff_output

        context_output, content_output = torch.split(
            output, [encoder_hidden_states.shape[1], hidden_states.shape[1]], dim=1
        )
        encoder_hidden_states = encoder_hidden_states + c_mod_gate * context_output
        hidden_states = hidden_states + mod_gate * content_output

        return encoder_hidden_states, hidden_states
