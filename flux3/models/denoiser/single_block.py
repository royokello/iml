from __future__ import annotations

import torch
import torch.nn as nn

from flux2.models.denoiser.feed_forward import Flux2FeedForward

from .attention import Flux3Attention
from .precision import linear_fp32


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

        # The text stream can be fp32. Its norm, modulation, residual add, and feed-forward
        # output projection run in fp32. The attention and the other layers run in the dtype
        # of the modulation.
        compute_dtype = temb_mod.dtype
        text_dtype = encoder_hidden_states.dtype

        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

        norm_encoder_hidden_states = self.norm(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_mod_scale.to(text_dtype)) * norm_encoder_hidden_states + c_mod_shift.to(
            text_dtype
        )
        norm_encoder_hidden_states = norm_encoder_hidden_states.to(compute_dtype)

        combined = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)

        attn_output = self.attn(hidden_states=combined, image_rotary_emb=image_rotary_emb)
        ff_hidden = self.ff.act_fn(self.ff.linear_in(combined))

        text_length = encoder_hidden_states.shape[1]
        attn_context, attn_content = torch.split(attn_output, [text_length, hidden_states.shape[1]], dim=1)
        ff_context, ff_content = torch.split(ff_hidden, [text_length, hidden_states.shape[1]], dim=1)
        if text_dtype == torch.float32:
            # The fp16 output projection of the text tokens overflows.
            context_output = attn_context.float() + linear_fp32(self.ff.linear_out, ff_context)
        else:
            context_output = attn_context + self.ff.linear_out(ff_context)
        content_output = attn_content + self.ff.linear_out(ff_content)
        encoder_hidden_states = encoder_hidden_states + c_mod_gate.to(text_dtype) * context_output.to(text_dtype)
        hidden_states = hidden_states + mod_gate * content_output

        return encoder_hidden_states, hidden_states
