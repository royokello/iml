from typing import Any

import torch
import torch.nn as nn

from ..attention.parallel_self import Flux2ParallelSelfAttention
from ..attention.parallel_self_processor import Flux2ParallelSelfAttnProcessor
from ..modulation import Flux2Modulation


class Flux2SingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        # Note that the MLP in/out linear layers are fused with the attention QKV/out projections, respectively; this
        # is often called a "parallel" transformer block. See the [ViT-22B paper](https://arxiv.org/abs/2302.05442)
        # for a visual depiction of this type of transformer block.
        self.attn = Flux2ParallelSelfAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            out_bias=bias,
            eps=eps,
            mlp_ratio=mlp_ratio,
            mlp_mult_factor=2,
            processor=Flux2ParallelSelfAttnProcessor(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        temb_mod: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        split_hidden_states: bool = False,
        text_seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # If encoder_hidden_states is None, hidden_states is assumed to have encoder_hidden_states already
        # concatenated
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        mod_shift, mod_scale, mod_gate = Flux2Modulation.split(temb_mod, 1)[0]

        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = hidden_states + mod_gate * attn_output
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        if split_hidden_states:
            encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
            return encoder_hidden_states, hidden_states
        else:
            return hidden_states
