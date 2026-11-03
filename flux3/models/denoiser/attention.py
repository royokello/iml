from __future__ import annotations

import torch
import torch.nn.functional as F

from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb

from flux3.models.denoiser.precision import rms_norm_fp32


def _get_projections(attn: "Flux3Attention", hidden_states: torch.Tensor):
    return attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)


def _get_fused_projections(attn: "Flux3Attention", hidden_states: torch.Tensor):
    return attn.to_qkv(hidden_states).chunk(3, dim=-1)


def _get_qkv_projections(attn: "Flux3Attention", hidden_states: torch.Tensor):
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states)
    return _get_projections(attn, hidden_states)


class Flux3AttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")

    def __call__(
        self,
        attn: "Flux3Attention",
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        query, key, value = _get_qkv_projections(attn, hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # QK norm and RoPE run in fp32 and are cast back once. The norm weights are bf16.
        query = rms_norm_fp32(attn.norm_q, query)
        key = rms_norm_fp32(attn.norm_k, key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        query, key = query.to(value.dtype), key.to(value.dtype)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class Flux3Attention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = Flux3AttnProcessor
    _available_processors = [Flux3AttnProcessor]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        eps: float = 1e-6,
        processor=None,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.query_dim = query_dim
        self.heads = heads
        self.dropout = dropout
        self.use_bias = False

        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = torch.nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_v = torch.nn.Linear(query_dim, self.inner_dim, bias=False)

        self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=True)

        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(torch.nn.Linear(self.inner_dim, query_dim, bias=False))
        self.to_out.append(torch.nn.Dropout(dropout))

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, attention_mask, image_rotary_emb)
