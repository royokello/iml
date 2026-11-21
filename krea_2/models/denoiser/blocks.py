from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Krea2Attention
from .rms_norm import Krea2RMSNorm


class Krea2TimestepEmbedding(nn.Module):
    """Cosine-first sinusoidal embedding of the flow time, then a two-layer MLP.

    The sequence axis stays at size 1, so the per-block modulations broadcast
    over tokens.
    """

    def __init__(self, embed_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_1 = nn.Linear(embed_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(1e4) * torch.arange(half, dtype=torch.float32, device=timestep.device) / half
        )
        args = (timestep.float() * 1e3)[:, None, None] * freqs
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)
        return self.linear_2(F.gelu(self.linear_1(emb), approximate="tanh"))


class Krea2SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(hidden_states)) * self.up(hidden_states))


class Krea2TextFusionBlock(nn.Module):
    """Pre-norm block without time modulation, used by the text fusion stage."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = Krea2RMSNorm(dim, eps=eps)
        self.norm2 = Krea2RMSNorm(dim, eps=eps)
        self.attn = Krea2Attention(dim, num_heads, num_kv_heads, eps=eps)
        self.ff = Krea2SwiGLU(dim, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), attention_mask=attention_mask)
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class Krea2TextFusion(nn.Module):
    """Fuse the tapped text hidden states into one sequence of text features."""

    def __init__(
        self,
        num_text_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        num_layerwise_blocks: int,
        num_refiner_blocks: int,
        eps: float,
    ) -> None:
        super().__init__()
        self.layerwise_blocks = nn.ModuleList(
            [
                Krea2TextFusionBlock(dim, num_heads, num_kv_heads, intermediate_size, eps)
                for _ in range(num_layerwise_blocks)
            ]
        )
        self.projector = nn.Linear(num_text_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList(
            [
                Krea2TextFusionBlock(dim, num_heads, num_kv_heads, intermediate_size, eps)
                for _ in range(num_refiner_blocks)
            ]
        )

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, num_text_layers, dim = encoder_hidden_states.shape
        hidden_states = encoder_hidden_states.reshape(batch_size * seq_len, num_text_layers, dim)
        for block in self.layerwise_blocks:
            hidden_states = block(hidden_states.contiguous())

        hidden_states = hidden_states.reshape(batch_size, seq_len, num_text_layers, dim).permute(0, 1, 3, 2)
        hidden_states = self.projector(hidden_states).squeeze(-1)

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)
        return hidden_states


class Krea2TransformerBlock(nn.Module):
    """One single-stream MMDiT block with shared time modulation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(6, hidden_size))
        self.norm1 = Krea2RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = Krea2RMSNorm(hidden_size, eps=norm_eps)
        self.attn = Krea2Attention(hidden_size, num_heads, num_kv_heads, eps=norm_eps)
        self.ff = Krea2SwiGLU(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        modulation = temb.unflatten(-1, (6, -1)) + self.scale_shift_table
        prescale, preshift, pregate, postscale, postshift, postgate = modulation.unbind(-2)

        attn_out = self.attn(
            (1.0 + prescale) * self.norm1(hidden_states) + preshift,
            attention_mask=attention_mask,
            rotary_emb=rotary_emb,
        )
        hidden_states = hidden_states + pregate * attn_out
        ff_out = self.ff((1.0 + postscale) * self.norm2(hidden_states) + postshift)
        hidden_states = hidden_states + postgate * ff_out
        return hidden_states


class Krea2TextProjection(nn.Module):
    """Project the fused text features into the transformer width."""

    def __init__(self, text_dim: int, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.norm = Krea2RMSNorm(text_dim, eps=eps)
        self.linear_1 = nn.Linear(text_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(self.norm(hidden_states))
        return self.linear_2(F.gelu(hidden_states, approximate="tanh"))


class Krea2FinalLayer(nn.Module):
    """Final adaptive RMSNorm and output projection."""

    def __init__(self, hidden_size: int, out_channels: int, eps: float) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(2, hidden_size))
        self.norm = Krea2RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        modulation = temb + self.scale_shift_table
        scale, shift = modulation.chunk(2, dim=1)
        hidden_states = (1.0 + scale) * self.norm(hidden_states) + shift
        return self.linear(hidden_states)


__all__ = [
    "Krea2FinalLayer",
    "Krea2SwiGLU",
    "Krea2TextFusion",
    "Krea2TextFusionBlock",
    "Krea2TextProjection",
    "Krea2TimestepEmbedding",
    "Krea2TransformerBlock",
]
