import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.linear import QuantLinear


def _linear_fp16(in_features: int, out_features: int, bias: bool = True) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, bias=bias)
    return layer.to(torch.float16)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int | None = None,
        cross_attention_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        self.head_dim = head_dim or (dim // num_heads)
        self.inner_dim = self.head_dim * num_heads

        self.cross_attention_dim = cross_attention_dim or dim

        self.to_q = _linear_fp16(
            in_features=dim,
            out_features=self.inner_dim,
            bias=False,
        )
        self.to_k = _linear_fp16(
            in_features=self.cross_attention_dim,
            out_features=self.inner_dim,
            bias=False,
        )
        self.to_v = QuantLinear(
            in_features=self.cross_attention_dim,
            out_features=self.inner_dim,
            bias=False,
        )

        self.to_out = nn.Sequential(
            QuantLinear(
                in_features=self.inner_dim,
                out_features=dim,
                bias=True,
            ),
            nn.Dropout(dropout),
        )

        self.scale = self.head_dim ** -0.5

    def _reshape_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        bsz, q_len, _ = hidden_states.shape
        _, kv_len, _ = encoder_hidden_states.shape

        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        q = self._reshape_heads(q, bsz)
        k = self._reshape_heads(k, bsz)
        v = self._reshape_heads(v, bsz)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            attn_scores = attn_scores + (
                attention_mask * torch.finfo(attn_scores.dtype).min
            )

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.inner_dim)

        attn_output = self.to_out(attn_output)

        return attn_output
