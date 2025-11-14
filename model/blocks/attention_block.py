import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks.linear import Linear
from model.utils.quantization import quantize_input_and_attach_scale

class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,                 # channel/embedding dim of the UNet features (queries)
        num_heads: int = 8,       # number of attention heads
        head_dim: int | None = None,  # per-head dim; if None, use dim // num_heads
        cross_attention_dim: int | None = None,  # dim of context (K,V) for cross-attn; if None, same as dim
        dropout: float = 0.0,     # dropout on attention probs / output proj
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        # compute head_dim and total inner dim for Q,K,V
        self.head_dim = head_dim or (dim // num_heads)
        self.inner_dim = self.head_dim * num_heads

        # if cross_attention_dim is None, K/V come from same dim as Q (self-attn case)
        self.cross_attention_dim = cross_attention_dim or dim

        # linear projections for queries, keys, values
        self.to_q = Linear(
            in_features=dim,
            out_features=self.inner_dim,
            bias=False,
        )
        self.to_k = Linear(
            in_features=self.cross_attention_dim,
            out_features=self.inner_dim,
            bias=False,
        )
        self.to_v = Linear(
            in_features=self.cross_attention_dim,
            out_features=self.inner_dim,
            bias=False,
        )

        # output projection after merging heads
        self.proj_out = Linear(
            in_features=self.inner_dim,
            out_features=dim,
            bias=True,
        )
        self.out_dropout = nn.Dropout(dropout)

        # scaling factor for dot-product attention
        self.scale = self.head_dim ** -0.5

    def _reshape_heads(self, x: torch.Tensor, batch_size: int):
        # x: [B, N, inner_dim] â†’ [B, num_heads, N, head_dim]
        return (
            x.view(batch_size, -1, self.num_heads, self.head_dim)
             .transpose(1, 2)   # [B, N, H, D] -> [B, H, N, D]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,             # [B, N, dim] queries (and keys/values in self-attn)
        encoder_hidden_states: torch.Tensor | None = None,  # [B, M, cross_attention_dim] for cross-attn
        attention_mask: torch.Tensor | None = None,         # optional mask over M
    ) -> torch.Tensor:
        """
        If encoder_hidden_states is None â†’ self-attention.
        Else â†’ cross-attention (Q from hidden_states, K/V from encoder_hidden_states).
        """

        bsz, q_len, _ = hidden_states.shape

        # for self-attention, attend over the same sequence
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        _, kv_len, _ = encoder_hidden_states.shape

        # project queries, keys, values
        tensor_q = quantize_input_and_attach_scale(self.to_q, hidden_states)
        q = self.to_q(tensor_q)  # [B, Nq, inner_dim]

        tensor_q = quantize_input_and_attach_scale(self.to_k, encoder_hidden_states)
        k = self.to_k(tensor_q)  # [B, Nk, inner_dim]

        tensor_q = quantize_input_and_attach_scale(self.to_v, encoder_hidden_states)
        v = self.to_v(tensor_q)  # [B, Nk, inner_dim]

        # split into heads
        q = self._reshape_heads(q, bsz)        # [B, H, Nq, D]
        k = self._reshape_heads(k, bsz)        # [B, H, Nk, D]
        v = self._reshape_heads(v, bsz)        # [B, H, Nk, D]

        # scaled dot-product attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, Nq, Nk]

        # apply attention mask if provided
        if attention_mask is not None:
            # allow masks of shape [B, Nk] or [B, 1, 1, Nk]
            if attention_mask.dim() == 2:
                # [B, Nk] -> [B, 1, 1, Nk]
                attention_mask = attention_mask[:, None, None, :]
            # masked positions should get large negative value so softmax ~ 0 there
            attn_scores = attn_scores + (attention_mask * torch.finfo(attn_scores.dtype).min)

        # softmax to get attention weights
        attn_probs = F.softmax(attn_scores, dim=-1)  # [B, H, Nq, Nk]

        # weighted sum of values
        attn_output = torch.matmul(attn_probs, v)    # [B, H, Nq, D]

        # merge heads back: [B, H, N, D] -> [B, N, H*D]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, N, H, D]
        attn_output = attn_output.view(bsz, q_len, self.inner_dim)

        # final linear projection back to dim
        tensor_q = quantize_input_and_attach_scale(self.proj_out, attn_output)
        attn_output = self.proj_out(tensor_q)  # [B, N, dim]
        attn_output = self.out_dropout(attn_output)

        return attn_output
