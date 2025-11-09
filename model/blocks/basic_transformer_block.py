import torch
import torch.nn as nn

from model.blocks.attention_block import AttentionBlock
from model.blocks.feed_forward import FeedForward

# assuming AttentionBlock and FeedForward are already defined as we wrote before

class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,                    # token embedding dimension
        num_heads: int,              # number of attention heads
        head_dim: int | None = None, # per-head dimension (if None: dim // num_heads)
        cross_attention_dim: int | None = None,  # dim of text/context embeddings
        dropout: float = 0.0,
        ff_mult: float = 4.0,        # expansion factor in the feed-forward block
    ):
        super().__init__()

        self.dim = dim

        # self-attention (queries/keys/values all from hidden_states)
        self.attn1 = AttentionBlock(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            cross_attention_dim=None,  # self-attn: K/V dim = dim
            dropout=dropout,
        )

        # cross-attention (queries from hidden_states, K/V from encoder_hidden_states)
        self.attn2 = AttentionBlock(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            cross_attention_dim=cross_attention_dim or dim,
            dropout=dropout,
        )

        # feed-forward MLP
        self.ff = FeedForward(
            dim=dim,
            mult=ff_mult,
            dropout=dropout,
            use_gelu=True,
        )

        # three LayerNorms for pre-norm branches
        self.norm1 = nn.LayerNorm(dim)  # before self-attn
        self.norm2 = nn.LayerNorm(dim)  # before cross-attn
        self.norm3 = nn.LayerNorm(dim)  # before FFN

    def forward(
        self,
        hidden_states: torch.Tensor,                # [B, N, dim]
        encoder_hidden_states: torch.Tensor | None, # [B, M, cross_attention_dim]
        attention_mask: torch.Tensor | None = None, # optional mask over encoder tokens
    ) -> torch.Tensor:
        x = hidden_states

        # 1) self-attention: attend over spatial tokens themselves
        x_res = x
        x_norm = self.norm1(x)
        x_attn = self.attn1(
            hidden_states=x_norm,
            encoder_hidden_states=None,   # self-attn
            attention_mask=None,
        )
        x = x_res + x_attn

        # 2) cross-attention: attend to text/context if provided
        if encoder_hidden_states is not None:
            x_res = x
            x_norm = self.norm2(x)
            x_attn = self.attn2(
                hidden_states=x_norm,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )
            x = x_res + x_attn

        # 3) feed-forward network
        x_res = x
        x_norm = self.norm3(x)
        x_ff = self.ff(x_norm)
        x = x_res + x_ff

        return x
