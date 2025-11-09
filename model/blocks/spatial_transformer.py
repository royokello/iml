import torch
import torch.nn as nn

from model.blocks.basic_transformer_block import BasicTransformerBlock

class SpatialTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,             # channels coming from the UNet (C)
        num_heads: int,               # number of attention heads
        head_dim: int,                # per-head dim; total dim = num_heads * head_dim
        depth: int = 1,               # how many BasicTransformerBlocks to stack
        cross_attention_dim: int | None = None,  # text/context embedding dim
        num_groups: int = 32,         # for GroupNorm over channels
    ):
        super().__init__()

        self.in_channels = in_channels
        self.inner_dim = num_heads * head_dim   # transformer token dim

        # normalize across channels before projecting into transformer space
        self.norm = nn.GroupNorm(num_groups, in_channels, eps=1e-5, affine=True)

        # 1x1 conv to map C → inner_dim (acts like per-pixel Linear)
        self.proj_in = nn.Conv2d(in_channels, self.inner_dim, kernel_size=1)

        # stack of BasicTransformerBlocks operating on tokens
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=self.inner_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for _ in range(depth)
            ]
        )

        # 1x1 conv to map inner_dim → C to go back into UNet
        self.proj_out = nn.Conv2d(self.inner_dim, in_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,                     # [B, C, H, W] feature map from UNet
        encoder_hidden_states: torch.Tensor, # [B, T, cross_attention_dim] text/context
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, c, h, w = x.shape

        # keep original for residual connection
        residual = x

        # 1) norm over channels to stabilize before transformer
        x = self.norm(x)                # [B, C, H, W]

        # 2) project channels C → inner_dim (token embedding dim)
        x = self.proj_in(x)             # [B, inner_dim, H, W]

        # 3) reshape spatial map into sequence of tokens
        x = x.view(b, self.inner_dim, h * w)    # [B, inner_dim, H*W]
        x = x.transpose(1, 2)                   # [B, H*W, inner_dim] = [B, N, dim]

        # 4) run through stacked BasicTransformerBlocks
        for block in self.transformer_blocks:
            x = block(
                hidden_states=x,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )   # [B, N, inner_dim]

        # 5) reshape tokens back to spatial map
        x = x.transpose(1, 2)                   # [B, inner_dim, N]
        x = x.view(b, self.inner_dim, h, w)     # [B, inner_dim, H, W]

        # 6) project inner_dim → C to match UNet channels
        x = self.proj_out(x)                    # [B, C, H, W]

        # 7) residual add: spatial_attn_out + original UNet features
        x = x + residual

        return x
