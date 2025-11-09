import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks.cross_attn_down_block_2d import CrossAttnDownBlock2D
from model.blocks.cross_attn_up_block_2d import CrossAttnUpBlock2D
from model.blocks.down_block_2d import DownBlock2D
from model.blocks.unet_mid_block_2d_cross_attn import UNetMidBlock2DCrossAttn
from model.blocks.up_block_2d import UpBlock2D


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() == 0:
            timesteps = timesteps[None]
        emb = timestep_embedding(timesteps, self.dim)
        return self.mlp(emb)


class SDXLUNet(nn.Module):
    def __init__(self, model: dict[str, torch.Tensor] | None = None):
        super().__init__()

        # hardcoded SDXL-ish core params (simplified, but fixed)
        self.in_channels = 4
        self.out_channels = 4
        self.base_channels = 320
        self.time_embed_dim = 1280
        self.cross_attention_dim = 2048
        self.num_heads = 8
        self.head_dim = 64
        self.num_res_blocks = 2

        # 1) input conv
        self.conv_in = nn.Conv2d(self.in_channels, self.base_channels, 3, 1, 1)

        # 2) time embedding
        self.time_embed = TimeEmbedding(self.time_embed_dim)

        # 3) down blocks: one plain, one cross-attn
        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(
            DownBlock2D(
                in_channels=self.base_channels,
                out_channels=self.base_channels,
                temb_channels=self.time_embed_dim,
                num_layers=self.num_res_blocks,
                add_downsample=True,
                use_conv_down=True,
            )
        )
        self.down_blocks.append(
            CrossAttnDownBlock2D(
                in_channels=self.base_channels,
                out_channels=self.base_channels,
                temb_channels=self.time_embed_dim,
                num_layers=self.num_res_blocks,
                num_attention_heads=self.num_heads,
                head_dim=self.head_dim,
                cross_attention_dim=self.cross_attention_dim,
                add_downsample=False,
                num_groups=32,
                transformer_depth=1,
                use_conv_down=True,
            )
        )

        # 4) mid block
        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=self.base_channels,
            out_channels=self.base_channels,
            temb_channels=self.time_embed_dim,
            num_attention_heads=self.num_heads,
            head_dim=self.head_dim,
            cross_attention_dim=self.cross_attention_dim,
            num_layers=1,
            num_groups=32,
        )

        # 5) up blocks: mirror
        self.up_blocks = nn.ModuleList()
        self.up_blocks.append(
            CrossAttnUpBlock2D(
                in_channels=self.base_channels,
                out_channels=self.base_channels,
                temb_channels=self.time_embed_dim,
                num_layers=self.num_res_blocks,
                num_attention_heads=self.num_heads,
                head_dim=self.head_dim,
                cross_attention_dim=self.cross_attention_dim,
                add_upsample=True,
                num_groups=32,
                transformer_depth=1,
                use_conv_up=True,
            )
        )
        self.up_blocks.append(
            UpBlock2D(
                in_channels=self.base_channels,
                out_channels=self.base_channels,
                temb_channels=self.time_embed_dim,
                num_layers=self.num_res_blocks,
                add_upsample=False,
                num_groups=32,
                use_conv_up=True,
            )
        )

        # 6) output head
        self.conv_norm_out = nn.GroupNorm(32, self.base_channels, eps=1e-5, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(self.base_channels, self.out_channels, 3, 1, 1)

        # 7) optional filtered weight loading from safetensors/state_dict
        if model is not None:
            self._load_filtered_weights(model)

    def _load_filtered_weights(self, model: dict[str, torch.Tensor]) -> None:
        """
        Filter-load only keys that exist in this UNet and match in shape.
        `model` is expected to be a dict like from safetensors or state_dict().
        """
        own = self.state_dict()
        filtered = {}
        for k, v in model.items():
            if k in own and own[k].shape == v.shape:
                filtered[k] = v

        own.update(filtered)
        self.load_state_dict(own, strict=False)

    def forward(
        self,
        sample: torch.Tensor,                 # [B, 4, H, W]
        timesteps: torch.Tensor,              # [B] or scalar
        encoder_hidden_states: torch.Tensor,  # [B, T, 2048]
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.conv_in(sample)
        temb = self.time_embed(timesteps)

        res_hidden_states: list[torch.Tensor] = []

        # down path
        for block in self.down_blocks:
            if isinstance(block, CrossAttnDownBlock2D):
                x, res = block(
                    x,
                    temb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
            else:
                x, res = block(x, temb)
            res_hidden_states.extend(res)

        # mid
        x = self.mid_block(
            x,
            temb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )

        # up path
        for block in self.up_blocks:
            if isinstance(block, CrossAttnUpBlock2D):
                x = block(
                    x,
                    temb,
                    res_hidden_states_list=res_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
            else:
                x = block(
                    x,
                    temb,
                    res_hidden_states_list=res_hidden_states,
                )

        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x
