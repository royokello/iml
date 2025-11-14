import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks.conv_2d import Conv2d
from model.blocks.cross_attn_down_block_2d import CrossAttnDownBlock2D
from model.blocks.cross_attn_up_block_2d import CrossAttnUpBlock2D
from model.blocks.down_block_2d import DownBlock2D
from model.blocks.linear import Linear
from model.blocks.unet_mid_block_2d_cross_attn import UNetMidBlock2DCrossAttn
from model.blocks.up_block_2d import UpBlock2D
from model.utils.quantization import quantize_input_and_attach_scale


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
        self.linear1 = Linear(
            in_features=dim,
            out_features=dim,
            bias=True,
        )
        self.linear2 = Linear(
            in_features=dim,
            out_features=dim,
            bias=True,
        )
        self.act = nn.SiLU()

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() == 0:
            timesteps = timesteps[None]
        emb = timestep_embedding(timesteps, self.dim)
        tensor_q = quantize_input_and_attach_scale(self.linear1, emb)
        emb = self.linear1(tensor_q)
        emb = self.act(emb)
        tensor_q = quantize_input_and_attach_scale(self.linear2, emb)
        emb = self.linear2(tensor_q)
        return emb


class SDXLUNet(nn.Module):
    def __init__(self, model: dict[str, torch.Tensor] | None = None):
        super().__init__()

        # SDXL-base UNet configuration
        self.in_channels = 4
        self.out_channels = 4
        self.block_out_channels = (320, 640, 1280, 1280)
        self.layers_per_block = 2
        self.time_embed_dim = 1280
        self.cross_attention_dim = 2048
        self.attention_head_dim = 64
        self.down_block_types = (
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        )
        self.up_block_types = (
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
        )

        # 1) input conv
        self.conv_in = Conv2d(
            in_channels=self.in_channels,
            out_channels=self.block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # 2) time embedding
        self.time_embed = TimeEmbedding(self.time_embed_dim)

        # 3) down blocks
        self.down_blocks = nn.ModuleList()
        curr_in_channels = self.block_out_channels[0]
        for idx, (block_type, out_channels) in enumerate(
            zip(self.down_block_types, self.block_out_channels)
        ):
            add_downsample = idx < len(self.block_out_channels) - 1
            if block_type == "DownBlock2D":
                block = DownBlock2D(
                    in_channels=curr_in_channels,
                    out_channels=out_channels,
                    temb_channels=self.time_embed_dim,
                    num_layers=self.layers_per_block,
                    add_downsample=add_downsample,
                    use_conv_down=True,
                )
            elif block_type == "CrossAttnDownBlock2D":
                num_heads = max(1, out_channels // self.attention_head_dim)
                block = CrossAttnDownBlock2D(
                    in_channels=curr_in_channels,
                    out_channels=out_channels,
                    temb_channels=self.time_embed_dim,
                    num_layers=self.layers_per_block,
                    num_attention_heads=num_heads,
                    head_dim=self.attention_head_dim,
                    cross_attention_dim=self.cross_attention_dim,
                    add_downsample=add_downsample,
                    num_groups=32,
                    transformer_depth=1,
                    use_conv_down=True,
                )
            else:
                raise ValueError(f"Unsupported down block type: {block_type}")
            self.down_blocks.append(block)
            curr_in_channels = out_channels

        # 4) mid block
        mid_channels = self.block_out_channels[-1]
        mid_heads = max(1, mid_channels // self.attention_head_dim)
        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=mid_channels,
            out_channels=mid_channels,
            temb_channels=self.time_embed_dim,
            num_attention_heads=mid_heads,
            head_dim=self.attention_head_dim,
            cross_attention_dim=self.cross_attention_dim,
            num_layers=1,
            num_groups=32,
        )

        # 5) up blocks
        self.up_blocks = nn.ModuleList()
        curr_in_channels = self.block_out_channels[-1]
        reversed_block_out_channels = list(reversed(self.block_out_channels))
        for idx, (block_type, out_channels) in enumerate(
            zip(self.up_block_types, reversed_block_out_channels)
        ):
            add_upsample = idx < len(self.up_block_types) - 1
            if block_type == "CrossAttnUpBlock2D":
                num_heads = max(1, out_channels // self.attention_head_dim)
                block = CrossAttnUpBlock2D(
                    in_channels=curr_in_channels,
                    out_channels=out_channels,
                    temb_channels=self.time_embed_dim,
                    num_layers=self.layers_per_block,
                    num_attention_heads=num_heads,
                    head_dim=self.attention_head_dim,
                    cross_attention_dim=self.cross_attention_dim,
                    add_upsample=add_upsample,
                    num_groups=32,
                    transformer_depth=1,
                    use_conv_up=True,
                )
            elif block_type == "UpBlock2D":
                block = UpBlock2D(
                    in_channels=curr_in_channels,
                    out_channels=out_channels,
                    temb_channels=self.time_embed_dim,
                    num_layers=self.layers_per_block,
                    add_upsample=add_upsample,
                    num_groups=32,
                    use_conv_up=True,
                )
            else:
                raise ValueError(f"Unsupported up block type: {block_type}")
            self.up_blocks.append(block)
            curr_in_channels = out_channels

        # 6) output head
        self.conv_norm_out = nn.GroupNorm(32, self.block_out_channels[0], eps=1e-5, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = Conv2d(
            in_channels=self.block_out_channels[0],
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

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
        tensor_q = quantize_input_and_attach_scale(self.conv_in, sample)
        x = self.conv_in(tensor_q)
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
        tensor_q = quantize_input_and_attach_scale(self.conv_out, x)
        x = self.conv_out(tensor_q)
        return x
