import json
import math
from pathlib import Path
from typing import Dict

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


def load_unet_key_mapping(base_dir: str) -> Dict[str, str]:
    """
    Load `<base>/unet/mapping.json` and return it as-is.
    """
    mapping_path = Path(base_dir) / "unet" / "mapping.json"
    return json.loads(mapping_path.read_text(encoding="utf-8"))


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
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.linear1 = Linear(
            in_features=input_dim,
            out_features=embed_dim,
            bias=True,
        )
        self.linear2 = Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=True,
        )
        self.act = nn.SiLU()

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() == 0:
            timesteps = timesteps[None]
        emb = timestep_embedding(timesteps, self.input_dim)
        tensor_q = quantize_input_and_attach_scale(self.linear1, emb, channel_dim=emb.ndim - 1)
        emb = self.linear1(tensor_q)
        emb = self.act(emb)
        tensor_q = quantize_input_and_attach_scale(self.linear2, emb, channel_dim=emb.ndim - 1)
        emb = self.linear2(tensor_q)
        return emb


class SDXLUNet(nn.Module):

    def __init__(
        self,
        model: dict[str, torch.Tensor] | None = None,
        key_mapping: Dict[str, str] | None = None,
    ):
        super().__init__()
        self._key_mapping = dict(key_mapping or {})

        # SDXL-base UNet configuration
        self.in_channels = 4
        self.out_channels = 4
        self.block_out_channels = (320, 640, 1280)
        self.time_embed_dim = 1280
        self.cross_attention_dim = 2048
        self.attention_head_dim = 64
        # 1) input conv
        self.conv_in = Conv2d(
            in_channels=self.in_channels,
            out_channels=self.block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # 2) time embedding
        self.time_embed = TimeEmbedding(self.block_out_channels[0], self.time_embed_dim)
        self.label_emb_in_dim = 2816
        self.label_emb = nn.Sequential(
            Linear(self.label_emb_in_dim, self.time_embed_dim, bias=True),
            nn.SiLU(),
            Linear(self.time_embed_dim, self.time_embed_dim, bias=True),
        )

        # 3) down blocks
        self.down_blocks = nn.ModuleList()
        # First plain DownBlock (320 -> 320) with downsample.
        self.down_blocks.append(
            DownBlock2D(
                in_channels=320,
                out_channels=320,
                temb_channels=1280,
                num_layers=2,
                add_downsample=True,
                use_conv_down=True,
            )
        )
        # Cross-attn DownBlock (320 -> 640) with transformer depth 2.
        self.down_blocks.append(
            CrossAttnDownBlock2D(
                in_channels=320,
                out_channels=640,
                temb_channels=1280,
                num_layers=2,
                num_attention_heads=10,
                head_dim=64,
                cross_attention_dim=2048,
                add_downsample=True,
                num_groups=32,
                transformer_depth=2,
                use_conv_down=True,
            )
        )
        # Final CrossAttn DownBlock (640 -> 1280) without downsample, transformer depth 10.
        self.down_blocks.append(
            CrossAttnDownBlock2D(
                in_channels=640,
                out_channels=1280,
                temb_channels=1280,
                num_layers=2,
                num_attention_heads=20,
                head_dim=64,
                cross_attention_dim=2048,
                add_downsample=False,
                num_groups=32,
                transformer_depth=10,
                use_conv_down=True,
            )
        )

        # 4) mid block
        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=1280,
            out_channels=1280,
            temb_channels=1280,
            num_attention_heads=20,
            head_dim=64,
            cross_attention_dim=2048,
            num_layers=1,
            transformer_depth=10,
            num_groups=32,
        )

        # 4.5) figure out skip channel ordering for up blocks
        skip_channels_stack = self._collect_skip_channels_for_up_blocks()

        def take_skip_channels(num_layers: int) -> list[int]:
            if len(skip_channels_stack) < num_layers:
                raise ValueError(
                    f"Not enough skip tensors ({len(skip_channels_stack)}) for {num_layers} layers"
                )
            return [skip_channels_stack.pop() for _ in range(num_layers)]

        # 5) up blocks
        self.up_blocks = nn.ModuleList()
        # CrossAttn block at the lowest resolution with three layers (each depth 10).
        num_heads = max(1, 1280 // 64)
        skip_channels = take_skip_channels(3)
        self.up_blocks.append(
            CrossAttnUpBlock2D(
                in_channels=1280,
                out_channels=1280,
                temb_channels=1280,
                num_layers=3,
                num_attention_heads=num_heads,
                head_dim=64,
                cross_attention_dim=2048,
                add_upsample=True,
                num_groups=32,
                layer_transformer_depths=(10, 10, 10),
                skip_channels_per_layer=skip_channels,
                use_conv_up=True,
            )
        )

        # Mid-resolution CrossAttn block with three layers (each depth 2).
        num_heads = max(1, 640 // 64)
        skip_channels = take_skip_channels(3)
        self.up_blocks.append(
            CrossAttnUpBlock2D(
                in_channels=1280,
                out_channels=640,
                temb_channels=1280,
                num_layers=3,
                num_attention_heads=num_heads,
                head_dim=64,
                cross_attention_dim=2048,
                add_upsample=True,
                num_groups=32,
                layer_transformer_depths=(2, 2, 2),
                skip_channels_per_layer=skip_channels,
                use_conv_up=True,
            )
        )

        # Final plain UpBlock with three ResNets (no attention).
        skip_channels = take_skip_channels(3)
        self.up_blocks.append(
            UpBlock2D(
                in_channels=640,
                out_channels=320,
                temb_channels=1280,
                num_layers=3,
                add_upsample=False,
                num_groups=32,
                skip_channels_per_layer=skip_channels,
                use_conv_up=True,
            )
        )

        if skip_channels_stack:
            raise ValueError(
                f"Unused skip channel definitions remain: {len(skip_channels_stack)}"
            )

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

    def _collect_skip_channels_for_up_blocks(self) -> list[int]:
        """
        Build the list of skip-connection channel counts in the exact order that
        `res_hidden_states` will be populated during the forward pass. This allows
        us to size the up block ResNets to match checkpoint expectations.
        """
        skip_channels: list[int] = [self.block_out_channels[0]]
        for block in self.down_blocks:
            resnets = getattr(block, "resnets", [])
            for resnet in resnets:
                skip_channels.append(resnet.out_channels)
            downsamplers = getattr(block, "downsamplers", [])
            for downsampler in downsamplers:
                skip_channels.append(downsampler.channels)
        return skip_channels

    def _load_filtered_weights(self, model: dict[str, torch.Tensor]) -> None:
        """
        Filter-load only keys that exist in this UNet and match in shape.
        `model` is expected to be a dict like from safetensors or state_dict().
        """
        state = self.state_dict()
        updated: dict[str, torch.Tensor] = {}
        matched = 0
        missing: list[str] = []

        for name, target in state.items():
            if name.endswith(".scale_x"):
                continue
            source_name = self._key_mapping.get(name, name)
            tensor = model.get(source_name)
            if tensor is None:
                missing.append(f"{name} (looked for {source_name})")
                continue

            updated[name] = tensor.to(dtype=target.dtype)
            matched += 1

        missing_count = len(missing)
        print(f"[unet-load] matched={matched}, missing={missing_count}")
        if missing_count:
            missing_keys = ", ".join(missing)
            raise SystemExit(
                f"[unet-load] aborting because tensors are missing: {missing_keys}"
            )

        state.update(updated)
        self.load_state_dict(state, strict=False)

    def forward(
        self,
        sample: torch.Tensor,                 # [B, 4, H, W]
        timesteps: torch.Tensor,              # [B] or scalar
        encoder_hidden_states: torch.Tensor,  # [B, T, 2048]
        attention_mask: torch.Tensor | None = None,
        pooled_embeds: torch.Tensor | None = None,  # [B, 2816]
    ) -> torch.Tensor:
        print("[debug] sample stats:", float(sample.min()), float(sample.max()))
        tensor_q = quantize_input_and_attach_scale(self.conv_in, sample, channel_dim=1)
        x = self.conv_in(tensor_q)
        print("[debug] conv_in output stats:", float(x.min()), float(x.max()))
        temb = self.time_embed(timesteps)
        if pooled_embeds is not None:
            print("[debug] pooled_embeds stats:", float(pooled_embeds.min()), float(pooled_embeds.max()))
            tensor_q = quantize_input_and_attach_scale(
                self.label_emb[0], pooled_embeds, channel_dim=pooled_embeds.ndim - 1
            )
            pooled = self.label_emb(tensor_q)
            temb = temb + pooled

        res_hidden_states: list[torch.Tensor] = [x]

        # down path
        for idx, block in enumerate(self.down_blocks):
            if isinstance(block, CrossAttnDownBlock2D):
                x, res = block(
                    x,
                    temb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
            else:
                x, res = block(x, temb)
            print(f"[debug] down block {idx} output stats:", float(x.min()), float(x.max()))
            res_hidden_states.extend(res)

        # mid
        x = self.mid_block(
            x,
            temb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        print("[debug] mid block output stats:", float(x.min()), float(x.max()))

        # up path
        for idx, block in enumerate(self.up_blocks):
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
            print(f"[debug] up block {idx} output stats:", float(x.min()), float(x.max()))

        x = self.conv_norm_out(x)
        print("[debug] conv_norm_out stats:", float(x.min()), float(x.max()))
        x = self.conv_act(x)
        print("[debug] conv_act stats:", float(x.min()), float(x.max()))
        tensor_q = quantize_input_and_attach_scale(self.conv_out, x, channel_dim=1)
        x = self.conv_out(tensor_q)
        print("[debug] conv_out stats:", float(x.min()), float(x.max()))
        return x
