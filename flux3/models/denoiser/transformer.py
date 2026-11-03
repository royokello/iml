from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import AttentionMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import apply_lora_scale

from flux2.models.denoiser.embeddings.positional import Flux2PosEmbed

from .modulation import Flux3AdaLayerNorm, Flux3Modulation
from .precision import linear_fp32, timestep_embedding_fp32
from .mode_block import Flux3ModeTransformerBlock
from .single_block import Flux3SingleTransformerBlock


class Flux3Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, AttentionMixin):
    """The Flux 3 denoiser.

    Three token modes share one joint sequence: the noisy image tokens, the
    optional reference image tokens (`image_cond`), and the text tokens. Each
    mode first passes through its own mode blocks. The text tokens and the
    image tokens then pass through the joint single blocks. Only the image
    tokens reach the output layers.

    The image tokens use the flow timestep. The reference tokens and the text
    tokens use timestep 0. A flow-map model (`flow_map=True`, the distilled
    model) also adds an embedding of the target timestep to every mode.

    Args:
        in_channels: Channel width of the image tokens.
        cond_in_channels: Channel width of the reference image tokens.
        num_layers: Number of mode blocks in each mode.
        num_single_layers: Number of joint single blocks.
        attention_head_dim: Size of each attention head.
        num_attention_heads: Number of attention heads.
        joint_attention_dim: Channel width of `encoder_hidden_states`.
        timestep_channels: Channel width of the sinusoidal timestep embedding.
        mlp_ratio: Expansion ratio of the feed-forward layers.
        axes_dims_rope: Size of each RoPE axis.
        rope_theta: Base frequency of the RoPE.
        eps: Epsilon of the layer norms and the QK norms.
        flow_map: If true, add `target_timestep_embedder` and require
            `target_timestep` in `forward`.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["Flux3ModeTransformerBlock", "Flux3SingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["Flux3ModeTransformerBlock", "Flux3SingleTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        cond_in_channels: int = 128,
        num_layers: int = 5,
        num_single_layers: int = 28,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 20480,
        timestep_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 10000,
        eps: float = 1e-6,
        flow_map: bool = False,
    ):
        super().__init__()

        self.inner_dim = num_attention_heads * attention_head_dim
        in_channels_per_mode = {"image": in_channels, "image_cond": cond_in_channels}
        modes = list(in_channels_per_mode.keys())

        # 1. RoPE for all token modes.
        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=list(axes_dims_rope))

        # 2. Timestep embedding.
        self.time_proj = Timesteps(num_channels=timestep_channels, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=timestep_channels, time_embed_dim=self.inner_dim, sample_proj_bias=False
        )
        self.target_timestep_embedder = (
            TimestepEmbedding(in_channels=timestep_channels, time_embed_dim=self.inner_dim, sample_proj_bias=False)
            if flow_map
            else None
        )

        # 3. Modulation heads. One per mode and one for the text.
        self.early_stream_modulations = nn.ModuleDict(
            {name: Flux3Modulation(self.inner_dim, bias=False) for name in [*modes, "txt"]}
        )
        self.single_stream_modulations = nn.ModuleDict(
            {name: Flux3Modulation(self.inner_dim, bias=False) for name in [*modes, "txt"]}
        )

        # 4. Input projections.
        self.x_embedders = nn.ModuleDict(
            {name: nn.Linear(channels, self.inner_dim, bias=False) for name, channels in in_channels_per_mode.items()}
        )
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)

        # 5. Mode blocks.
        self.mode_blocks = nn.ModuleDict(
            {
                name: nn.ModuleList(
                    [
                        Flux3ModeTransformerBlock(
                            dim=self.inner_dim,
                            num_attention_heads=num_attention_heads,
                            attention_head_dim=attention_head_dim,
                            mlp_ratio=mlp_ratio,
                            eps=eps,
                        )
                        for _ in range(num_layers)
                    ]
                )
                for name in modes
            }
        )
        self.txt_mode_blocks = nn.ModuleList(
            [
                Flux3ModeTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )

        # 6. Joint single blocks.
        self.single_transformer_blocks = nn.ModuleList(
            [
                Flux3SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                )
                for _ in range(num_single_layers)
            ]
        )

        # 7. Output layers. Image tokens only.
        self.norm_out = Flux3AdaLayerNorm(self.inner_dim, eps=eps)
        self.proj_out = nn.Linear(self.inner_dim, in_channels, bias=False)

        self.gradient_checkpointing = False
        self._offload_device: torch.device | None = None

    def enable_block_offload(self, device: torch.device) -> None:
        """Stream the blocks to `device` one at a time during `forward`.

        Each block moves to `device` before it runs and back to the CPU after
        it runs. The resident layers must already be on `device`. Call
        `move_resident_layers_to(model, device)` from the loader first.

        Args:
            device: The device that runs the blocks.
        """
        self._offload_device = device

    def disable_block_offload(self) -> None:
        """Stop the block streaming. `forward` then uses the current block devices."""
        self._offload_device = None

    def _run_block(self, block: nn.Module, *args: torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
        if self._offload_device is not None:
            block.to(self._offload_device, non_blocking=True)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            output = self._gradient_checkpointing_func(block, *args)
        else:
            output = block(*args)

        if self._offload_device is not None:
            block.to("cpu", non_blocking=True)
        return output

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        image_cond_hidden_states: torch.Tensor | None = None,
        image_cond_ids: torch.Tensor | None = None,
        target_timestep: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        """Predict the velocity of the image tokens.

        Args:
            hidden_states: Noisy image tokens. Shape `(batch, image_len, in_channels)`.
            encoder_hidden_states: Text embeddings. Shape `(batch, text_len, joint_attention_dim)`.
            timestep: Flow timestep in `[0, 1]`. Shape `(batch,)`.
            img_ids: Position ids of the image tokens. Shape `(image_len, 4)`.
            txt_ids: Position ids of the text tokens. Shape `(text_len, 4)`.
            image_cond_hidden_states: Reference image tokens. Shape
                `(batch, cond_len, cond_in_channels)`.
            image_cond_ids: Position ids of the reference tokens. Required with
                `image_cond_hidden_states`.
            target_timestep: Flow timestep that the step jumps to. Required by,
                and only by, a flow-map model.
            attention_kwargs: Extra arguments. The `scale` entry sets the LoRA scale.
            return_dict: If true, return a `Transformer2DModelOutput`. If false,
                return a tuple.

        Returns:
            The predicted velocity. It has the shape of `hidden_states`.

        Raises:
            ValueError: If `image_cond_hidden_states` and `image_cond_ids` are
                not given together, or if `target_timestep` does not match
                `flow_map`.
        """
        if image_cond_hidden_states is not None and image_cond_ids is None:
            raise ValueError("`image_cond_hidden_states` requires `image_cond_ids`.")
        if image_cond_hidden_states is None and image_cond_ids is not None:
            raise ValueError("`image_cond_ids` was passed without `image_cond_hidden_states`.")
        if (target_timestep is not None) != bool(self.config.flow_map):
            raise ValueError(
                "`target_timestep` is required by, and only by, a flow-map transformer (`flow_map=True`)."
            )

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if image_cond_ids is not None and image_cond_ids.ndim == 3:
            image_cond_ids = image_cond_ids[0]

        batch_size, image_seq_len = hidden_states.shape[:2]
        dtype = hidden_states.dtype

        # 1. Timestep embeddings. The image tokens use `timestep`. The text and
        #    reference tokens use 0.
        image_temb = self.time_proj(timestep.to(dtype) * 1000)
        image_temb = timestep_embedding_fp32(self.timestep_embedder, image_temb)[:, None, :].to(dtype)
        zero_temb = self.time_proj(torch.zeros(batch_size, device=hidden_states.device, dtype=dtype))
        zero_temb = timestep_embedding_fp32(self.timestep_embedder, zero_temb)[:, None, :].to(dtype)
        if target_timestep is not None:
            target_temb = self.time_proj(target_timestep.to(dtype) * 1000)
            target_temb = timestep_embedding_fp32(self.target_timestep_embedder, target_temb)[:, None, :].to(dtype)
            image_temb = image_temb + target_temb
            zero_temb = zero_temb + target_temb
        image_temb = image_temb.expand(batch_size, image_seq_len, -1)
        zero_temb_act = F.silu(zero_temb)

        # 2. Text tokens.
        txt_temb_act = zero_temb_act.expand(batch_size, encoder_hidden_states.shape[1], -1)
        txt_early_mod = self.early_stream_modulations["txt"](txt_temb_act)
        txt_single_mod = self.single_stream_modulations["txt"](txt_temb_act)
        # The text stream runs in fp32. The Qwen hidden states hold large outliers, and the
        # fp16 sum over 20480 channels overflows. The blocks keep this stream in fp32.
        encoder_hidden_states = linear_fp32(self.context_embedder, encoder_hidden_states)
        txt_rotary_emb = self.pos_embed(txt_ids)

        # 3. Image tokens.
        image_temb_act = F.silu(image_temb)
        image_early_mod = self.early_stream_modulations["image"](image_temb_act)
        image_single_mod = self.single_stream_modulations["image"](image_temb_act)
        hidden_states = linear_fp32(self.x_embedders["image"], hidden_states).to(dtype)
        image_rotary_emb = self.pos_embed(img_ids)

        # 4. Reference image tokens. Optional.
        if image_cond_hidden_states is not None:
            cond_temb_act = zero_temb_act.expand(batch_size, image_cond_hidden_states.shape[1], -1)
            cond_early_mod = self.early_stream_modulations["image_cond"](cond_temb_act)
            cond_single_mod = self.single_stream_modulations["image_cond"](cond_temb_act)
            image_cond_hidden_states = linear_fp32(self.x_embedders["image_cond"], image_cond_hidden_states).to(dtype)
            cond_rotary_emb = self.pos_embed(image_cond_ids)

        # 5. Mode blocks. Each mode attends to itself only.
        for block in self.txt_mode_blocks:
            encoder_hidden_states = self._run_block(block, encoder_hidden_states, txt_early_mod, txt_rotary_emb)

        for block in self.mode_blocks["image"]:
            hidden_states = self._run_block(block, hidden_states, image_early_mod, image_rotary_emb)

        if image_cond_hidden_states is not None:
            for block in self.mode_blocks["image_cond"]:
                image_cond_hidden_states = self._run_block(
                    block, image_cond_hidden_states, cond_early_mod, cond_rotary_emb
                )

        # 6. Joint single blocks over [txt, image, image_cond].
        if image_cond_hidden_states is not None:
            content_states = torch.cat([hidden_states, image_cond_hidden_states], dim=1)
            content_mod = torch.cat([image_single_mod, cond_single_mod], dim=1)
            joint_rotary_emb = (
                torch.cat([txt_rotary_emb[0], image_rotary_emb[0], cond_rotary_emb[0]], dim=0),
                torch.cat([txt_rotary_emb[1], image_rotary_emb[1], cond_rotary_emb[1]], dim=0),
            )
        else:
            content_states = hidden_states
            content_mod = image_single_mod
            joint_rotary_emb = (
                torch.cat([txt_rotary_emb[0], image_rotary_emb[0]], dim=0),
                torch.cat([txt_rotary_emb[1], image_rotary_emb[1]], dim=0),
            )

        # The last single blocks hold image values near the fp16 limit (65504). The content
        # stream runs in fp32 through them. The mode blocks stay below 5000 and stay in fp16.
        content_states = content_states.float()
        for block in self.single_transformer_blocks:
            encoder_hidden_states, content_states = self._run_block(
                block, content_states, encoder_hidden_states, content_mod, txt_single_mod, joint_rotary_emb
            )

        # 7. Output layers. The image tokens come first in the content sequence.
        hidden_states = content_states[:, :image_seq_len]
        hidden_states = self.norm_out(hidden_states, image_temb)
        output = linear_fp32(self.proj_out, hidden_states).to(dtype)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
