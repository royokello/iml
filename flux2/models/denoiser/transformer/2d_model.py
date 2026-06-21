
from typing import Any

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FluxTransformer2DLoadersMixin, FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention import AttentionMixin
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.utils import apply_lora_scale, logging

from ..embeddings.positional import Flux2PosEmbed
from ..embeddings.timestep import Flux2TimestepGuidanceEmbeddings
from ..kv.cache import Flux2KVCache
from ..modulation import Flux2Modulation
from .block import Flux2TransformerBlock
from .output import Flux2Transformer2DModelOutput
from .single_block import Flux2SingleTransformerBlock


def _blend_mod_params(
    img_params: tuple[torch.Tensor, ...],
    ref_params: tuple[torch.Tensor, ...],
    num_ref: int,
    seq_len: int,
) -> tuple[torch.Tensor, ...]:
    """Blend modulation parameters so that the first `num_ref` positions use `ref_params`."""
    blended = []
    for im, rm in zip(img_params, ref_params):
        if im.ndim == 2:
            im = im.unsqueeze(1)
            rm = rm.unsqueeze(1)
        B = im.shape[0]
        blended.append(
            torch.cat(
                [rm.expand(B, num_ref, -1), im.expand(B, seq_len, -1)[:, num_ref:, :]],
                dim=1,
            )
        )
    return tuple(blended)


def _blend_double_block_mods(
    img_mod: torch.Tensor,
    ref_mod: torch.Tensor,
    num_ref: int,
    seq_len: int,
) -> torch.Tensor:
    """Blend double-block image-stream modulations for a [ref, img] sequence layout.

    Takes raw modulation tensors (before `Flux2Modulation.split`) and returns a blended raw tensor that is compatible
    with `Flux2Modulation.split(mod, 2)`.
    """
    if img_mod.ndim == 2:
        img_mod = img_mod.unsqueeze(1)
        ref_mod = ref_mod.unsqueeze(1)
    img_chunks = torch.chunk(img_mod, 6, dim=-1)
    ref_chunks = torch.chunk(ref_mod, 6, dim=-1)
    img_mods = (img_chunks[0:3], img_chunks[3:6])
    ref_mods = (ref_chunks[0:3], ref_chunks[3:6])

    all_params = []
    for img_set, ref_set in zip(img_mods, ref_mods):
        blended = _blend_mod_params(img_set, ref_set, num_ref, seq_len)
        all_params.extend(blended)
    return torch.cat(all_params, dim=-1)


def _blend_single_block_mods(
    single_mod: torch.Tensor,
    ref_mod: torch.Tensor,
    num_txt: int,
    num_ref: int,
    seq_len: int,
) -> torch.Tensor:
    """Blend single-block modulations for a [txt, ref, img] sequence layout.

    Takes raw modulation tensors and returns a blended raw tensor compatible with `Flux2Modulation.split(mod, 1)`.
    """
    if single_mod.ndim == 2:
        single_mod = single_mod.unsqueeze(1)
        ref_mod = ref_mod.unsqueeze(1)
    img_params = torch.chunk(single_mod, 3, dim=-1)
    ref_params = torch.chunk(ref_mod, 3, dim=-1)

    blended = []
    for im, rm in zip(img_params, ref_params):
        if im.ndim == 2:
            im = im.unsqueeze(1)
            rm = rm.unsqueeze(1)
        B = im.shape[0]
        im_expanded = im.expand(B, seq_len, -1)
        rm_expanded = rm.expand(B, num_ref, -1)
        blended.append(
            torch.cat(
                [im_expanded[:, :num_txt, :], rm_expanded, im_expanded[:, num_txt + num_ref :, :]],
                dim=1,
            )
        )
    return torch.cat(blended, dim=-1)


class Flux2Transformer2DModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    FluxTransformer2DLoadersMixin,
    CacheMixin,
    AttentionMixin,
):
    """
    The Transformer model introduced in Flux 2.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        patch_size (`int`, defaults to `1`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `128`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `8`):
            The number of layers of dual stream DiT blocks to use.
        num_single_layers (`int`, defaults to `48`):
            The number of layers of single stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `48`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `15360`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        pooled_projection_dim (`int`, defaults to `768`):
            The number of dimensions to use for the pooled projection.
        guidance_embeds (`bool`, defaults to `True`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`tuple[int]`, defaults to `(32, 32, 32, 32)`):
            The dimensions to use for the rotary positional embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["Flux2TransformerBlock", "Flux2SingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["Flux2TransformerBlock", "Flux2SingleTransformerBlock"]
    _cp_plan = {
        "": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "img_ids": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "txt_ids": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: int | None = None,
        num_layers: int = 8,
        num_single_layers: int = 48,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        joint_attention_dim: int = 15360,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 2000,
        eps: float = 1e-6,
        guidance_embeds: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        # 1. Sinusoidal positional embedding for RoPE on image and text tokens
        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)

        # 2. Combined timestep + guidance embedding
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            bias=False,
            guidance_embeds=guidance_embeds,
        )

        # 3. Modulation (double stream and single stream blocks share modulation parameters, resp.)
        # Two sets of shift/scale/gate modulation parameters for the double stream attn and FF sub-blocks
        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        # Only one set of modulation parameters as the attn and FF sub-blocks are run in parallel for single stream
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1, bias=False)

        # 4. Input projections
        self.x_embedder = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)

        # 5. Double Stream Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                Flux2TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_layers)
            ]
        )

        # 6. Single Stream Transformer Blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                Flux2SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_single_layers)
            ]
        )

        # 7. Output layers
        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=eps, bias=False
        )
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)

        self.gradient_checkpointing = False
        self._offload_device: torch.device | None = None
        self._offload_non_blocking: bool = True

    _skip_keys = ["kv_cache"]

    def enable_block_offload(
        self,
        device: torch.device,
        *,
        non_blocking: bool = True,
    ) -> None:
        """Stream transformer blocks to `device` one at a time during forward.

        When enabled, the two block loops in `forward` move a single block to
        `device`, run it, and move it back to CPU before the next block is
        loaded. Always-resident layers (embedders, modulations, output) must
        already be on `device`; call `move_resident_layers_to(model, device)`
        from the loader before enabling.
        """
        self._offload_device = device
        self._offload_non_blocking = bool(non_blocking)

    @apply_lora_scale("joint_attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
        kv_cache: "Flux2KVCache | None" = None,
        kv_cache_mode: str | None = None,
        num_ref_tokens: int = 0,
        ref_fixed_timestep: float = 0.0,
    ) -> torch.Tensor | Flux2Transformer2DModelOutput:
        """
        The [`Flux2Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
            kv_cache (`Flux2KVCache`, *optional*):
                KV cache for reference image tokens. When `kv_cache_mode` is "extract", a new cache is created and
                returned. When "cached", the provided cache is used to inject ref K/V during attention.
            kv_cache_mode (`str`, *optional*):
                One of "extract" (first step with ref tokens) or "cached" (subsequent steps using cached ref K/V). When
                `None`, standard forward pass without KV caching.
            num_ref_tokens (`int`, defaults to `0`):
                Number of reference image tokens prepended to `hidden_states` (only used when
                `kv_cache_mode="extract"`).
            ref_fixed_timestep (`float`, defaults to `0.0`):
                Fixed timestep for reference token modulation (only used when `kv_cache_mode="extract"`).

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor. When `kv_cache_mode="extract"`, also returns the
            populated `Flux2KVCache`.
        """
        num_txt_tokens = encoder_hidden_states.shape[1]

        # 1. Calculate timestep embedding and modulation parameters
        timestep = timestep.to(hidden_states.dtype) * 1000

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = self.time_guidance_embed(timestep, guidance)

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)

        # KV extract mode: create cache and blend modulations for ref tokens
        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            num_img_tokens = hidden_states.shape[1]  # includes ref tokens

            kv_cache = Flux2KVCache(
                num_double_layers=len(self.transformer_blocks),
                num_single_layers=len(self.single_transformer_blocks),
            )
            kv_cache.num_ref_tokens = num_ref_tokens

            # Ref tokens use a fixed timestep for modulation
            ref_timestep = torch.full_like(timestep, ref_fixed_timestep * 1000)
            ref_temb = self.time_guidance_embed(ref_timestep, guidance)

            ref_double_mod_img = self.double_stream_modulation_img(ref_temb)
            ref_single_mod = self.single_stream_modulation(ref_temb)

            # Blend double block img modulation: [ref_mod, img_mod]
            double_stream_mod_img = _blend_double_block_mods(
                double_stream_mod_img, ref_double_mod_img, num_ref_tokens, num_img_tokens
            )

        # 2. Input projection for image (hidden_states) and conditioning text (encoder_hidden_states)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # 3. Calculate RoPE embeddings from image and text tokens
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        # 4. Build joint_attention_kwargs with KV cache info
        if kv_cache_mode == "extract":
            kv_attn_kwargs = {
                **(joint_attention_kwargs or {}),
                "kv_cache": None,
                "kv_cache_mode": "extract",
                "num_ref_tokens": num_ref_tokens,
            }
        elif kv_cache_mode == "cached" and kv_cache is not None:
            kv_attn_kwargs = {
                **(joint_attention_kwargs or {}),
                "kv_cache": None,
                "kv_cache_mode": "cached",
                "num_ref_tokens": kv_cache.num_ref_tokens,
            }
        else:
            kv_attn_kwargs = joint_attention_kwargs

        # 5. Double Stream Transformer Blocks
        for index_block, block in enumerate(self.transformer_blocks):
            if kv_cache_mode is not None and kv_cache is not None:
                kv_attn_kwargs["kv_cache"] = kv_cache.get_double(index_block)

            if self._offload_device is not None:
                block.to(self._offload_device, non_blocking=self._offload_non_blocking)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    double_stream_mod_img,
                    double_stream_mod_txt,
                    concat_rotary_emb,
                    kv_attn_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod_img=double_stream_mod_img,
                    temb_mod_txt=double_stream_mod_txt,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=kv_attn_kwargs,
                )

            if self._offload_device is not None:
                block.to("cpu", non_blocking=self._offload_non_blocking)

        # Concatenate text and image streams for single-block inference
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # Blend single block modulation for extract mode: [txt_mod, ref_mod, img_mod]
        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            total_single_len = hidden_states.shape[1]
            single_stream_mod = _blend_single_block_mods(
                single_stream_mod, ref_single_mod, num_txt_tokens, num_ref_tokens, total_single_len
            )

        # Build single-block KV kwargs (single blocks need num_txt_tokens)
        if kv_cache_mode is not None:
            kv_attn_kwargs_single = {**kv_attn_kwargs, "num_txt_tokens": num_txt_tokens}
        else:
            kv_attn_kwargs_single = kv_attn_kwargs

        # 6. Single Stream Transformer Blocks
        for index_block, block in enumerate(self.single_transformer_blocks):
            if kv_cache_mode is not None and kv_cache is not None:
                kv_attn_kwargs_single["kv_cache"] = kv_cache.get_single(index_block)

            if self._offload_device is not None:
                block.to(self._offload_device, non_blocking=self._offload_non_blocking)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    None,
                    single_stream_mod,
                    concat_rotary_emb,
                    kv_attn_kwargs_single,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=None,
                    temb_mod=single_stream_mod,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=kv_attn_kwargs_single,
                )

            if self._offload_device is not None:
                block.to("cpu", non_blocking=self._offload_non_blocking)

        # Remove text tokens (and ref tokens in extract mode) from concatenated stream
        if kv_cache_mode == "extract" and num_ref_tokens > 0:
            hidden_states = hidden_states[:, num_txt_tokens + num_ref_tokens :, ...]
        else:
            hidden_states = hidden_states[:, num_txt_tokens:, ...]

        # 7. Output layers
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if kv_cache_mode == "extract":
            if not return_dict:
                return (output, kv_cache)
            return Flux2Transformer2DModelOutput(sample=output, kv_cache=kv_cache)

        if not return_dict:
            return (output,)

        return Flux2Transformer2DModelOutput(sample=output)
