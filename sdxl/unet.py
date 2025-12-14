import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from models.blocks.conv_2d import Conv2d
from models.blocks.cross_attn_down_block_2d import CrossAttnDownBlock2D
from models.blocks.cross_attn_up_block_2d import CrossAttnUpBlock2D
from models.blocks.down_block_2d import DownBlock2D
from models.blocks.linear import LinearFP16, LinearInt8
from models.blocks.unet_mid_block_2d_cross_attn import UNetMidBlock2DCrossAttn
from models.blocks.up_block_2d import UpBlock2D
from models.embeddings import GaussianFourierProjection, ImageHintTimeEmbedding, ImageProjection, ImageTimeEmbedding, TextImageProjection, TextImageTimeEmbedding, TextTimeEmbedding, TimestepEmbedding, Timesteps, get_activation


def load_unet_key_mapping(base_dir: str) -> Dict[str, str]:
    """
    Load `<base>/unet/mapping.json` and return it as-is.
    """
    mapping_path = Path(base_dir) / "unet" / "mapping.json"
    return json.loads(mapping_path.read_text(encoding="utf-8"))


class AddEmbedding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.linear_1 = LinearFP16(
            in_features=input_dim,
            out_features=embed_dim,
            bias=True,
        )
        self.linear_2 = LinearFP16(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=True,
        )
        self.act = nn.SiLU()

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(embeds)
        x = self.act(x)
        x = self.linear_2(x)
        return x


OPTIONAL_TENSORS = {
    # Fourier time projection weights are procedurally re-created from config, so
    # they are not serialized in the original SDXL checkpoints.
    "time_proj.weight",
}


class SDXLUNet(nn.Module):

    def __init__(
        self,
        model: dict[str, torch.Tensor] | None = None,
        key_mapping: Dict[str, str] | None = None,
    ):
        super().__init__()
        self._key_mapping = dict(key_mapping or {})

        # SDXL-base UNet configuration
        self.config = {
            "in_channels": 4,
            "out_channels": 4,
            "block_out_channels": (320, 640, 1280),
            "time_embed_dim": 1280,
            "cross_attention_dim": 2048,
            "attention_head_dim": 64,
            # Hard-coded SDXL defaults for the inherited Diffusers helpers.
            "time_embedding_type": "positional",
            "flip_sin_to_cos": True,
            "freq_shift": 0,
            "time_embedding_dim": 1280,
            "act_fn": "silu",
            "timestep_post_act": None,
            "time_cond_proj_dim": None,
            "encoder_hid_dim_type": None,
            "encoder_hid_dim": None,
            "class_embed_type": None,
            "num_class_embeds": None,
            "projection_class_embeddings_input_dim": 1280,
            "addition_embed_type": "text_time",
            "addition_embed_type_num_heads": 64,
            "addition_time_embed_dim": 256,
            "time_embedding_act_fn": None,
            "class_embeddings_concat": False,
        }

        # time
        time_embed_dim, timestep_input_dim = self._set_time_proj(
            time_embedding_type=self.config["time_embedding_type"],
            block_out_channels=self.config["block_out_channels"],
            flip_sin_to_cos=self.config["flip_sin_to_cos"],
            freq_shift=self.config["freq_shift"],
            time_embedding_dim=self.config["time_embedding_dim"],
        )

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=self.config["act_fn"],
            post_act_fn=self.config["timestep_post_act"],
            cond_proj_dim=self.config["time_cond_proj_dim"],
        )

        self._set_encoder_hid_proj(
            encoder_hid_dim_type=self.config["encoder_hid_dim_type"],
            cross_attention_dim=self.config["cross_attention_dim"],
            encoder_hid_dim=self.config["encoder_hid_dim"],
        )

        # class embedding
        self._set_class_embedding(
            self.config["class_embed_type"],
            act_fn=self.config["act_fn"],
            num_class_embeds=self.config["num_class_embeds"],
            projection_class_embeddings_input_dim=self.config["projection_class_embeddings_input_dim"],
            time_embed_dim=time_embed_dim,
            timestep_input_dim=timestep_input_dim,
        )

        self._set_add_embedding(
            self.config["addition_embed_type"],
            addition_embed_type_num_heads=self.config["addition_embed_type_num_heads"],
            addition_time_embed_dim=self.config["addition_time_embed_dim"],
            cross_attention_dim=self.config["cross_attention_dim"],
            encoder_hid_dim=self.config["encoder_hid_dim"],
            flip_sin_to_cos=self.config["flip_sin_to_cos"],
            freq_shift=self.config["freq_shift"],
            projection_class_embeddings_input_dim=self.config["projection_class_embeddings_input_dim"],
            time_embed_dim=time_embed_dim,
        )

        if self.config["time_embedding_act_fn"] is None:
            self.time_embed_act = None
        else:
            self.time_embed_act = get_activation(self.config["time_embedding_act_fn"])


        # 1) input conv
        self.conv_in = Conv2d(
            in_channels=self.config["in_channels"],
            out_channels=self.config["block_out_channels"][0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # 2) time embedding
        self.label_emb_in_dim = 2816
        self.add_embedding = AddEmbedding(
            self.label_emb_in_dim, self.config["time_embed_dim"]
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
        self.conv_norm_out = nn.GroupNorm(32, self.config["block_out_channels"][0], eps=1e-5, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = Conv2d(
            in_channels=self.config["block_out_channels"][0],
            out_channels=self.config["out_channels"],
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
        skip_channels: list[int] = [self.config["block_out_channels"][0]]
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
        missing_details: list[tuple[str, tuple[int, ...]]] = []

        for name, target in state.items():
            source_name = self._key_mapping.get(name, name)
            tensor = model.get(source_name)
            if tensor is None:
                if name in OPTIONAL_TENSORS:
                    # Skip optional tensors that Diffusers rebuilds on init.
                    continue
                missing.append(f"{name} (looked for {source_name})")
                missing_details.append((name, tuple(int(dim) for dim in target.shape)))
                continue

            updated[name] = tensor.to(dtype=target.dtype)
            matched += 1

        missing_count = len(missing)
        total_expected = len(state)

        if missing_count:
            for tensor_name, tensor_shape in missing_details:
                print(f"[unet-load-missing] {tensor_name}: shape={tensor_shape}")
            print(f"[unet-load] matched={matched}, missing={missing_count}")
            print(f"[unet-load] expected={total_expected}, loaded={matched}")
            raise SystemExit("[unet-load] aborting because tensors are missing (see above)")
        else:
            print(f"[unet-load] matched={matched}, missing={missing_count}")
            print(f"[unet-load] expected={total_expected}, loaded={matched}")

        state.update(updated)
        self.load_state_dict(state, strict=False)

        for module in self.modules():
            if isinstance(module, LinearInt8):
                module._weights_loaded = True

    def get_time_embed(
        self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int]
    ) -> Optional[torch.Tensor]:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            is_npu = sample.device.type == "npu"
            if isinstance(timestep, float):
                dtype = torch.float32 if (is_mps or is_npu) else torch.float64
            else:
                dtype = torch.int32 if (is_mps or is_npu) else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)
        return t_emb
    
    def get_class_embed(self, sample: torch.Tensor, class_labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        class_emb = None
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config["class_embed_type"] == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)
        return class_emb
    
    def _set_time_proj(
        self,
        time_embedding_type: str,
        block_out_channels: int,
        flip_sin_to_cos: bool,
        freq_shift: float,
        time_embedding_dim: int,
    ) -> Tuple[int, int]:
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        return time_embed_dim, timestep_input_dim

    def _set_encoder_hid_proj(
        self,
        encoder_hid_dim_type: Optional[str],
        cross_attention_dim: Union[int, Tuple[int]],
        encoder_hid_dim: Optional[int],
    ):
        if encoder_hid_dim_type is None and encoder_hid_dim is not None:
            encoder_hid_dim_type = "text_proj"
            self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)

        if encoder_hid_dim is None and encoder_hid_dim_type is not None:
            raise ValueError(
                f"`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to {encoder_hid_dim_type}."
            )

        if encoder_hid_dim_type == "text_proj":
            self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
        elif encoder_hid_dim_type == "text_image_proj":
            # image_embed_dim DOESN'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image_proj"` (Kandinsky 2.1)`
            self.encoder_hid_proj = TextImageProjection(
                text_embed_dim=encoder_hid_dim,
                image_embed_dim=cross_attention_dim,
                cross_attention_dim=cross_attention_dim,
            )
        elif encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2
            self.encoder_hid_proj = ImageProjection(
                image_embed_dim=encoder_hid_dim,
                cross_attention_dim=cross_attention_dim,
            )
        elif encoder_hid_dim_type is not None:
            raise ValueError(
                f"`encoder_hid_dim_type`: {encoder_hid_dim_type} must be None, 'text_proj', 'text_image_proj', or 'image_proj'."
            )
        else:
            self.encoder_hid_proj = None

    def _set_class_embedding(
        self,
        class_embed_type: Optional[str],
        act_fn: str,
        num_class_embeds: Optional[int],
        projection_class_embeddings_input_dim: Optional[int],
        time_embed_dim: int,
        timestep_input_dim: int,
    ):
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, act_fn=act_fn)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
            # The projection `class_embed_type` is the same as the timestep `class_embed_type` except
            # 1. the `class_labels` inputs are not first converted to sinusoidal embeddings
            # 2. it projects from an arbitrary input dimension.
            #
            # Note that `TimestepEmbedding` is quite general, being mainly linear layers and activations.
            # When used for embedding actual timesteps, the timesteps are first converted to sinusoidal embeddings.
            # As a result, `TimestepEmbedding` can be passed arbitrary vectors.
            self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif class_embed_type == "simple_projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'simple_projection' requires `projection_class_embeddings_input_dim` be set"
                )
            self.class_embedding = nn.Linear(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.class_embedding = None

    def _set_add_embedding(
        self,
        addition_embed_type: str,
        addition_embed_type_num_heads: int,
        addition_time_embed_dim: Optional[int],
        flip_sin_to_cos: bool,
        freq_shift: float,
        cross_attention_dim: Optional[int],
        encoder_hid_dim: Optional[int],
        projection_class_embeddings_input_dim: Optional[int],
        time_embed_dim: int,
    ):
        if addition_embed_type == "text":
            if encoder_hid_dim is not None:
                text_time_embedding_from_dim = encoder_hid_dim
            else:
                text_time_embedding_from_dim = cross_attention_dim

            self.add_embedding = TextTimeEmbedding(
                text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads
            )
        elif addition_embed_type == "text_image":
            # text_embed_dim and image_embed_dim DON'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image"` (Kandinsky 2.1)`
            self.add_embedding = TextImageTimeEmbedding(
                text_embed_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, time_embed_dim=time_embed_dim
            )
        elif addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif addition_embed_type == "image":
            # Kandinsky 2.2
            self.add_embedding = ImageTimeEmbedding(image_embed_dim=encoder_hid_dim, time_embed_dim=time_embed_dim)
        elif addition_embed_type == "image_hint":
            # Kandinsky 2.2 ControlNet
            self.add_embedding = ImageHintTimeEmbedding(image_embed_dim=encoder_hid_dim, time_embed_dim=time_embed_dim)
        elif addition_embed_type is not None:
            raise ValueError(
                f"`addition_embed_type`: {addition_embed_type} must be None, 'text', 'text_image', 'text_time', 'image', or 'image_hint'."
            )
        
    def get_aug_embed(
        self, emb: torch.Tensor, encoder_hidden_states: torch.Tensor, added_cond_kwargs: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        aug_emb = None
        if self.config["addition_embed_type"] == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config["addition_embed_type"] == "text_image":
            # Kandinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            image_embs = added_cond_kwargs.get("image_embeds")
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
            aug_emb = self.add_embedding(text_embs, image_embs)
        elif self.config["addition_embed_type"] == "text_time":
            # SDXL - style
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
        elif self.config["addition_embed_type"] == "image":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            aug_emb = self.add_embedding(image_embs)
        elif self.config["addition_embed_type"] == "image_hint":
            # Kandinsky 2.2 ControlNet - style
            if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            hint = added_cond_kwargs.get("hint")
            aug_emb = self.add_embedding(image_embs, hint)
        return aug_emb

    def process_encoder_hidden_states(
        self,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # SDXL only concatenates pooled text and time embeddings via `get_aug_embed`, so
        # encoder hidden states pass through unchanged.
        return encoder_hidden_states

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        r"""
        The [`UNet2DConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
                Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                through the `self.time_embedding` layer to obtain the timestep embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            added_cond_kwargs: (`dict`, *optional*):
                A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.
            down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                A tuple of tensors that if specified are added to the residuals of down unet blocks.
            mid_block_additional_residual: (`torch.Tensor`, *optional*):
                A tensor that if specified is added to the residual of the middle unet block.
            down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
                additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)
            encoder_attention_mask (`torch.Tensor`):
                A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                which adds large negative values to the attention scores corresponding to "discard" tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            torch.Tensor
        """

        # SDXL-base UNet always applies two upsampling stages, so overall factor is fixed.
        default_overall_up_factor = 4

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        # SDXL latents are already centered; the generic `center_input_sample` flag is never used.

        # 1. time
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb)

        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config["class_embeddings_concat"]:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if self.config["addition_embed_type"] == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        # 2. pre-process (following the original modified logic)
        x = self.conv_in(sample)  # [B, 4, H, W] -> [B, 320, H, W]

        res_hidden_states: list[torch.Tensor] = [x]

        # 3. down path
        for idx, block in enumerate(self.down_blocks):
            if isinstance(block, CrossAttnDownBlock2D):
                x, res = block(
                    x,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
            else:
                x, res = block(x, emb)

            res_hidden_states.extend(res)

        # 4. mid
        x = self.mid_block(
            x,
            emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )

        # 5. up path
        for idx, block in enumerate(self.up_blocks):
            if isinstance(block, CrossAttnUpBlock2D):
                x = block(
                    x,
                    emb,
                    res_hidden_states_list=res_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
            else:
                x = block(
                    x,
                    emb,
                    res_hidden_states_list=res_hidden_states,
                )

        # 6. Post
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x
    
