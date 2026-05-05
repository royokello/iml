import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
try:
    from transformers.utils import TransformersKwargs
except ImportError:
    from typing import TypedDict

    class TransformersKwargs(TypedDict, total=False):
        pass

from gemma4.models.output_capturing import capture_outputs
from gemma4.models.pretrained import Gemma4PreTrainedModel
from gemma4.models.vision.attention import Gemma4VisionAttention
from gemma4.models.vision.config import Gemma4VisionConfig
from gemma4.models.vision.embedding import Gemma4VisionPatchEmbedder
from gemma4.models.vision.encoder import Gemma4VisionEncoder, Gemma4VisionEncoderLayer
from gemma4.models.vision.pooler import Gemma4VisionPooler


def auto_docstring(obj=None, **kwargs):
    if obj is None:
        return lambda wrapped: wrapped
    return obj


def merge_with_config_defaults(fn):
    return fn


class Gemma4VisionModel(Gemma4PreTrainedModel):
    """The Gemma 4 Vision Encoder."""

    config = Gemma4VisionConfig
    _can_record_outputs = {
        "hidden_states": Gemma4VisionEncoderLayer,
        "attentions": Gemma4VisionAttention,
    }

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__(config)
        self.patch_embedder = Gemma4VisionPatchEmbedder(config)
        self.encoder = Gemma4VisionEncoder(config)
        self.pooler = Gemma4VisionPooler(config)

        if self.config.standardize:
            self.register_buffer("std_bias", torch.empty(self.config.hidden_size))
            self.register_buffer("std_scale", torch.empty(self.config.hidden_size))

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring(custom_intro="Encodes image pixels to soft tokens from patches.")
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_position_ids: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor` or `list[torch.FloatTensor]`):
            The images to encode. Either a single `[batch, channels, height, width]` tensor
            (all images same size) or a list of `[1, channels, height, width]` tensors (different sizes).
        pixel_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`):
            The patch positions as (x, y) coordinates in the image. Padding patches are indicated by (-1, -1).
        """
        pooling_kernel_size = self.config.pooling_kernel_size
        output_length = pixel_values.shape[-2] // (pooling_kernel_size * pooling_kernel_size)

        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        inputs_embeds = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)
        output = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=~padding_positions,  # encoder expects True=valid, padding_positions is True=padding
            pixel_position_ids=pixel_position_ids,
            **kwargs,
        )

        hidden_states, pooler_mask = self.pooler(
            hidden_states=output.last_hidden_state,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )

        # Strip padding tokens. pooler_mask is True = valid, False = padding.
        hidden_states = hidden_states[pooler_mask]

        if self.config.standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale

        return BaseModelOutputWithPast(last_hidden_state=hidden_states)
