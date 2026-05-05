import torch
import torch.nn as nn
from transformers.processing_utils import Unpack
try:
    from transformers.utils import TransformersKwargs
except ImportError:
    from typing import TypedDict

    class TransformersKwargs(TypedDict, total=False):
        pass

from gemma4.models.audio.config import Gemma4AudioConfig
from gemma4.models.output_capturing import capture_outputs
from gemma4.models.pretrained import Gemma4PreTrainedModel


def auto_docstring(obj=None, **kwargs):
    if obj is None:
        return lambda wrapped: wrapped
    return obj


def merge_with_config_defaults(fn):
    return fn

class Gemma4AudioModel(Gemma4PreTrainedModel):
    """An audio encoder based on the [Universal Speech Model](https://huggingface.co/papers/2303.01037) architecture."""

    config: Gemma4AudioConfig
    main_input_name = "input_features"
    base_model_prefix = "model.audio_tower"  # prefix for Gemma4ForConditionalGeneration saved checkpoints, required for Gemma4AudioModel.from_pretrained()
    _can_record_outputs = {
        "hidden_states": Gemma4AudioLayer,
        "attentions": Gemma4AudioAttention,
    }

    def __init__(self, config: Gemma4AudioConfig):
        super().__init__(config)
        self.config = config

        self.subsample_conv_projection = Gemma4AudioSubSampleConvProjection(config)
        self.rel_pos_enc = Gemma4AudioRelPositionalEncoding(config)
        self.layers = nn.ModuleList(
            [Gemma4AudioLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.output_proj = nn.Linear(config.hidden_size, config.output_proj_dims, bias=True)

        self.post_init()

    def _convert_4d_mask_to_blocked_5d(self, mask_4d: torch.Tensor) -> torch.Tensor:
        """
        Convert a standard 4D attention mask `[batch_size, 1, seq_len, seq_len]` to the 5D blocked format
        `[batch_size, 1, num_blocks, chunk_size, context_size]` expected by the chunked local attention,
        """
        batch_size, _, seq_len, _ = mask_4d.shape
        device = mask_4d.device

        chunk_size = self.config.attention_chunk_size
        max_past_horizon = self.config.attention_context_left - 1
        max_future_horizon = self.config.attention_context_right

        num_blocks = (seq_len + chunk_size - 1) // chunk_size
        padded_seq_len = num_blocks * chunk_size
        pad_amount = padded_seq_len - seq_len

        mask_4d = F.pad(mask_4d, (0, pad_amount, 0, pad_amount), value=False)
        mask_5d = mask_4d.reshape(batch_size, 1, num_blocks, chunk_size, padded_seq_len)
        mask_5d = F.pad(mask_5d, (max_past_horizon, max_future_horizon), value=False)

        block_starts = torch.arange(num_blocks, device=device) * chunk_size
        offsets = torch.arange(chunk_size + max_past_horizon + max_future_horizon, device=device)
        kv_indices = block_starts[:, None] + offsets[None, :]
        kv_indices = kv_indices[None, None, :, None, :].expand(batch_size, 1, -1, chunk_size, -1)

        return mask_5d.gather(-1, kv_indices)

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring(custom_intro="Encodes audio features to soft tokens.")
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        hidden_states, output_mask = self.subsample_conv_projection(input_features, attention_mask)
        position_embeddings = self.rel_pos_enc(hidden_states)

        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=output_mask,
            and_mask_function=sliding_window_mask_function(
                (self.config.attention_context_left - 1, self.config.attention_context_right)
            ),
        )
        attention_mask = self._convert_4d_mask_to_blocked_5d(attention_mask)

        for encoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.output_proj(hidden_states)
        return Gemma4AudioModelOutput(last_hidden_state=hidden_states, attention_mask=output_mask)
