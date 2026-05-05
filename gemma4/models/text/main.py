import torch
import torch.nn as nn

from transformers import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
try:
    from transformers.utils import TransformersKwargs
except ImportError:
    from typing import TypedDict

    class TransformersKwargs(TypedDict, total=False):
        pass

from gemma4.models.output_capturing import OutputRecorder, capture_outputs
from gemma4.models.pretrained import Gemma4PreTrainedModel

from gemma4.models.norm import Gemma4RMSNorm
from gemma4.models.text.attention import Gemma4TextAttention
from gemma4.models.text.config import Gemma4TextConfig
from gemma4.models.text.decoder import Gemma4TextDecoderLayer
from gemma4.models.text.rotary_embedding import Gemma4TextRotaryEmbedding
from gemma4.models.text.router import Gemma4TextRouter
from gemma4.models.text.scaled_word_embedding import Gemma4TextScaledWordEmbedding


def auto_docstring(obj=None, **kwargs):
    if obj is None:
        return lambda wrapped: wrapped
    return obj


def merge_with_config_defaults(fn):
    return fn


# @auto_docstring(custom_intro="The base Gemma 4 language model without a language modeling head.")
class Gemma4TextModel(Gemma4PreTrainedModel):
    config: Gemma4TextConfig
    input_modalities = ("text",)
    _can_record_outputs = {
        "router_logits": OutputRecorder(Gemma4TextRouter, index=0),
        "hidden_states": Gemma4TextDecoderLayer,
        "attentions": Gemma4TextAttention,
    }

    def __init__(self, config: Gemma4TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Gemma4 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            [Gemma4TextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4TextRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.unique_layer_types = set(self.config.layer_types)

        # Per-Layer Embeddings (PLE): auxiliary embedding that feeds a residual signal
        # into each decoder layer. See `get_per_layer_inputs()` and `project_per_layer_inputs()`
        # for the full pipeline. The embedding is packed: total dim = num_layers * per_layer_dim.
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                self.padding_idx,
                embed_scale=config.hidden_size_per_layer_input**0.5,
            )
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_model_projection_scale = config.hidden_size**-0.5
            self.per_layer_projection_norm = Gemma4RMSNorm(config.hidden_size_per_layer_input, eps=config.rms_norm_eps)

        # Update `_keys_to_ignore_on_load_unexpected` to drop all k/v proj and norms for the shared layers
        self._keys_to_ignore_on_load_unexpected = []
        for i, layer in enumerate(self.layers):
            if layer.self_attn.is_kv_shared_layer:
                self._keys_to_ignore_on_load_unexpected.extend(
                    [f"layers.{i}.self_attn.{name}" for name in ("k_proj", "v_proj", "k_norm", "v_norm")]
                )

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        per_layer_inputs (`torch.Tensor` of shape `(batch_size, sequence_length, num_hidden_layers, hidden_size_per_layer_input)`, *optional*):
            Pre-computed per-layer input embeddings. When provided, these are used directly instead of being
            computed from `input_ids` via `get_per_layer_inputs()`. This is primarily used by the multimodal
            model (`Gemma4Model`) which pre-computes per-layer inputs from the original `input_ids` *before*
            merging multimodal soft tokens into `inputs_embeds` — at which point the original token ids are
            no longer recoverable.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(input_ids, inputs_embeds)
            per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        # embed positions
        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        # Initialize as empty dict - it will be filled in the right layers
        shared_kv_states = {}

        # decoder layers
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None

            hidden_states = decoder_layer(
                hidden_states,
                per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def get_per_layer_inputs(self, input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None) -> torch.Tensor:
        """Compute the token-identity component of Per-Layer Embeddings (PLE).

        Looks up `input_ids` in `embed_tokens_per_layer` (a scaled embedding that multiplies
        by `sqrt(hidden_size_per_layer_input)`) and reshapes the packed output from
        `[batch, seq, num_hidden_layers * hidden_size_per_layer_input]` to
        `[batch, seq, num_hidden_layers, hidden_size_per_layer_input]`.

        If only `inputs_embeds` is provided (no `input_ids`), reverses the main embedding
        to recover `input_ids` for the PLE lookup.
        """
        if not self.hidden_size_per_layer_input:
            raise RuntimeError(
                "Attempting to call get_per_layer_inputs() from a model initialized with a config that does not support"
                f" per-layer embeddings. {self.config}"
            )

        # If only inputs_embeds are provided, reverse main embedding to find the input_ids - this allows to `generate`
        # from `inputs_embeds` only as other models (otherwise it would need the value from both embeddings)
        if input_ids is None:
            with torch.no_grad():
                input_ids = (
                    (
                        inputs_embeds[:, :, None, :]
                        == self.embed_tokens.weight[None, None, :, :] * self.config.hidden_size**0.5
                    )
                    .all(dim=3)
                    .nonzero()[:, 2]
                )
                try:
                    input_ids = input_ids.view(inputs_embeds.shape[:2])
                except RuntimeError:
                    raise RuntimeError(
                        "It seems like you tried to call `forward` from `inputs_embeds` without providing `input_ids`, and that "
                        "the `inputs_embeds` you provided do not exactly match the embedding weights. Since Gemma4 needs to reverse "
                        "the embedding to compute another embedding, make sure you provide exact `inputs_embeds`"
                    )

        return self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the context-aware component of PLE and combine with token-identity.

        Projects `inputs_embeds` through `per_layer_model_projection` (Linear), scales by
        `1/sqrt(hidden_size)`, reshapes to `[batch, seq, num_layers, ple_dim]`, and normalizes
        with `per_layer_projection_norm` (RMSNorm).

        If `per_layer_inputs` (the token-identity component from `get_per_layer_inputs()`)
        is provided, combines both: `(context_projection + token_identity) * (1/sqrt(2))`.
        If `per_layer_inputs` is None (e.g. for multimodal inputs where input_ids are not
        available), returns just the context projection.
        """
        if not self.hidden_size_per_layer_input:
            raise RuntimeError(
                "Attempting to call project_per_layer_inputs() from a model initialized with a config that does not"
                f" support per-layer embeddings. {self.config}"
            )

        per_layer_projection = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale
