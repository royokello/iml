import math

import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import ROPE_INIT_FUNCTIONS, PreTrainedModel
from transformers.utils import is_accelerate_available
from gemma4.models.audio.attention import Gemma4AudioAttention
from gemma4.models.audio.encoding import Gemma4AudioRelPositionalEncoding
from gemma4.models.config import Gemma4Config
from gemma4.models.linear import Gemma4ClippableLinear
from gemma4.models.text.decoder import Gemma4TextDecoderLayer
from gemma4.models.text.experts import Gemma4TextExperts
from gemma4.models.text.rotary_embedding import Gemma4TextRotaryEmbedding
from gemma4.models.text.router import Gemma4TextRouter
from gemma4.models.text.scaled_word_embedding import Gemma4TextScaledWordEmbedding
from gemma4.models.vision.embedding import Gemma4VisionPatchEmbedder, Gemma4VisionRotaryEmbedding

if is_accelerate_available():
    from accelerate.hooks import add_hook_to_module

# @auto_docstring
class Gemma4PreTrainedModel(PreTrainedModel):
    config: Gemma4Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Gemma4TextDecoderLayer", "Gemma4VisionEncoderLayer", "Gemma4AudioLayer"]
    _skip_keys_device_placement = ["past_key_values", "shared_kv_states"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = None  # override
    input_modalities = ("image", "text", "video", "audio")

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Gemma4VisionPatchEmbedder):
            init.ones_(module.position_embedding_table)
        elif isinstance(module, Gemma4AudioRelPositionalEncoding):
            min_timescale = 1.0
            max_timescale = 10000.0
            num_timescales = module.hidden_size // 2
            log_timescale_increment = math.log(max_timescale / min_timescale) / max(num_timescales - 1, 1)
            inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales) * -log_timescale_increment)
            init.copy_(module.inv_timescales, inv_timescales.unsqueeze(0).unsqueeze(0))
        elif isinstance(module, Gemma4AudioAttention):
            init.constant_(module.softcap, module.attention_logits_soft_cap)
            init.zeros_(module.per_dim_scale)
        elif isinstance(module, Gemma4TextRotaryEmbedding):
            for layer_type, rope_init_fn in module.rope_init_fns.items():
                rope_init_fn_kwargs = {"layer_type": layer_type}
                if layer_type == "full_attention" and module.rope_type[layer_type] == "proportional":
                    rope_init_fn_kwargs["head_dim_key"] = "global_head_dim"

                curr_inv_freq, _ = rope_init_fn(module.config, **rope_init_fn_kwargs)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)
        elif isinstance(module, Gemma4VisionRotaryEmbedding):
            rope_fn = (
                ROPE_INIT_FUNCTIONS[module.rope_type]
                if module.rope_type != "default"
                else module.compute_default_rope_parameters
            )
            buffer_value, _ = rope_fn(module.config)
            init.copy_(module.inv_freq, buffer_value)
            init.copy_(module.original_inv_freq, buffer_value)
        elif isinstance(module, Gemma4TextScaledWordEmbedding):
            init.constant_(module.embed_scale, module.scalar_embed_scale)
        elif isinstance(module, Gemma4TextRouter):
            init.ones_(module.scale)
            init.ones_(module.per_expert_scale)
        elif isinstance(module, Gemma4TextExperts):
            std = self.config.initializer_range
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, Gemma4TextDecoderLayer):
            init.ones_(module.layer_scalar)
        elif isinstance(module, Gemma4ClippableLinear) and module.use_clipped_linears:
            init.constant_(module.input_min, -float("inf"))
            init.constant_(module.input_max, float("inf"))
            init.constant_(module.output_min, -float("inf"))
            init.constant_(module.output_max, float("inf"))
        elif module.__class__.__name__ == "Gemma4VisionModel" and module.config.standardize:
            init.zeros_(module.std_bias)
            init.ones_(module.std_scale)

    def get_per_layer_input_embeddings(self):
        return self.base_model.embed_tokens_per_layer

    def set_per_layer_input_embeddings(self, value):
        self.base_model.embed_tokens_per_layer = value

    def resize_token_embeddings(
        self,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        inputs_embeds = super().resize_token_embeddings(
            new_num_tokens=new_num_tokens,
            pad_to_multiple_of=pad_to_multiple_of,
            mean_resizing=mean_resizing,
        )
        self._resize_per_layer_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        return inputs_embeds

    def _resize_per_layer_embeddings(
        self,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = True,
    ):
        self.config.get_text_config().vocab_size_per_layer_input = self.vocab_size
        if self.config.get_text_config().hidden_size_per_layer_input:
            embed_tokens_per_layer = self.get_per_layer_input_embeddings()
            new_embeddings_per_layer = self._get_resized_embeddings(
                embed_tokens_per_layer, new_num_tokens, pad_to_multiple_of, mean_resizing
            )
            if hasattr(embed_tokens_per_layer, "_hf_hook"):
                hook = embed_tokens_per_layer._hf_hook
                add_hook_to_module(new_embeddings_per_layer, hook)
            new_embeddings_per_layer.requires_grad_(embed_tokens_per_layer.weight.requires_grad)
            self.set_per_layer_input_embeddings(new_embeddings_per_layer)
