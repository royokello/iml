import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
try:
    from transformers.utils import TransformersKwargs
except ImportError:
    from typing import TypedDict

    class TransformersKwargs(TypedDict, total=False):
        pass

from gemma4.models.pretrained import Gemma4PreTrainedModel
from gemma4.models.text.config import Gemma4TextConfig
from gemma4.models.text.main import Gemma4TextModel


class Gemma4TiedEmbeddingLMHead(nn.Module):
    def __init__(self, embed_tokens: nn.Module) -> None:
        super().__init__()
        self.embed_tokens = embed_tokens
        self.out_features = embed_tokens.num_embeddings
        self.in_features = embed_tokens.embedding_dim

    @property
    def weight(self):
        return self.embed_tokens.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hasattr(self.embed_tokens, "dequantize_weight"):
            weight = self.embed_tokens.dequantize_weight()
        else:
            weight = self.embed_tokens.weight
        return F.linear(hidden_states, weight.to(dtype=hidden_states.dtype))


def auto_docstring(obj=None, **kwargs):
    if obj is None:
        return lambda wrapped: wrapped
    return obj

class Gemma4ForCausalLM(Gemma4PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config: Gemma4TextConfig
    base_model_prefix = "model"

    def __init__(self, config: Gemma4TextConfig, model: Gemma4TextModel):
        super().__init__(config)
        self.model = model
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Grab the ones from the child
        self._keys_to_ignore_on_load_unexpected = [
            f"model.{name}" for name in self.model._keys_to_ignore_on_load_unexpected
        ]

        # Keep the already-loaded base model intact. Quantized embeddings
        # store packed integer buffers, so Transformers' generic parameter
        # tying cannot assign them into lm_head.weight.
        embed_weight = self.model.embed_tokens.weight
        print(
            "    lm_head tie source "
            f"type={type(self.model.embed_tokens).__name__} "
            f"weight_type={type(embed_weight).__name__} "
            f"dtype={getattr(embed_weight, 'dtype', None)}"
        )
        if isinstance(embed_weight, nn.Parameter) and torch.is_floating_point(embed_weight):
            self.tie_weights()
        else:
            self.lm_head = Gemma4TiedEmbeddingLMHead(self.model.embed_tokens)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Gemma4ForCausalLM

        >>> model = Gemma4ForCausalLM.from_pretrained("google/gemma-2-9b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
