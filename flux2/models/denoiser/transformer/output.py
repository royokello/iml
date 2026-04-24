from dataclasses import dataclass

import numpy as np
import PIL.Image

from diffusers.utils import BaseOutput

from ..kv.cache import Flux2KVCache


@dataclass
class Flux2Transformer2DModelOutput(BaseOutput):
    """
    The output of [`Flux2Transformer2DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on the `encoder_hidden_states` input.
        kv_cache (`Flux2KVCache`, *optional*):
            The populated KV cache for reference image tokens. Only returned when `kv_cache_mode="extract"`.
    """

    sample: "torch.Tensor"  # noqa: F821
    kv_cache: "Flux2KVCache | None" = None
