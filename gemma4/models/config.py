from typing import Any

from transformers.configuration_utils import PretrainedConfig

def auto_docstring(*args, **kwargs):
    def decorator(obj):
        return obj
    return decorator

from gemma4.models.audio.config import Gemma4AudioConfig
from gemma4.models.text.config import Gemma4TextConfig
from gemma4.models.vision.config import Gemma4VisionConfig


def strict(obj):
    return obj

@strict
class Gemma4Config(PretrainedConfig):
    r"""
    boi_token_id (`int`, *optional*, defaults to 255999):
        The begin-of-image token index to wrap the image prompt.
    eoi_token_id (`int`, *optional*, defaults to 258882):
        The end-of-image token index to wrap the image prompt.
    boa_token_id (`int`, *optional*, defaults to 256000):
        The begin-of-audio token index to wrap the audio prompt.
    eoa_token_index (`int`, *optional*, defaults to 258883):
        The end-of-audio token index to wrap the audio prompt.

    Example:

    ```python
    >>> from transformers import (
    >>>     Gemma4AudioConfig,
    >>>     Gemma4Config,
    >>>     Gemma4ForConditionalGeneration,
    >>>     Gemma4TextConfig,
    >>>     Gemma4VisionConfig,
    >>> )

    >>> # Initializing a Gemma 4 Audio config.
    >>> audio_config = Gemma4AudioConfig()

    >>> # Initializing a Gemma 4 Text config.
    >>> text_config = Gemma4TextConfig()

    >>> # Initializing a Gemma 4 vision config.
    >>> vision_config = Gemma4VisionConfig()

    >>> # Initializing a Gemma 4 config similar to google/gemma-4-e2b-it
    >>> configuration = Gemma4Config(text_config, vision_config, audio_config)

    >>> # Initializing a model from the google/gemma-4-e2b-it configuration
    >>> model = Gemma4ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gemma4"
    sub_configs = {
        "text_config": Gemma4TextConfig,
        "vision_config": Gemma4VisionConfig,
        "audio_config": Gemma4AudioConfig,
    }

    text_config: Gemma4TextConfig | dict[str, Any] | None = None
    vision_config: Gemma4VisionConfig | dict[str, Any] | None = None
    audio_config: Gemma4AudioConfig | dict[str, Any] | None = None
    boi_token_id: int | None = 255_999
    eoi_token_id: int | None = 258_882
    image_token_id: int | None = 258_880
    video_token_id: int | None = 258_884
    boa_token_id: int | None = 256_000
    eoa_token_index: int | None = 258_883
    audio_token_id: int | None = 258_881
    initializer_range: float | None = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            raise ValueError("text_config must be provided.")
        elif isinstance(self.text_config, dict):
            self.text_config = Gemma4TextConfig(**self.text_config)

        if self.vision_config is None:
            raise ValueError("vision_config must be provided.")
        if isinstance(self.vision_config, dict):
            self.vision_config = Gemma4VisionConfig(**self.vision_config)

        if self.audio_config is None:
            raise ValueError("audio_config must be provided.")
        if isinstance(self.audio_config, dict):
            self.audio_config = Gemma4AudioConfig(**self.audio_config)

        super().__post_init__(**kwargs)
