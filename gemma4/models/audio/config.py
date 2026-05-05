from transformers.configuration_utils import PretrainedConfig

def auto_docstring(*args, **kwargs):
    def decorator(obj):
        return obj
    return decorator


def strict(obj):
    return obj


@auto_docstring(checkpoint="google/gemma-4-e2b-it")
@strict
class Gemma4AudioConfig(PretrainedConfig):
    r"""
    subsampling_conv_channels (`list[int]`, defaults to `[128, 32]`):
        Channel sizes for the convolutional layers in the Sub-sample Convolution Projection.
    residual_weight (`float`, defaults to `0.5`):
        Scaling applied to hidden_states prior to combining with the residual in the feedforward.
    attention_chunk_size (`int`, defaults to `12`):
        The sub-sequence size for attention processing.
    attention_context_left (`int`, defaults to `13`):
        The leftward context size for the attention chunk.
    attention_context_right (`int`, defaults to `0`):
        The rightward context size for the attention chunk.
    attention_logit_cap (`float`, defaults to `50.0`):
        Cap applied to attention weights.
    attention_invalid_logits_value (`float`, defaults to `1e-9`):
        Value to use for invalid logits in attention.
    use_clipped_linears (`bool`, defaults to `True`):
        If true, apply clipping to the Linear layers, drawing bounds from the model checkpoint.
    gradient_clipping (`float`, defaults to `1e10`):
        Clipping value used to stabilize extremely large gradient values.
    output_proj_dims (`int`, defaults to `1536`):
        Dimension of the final linear projection from `hidden_size` to the model's output.
    """

    model_type = "gemma4_audio"

    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    hidden_act: str = "silu"

    # subsampling parameters
    subsampling_conv_channels: list[int] | tuple[int, int] = (128, 32)

    # conformer parameters
    conv_kernel_size: int = 5
    residual_weight: float = 0.5
    attention_chunk_size: int = 12
    attention_context_left: int = 13
    attention_context_right: int = 0
    attention_logit_cap: float = 50.0
    attention_invalid_logits_value: float = -1.0e9

    use_clipped_linears: bool = True
    rms_norm_eps: float = 1e-6
    gradient_clipping: float = 1e10
    output_proj_dims: int = 1536
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        # JSON serialization converts tuples to lists, convert back
        if isinstance(self.subsampling_conv_channels, tuple):
            self.subsampling_conv_channels = list(self.subsampling_conv_channels)
        super().__post_init__(**kwargs)
