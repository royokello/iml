from __future__ import annotations

import random

import torch

PROMPT_EMBEDS_KEY = "prompt_embeds"
POOLED_KEY = "pooled"


def split_tags(caption: str) -> list[str]:
    """Split a caption into tags.

    Args:
        caption: A comma-separated caption.

    Returns:
        The stripped tags. Empty entries are dropped.
    """
    return [tag.strip() for tag in caption.split(",") if tag.strip()]


def caption_has_tag(caption: str, tag: str) -> bool:
    """Return True when a caption holds an exact tag.

    Args:
        caption: A comma-separated caption.
        tag: The tag to look for.

    Returns:
        True when the tag is present after stripping.
    """
    return tag in split_tags(caption)


def drop_tags(
    caption: str,
    *,
    trigger: str,
    rate: float,
    rng: random.Random,
) -> str:
    """Return a caption with random tags removed.

    The trigger tag is always kept. Every other tag is kept with probability
    `1 - rate`.

    Args:
        caption: A comma-separated caption.
        trigger: The tag that must survive. It must be in the caption.
        rate: The probability that an eligible tag is removed.
        rng: The random source. Pass one instance per image so a rebuild
            reproduces the same captions.

    Returns:
        The remaining tags, joined with `", "`, in their original order.

    Raises:
        ValueError: If the caption does not hold the trigger, or the rate is
            outside `[0, 1)`.
    """
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"Tag dropout rate must be in [0, 1), got {rate}.")

    tags = split_tags(caption)
    if trigger not in tags:
        raise ValueError(f"Caption does not hold the trigger {trigger!r}: {caption!r}")

    return ", ".join(tag for tag in tags if tag == trigger or rng.random() >= rate)


def build_dropout_captions(
    caption: str,
    *,
    trigger: str,
    rate: float,
    count: int,
    seed: str,
) -> list[str]:
    """Return `count` captions with independent tag dropout.

    The seed fixes the result, so two builds of the same image produce the
    same list and the text cache can be reused.

    Args:
        caption: A comma-separated caption.
        trigger: The tag that must survive.
        rate: The probability that an eligible tag is removed.
        count: The number of captions to produce. At least one.
        seed: A stable seed string, unique per image.

    Returns:
        The generated captions, in a fixed order.

    Raises:
        ValueError: If `count` is below one.
    """
    if count < 1:
        raise ValueError(f"Caption count must be positive, got {count}.")

    rng = random.Random(seed)
    return [
        drop_tags(caption, trigger=trigger, rate=rate, rng=rng)
        for _ in range(count)
    ]


def encode_caption(
    prompt: str,
    tokenizer,
    tokenizer_2,
    text_encoder,
    text_encoder_2,
    device: torch.device | str,
    clip_skip: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one caption the way `sdxl.gen` encodes a prompt.

    The two hidden states concatenate on the last axis to 2048. The pooled
    vector comes from the final layer of `text_encoder_2`.

    Args:
        prompt: The caption.
        tokenizer: The CLIP-L tokenizer.
        tokenizer_2: The CLIP-G tokenizer.
        text_encoder: CLIP-L.
        text_encoder_2: CLIP-G.
        device: Where the encoders run.
        clip_skip: Layers to skip from the end. None selects the penultimate
            layer, which is SDXL's default.

    Returns:
        `(prompt_embeds, pooled)`, both fp16 on the CPU. `prompt_embeds` has
        shape `(77, 2048)` and `pooled` has shape `(1280,)`.

    Raises:
        ValueError: If the two encoders do not give widths that add up to 2048.
    """
    embeddings = []
    pooled = None

    with torch.no_grad():
        for current_tokenizer, current_encoder in (
            (tokenizer, text_encoder),
            (tokenizer_2, text_encoder_2),
        ):
            inputs = current_tokenizer(
                prompt,
                padding="max_length",
                max_length=current_tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            output = current_encoder(
                inputs.input_ids.to(device),
                output_hidden_states=True,
            )

            if pooled is None and output[0].ndim == 2:
                pooled = output[0]

            if clip_skip is None:
                hidden_states = output.hidden_states[-2]
            else:
                hidden_states = output.hidden_states[-(clip_skip + 2)]
            embeddings.append(hidden_states)

    prompt_embeds = torch.cat(embeddings, dim=-1)
    if prompt_embeds.shape[-1] != 2048:
        raise ValueError(
            f"Expected 2048 prompt embedding channels, got {prompt_embeds.shape[-1]}."
        )
    if pooled is None:
        raise ValueError("text_encoder_2 did not return a pooled embedding.")

    return (
        prompt_embeds.squeeze(0).to(device="cpu", dtype=torch.float16),
        pooled.squeeze(0).to(device="cpu", dtype=torch.float16),
    )


__all__ = [
    "POOLED_KEY",
    "PROMPT_EMBEDS_KEY",
    "build_dropout_captions",
    "caption_has_tag",
    "drop_tags",
    "encode_caption",
    "split_tags",
]
