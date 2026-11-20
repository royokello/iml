from __future__ import annotations

import torch

PROMPT_EMBEDS_KEY = "prompt_embeds"
POOLED_KEY = "pooled"


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


__all__ = ["POOLED_KEY", "PROMPT_EMBEDS_KEY", "encode_caption"]
