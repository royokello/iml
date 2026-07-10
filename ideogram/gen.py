#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import traceback
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoTokenizer
from transformers.masking_utils import create_causal_mask

from ideogram.autoencoder import AutoEncoder, AutoEncoderParams, convert_diffusers_state_dict
from ideogram.constants import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    SEQUENCE_PADDING_INDICATOR,
    QWEN3_VL_ACTIVATION_LAYERS,
)
from ideogram.denoiser.loader import load_ideogram_transformer_pair, pin_module_parameters
from ideogram.latent_norm import get_latent_norm
from ideogram.scheduler import get_schedule_for_resolution, make_step_intervals
from ideogram.text_encoder.loader import load_ideogram_text_encoder


def _tokenize(tokenizer, prompt: str, max_text_tokens: int) -> tuple[torch.Tensor, int]:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    token_ids = encoded["input_ids"][0]
    num_text_tokens = int(token_ids.shape[0])
    if num_text_tokens > max_text_tokens:
        raise ValueError(f"prompt has {num_text_tokens} tokens, exceeds max_text_tokens={max_text_tokens}")
    return token_ids, num_text_tokens


def _build_inputs(
    tokenizer,
    prompt: str,
    height: int,
    width: int,
    patch_size: int,
    ae_scale_factor: int,
    max_text_tokens: int,
    device: torch.device,
) -> dict:
    token_ids, num_text_tokens = _tokenize(tokenizer, prompt, max_text_tokens)

    patch = patch_size * ae_scale_factor
    if height % patch != 0 or width % patch != 0:
        raise ValueError(f"height/width must be divisible by patch_size*ae_scale_factor={patch}")
    grid_h = height // patch
    grid_w = width // patch
    num_image_tokens = grid_h * grid_w
    total_seq_len = max_text_tokens + num_image_tokens

    h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
    w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
    t_idx = torch.zeros_like(h_idx)
    image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

    pad_len = max_text_tokens - num_text_tokens
    offset = pad_len

    token_ids_padded = torch.zeros(total_seq_len, dtype=torch.long)
    token_ids_padded[offset: offset + num_text_tokens] = token_ids

    text_pos = torch.arange(num_text_tokens)
    text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
    position_ids = torch.zeros(total_seq_len, 3, dtype=torch.long)
    position_ids[offset: offset + num_text_tokens] = text_pos_3d
    position_ids[offset + num_text_tokens:] = image_pos

    indicator = torch.zeros(total_seq_len, dtype=torch.long)
    indicator[offset: offset + num_text_tokens] = LLM_TOKEN_INDICATOR
    indicator[offset + num_text_tokens:] = OUTPUT_IMAGE_INDICATOR

    segment_ids = torch.full((total_seq_len,), SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
    segment_ids[offset: offset + num_text_tokens + num_image_tokens] = 1

    return {
        "token_ids": token_ids_padded.unsqueeze(0).to(device),
        "position_ids": position_ids.unsqueeze(0).to(device),
        "segment_ids": segment_ids.unsqueeze(0).to(device),
        "indicator": indicator.unsqueeze(0).to(device),
        "num_image_tokens": num_image_tokens,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "max_text_tokens": max_text_tokens,
    }


def _encode_text(language_model, token_ids, position_ids, indicator) -> torch.Tensor:
    batch_size, seq_len = token_ids.shape

    attention_mask = (indicator == LLM_TOKEN_INDICATOR).to(torch.long)
    pos_2d = position_ids[..., 0].contiguous()

    is_offloaded = language_model._offload_device is not None
    if is_offloaded:
        language_model._move_non_layers_to("cuda")

    inputs_embeds = language_model.embed_tokens(token_ids)
    if is_offloaded:
        language_model.embed_tokens.to("cpu")

    position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
    text_position_ids = position_ids_4d[0]
    mrope_position_ids = position_ids_4d[1:]

    causal_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)

    if is_offloaded:
        inputs_embeds = inputs_embeds.to("cuda")
        if causal_mask is not None:
            causal_mask = causal_mask.to("cuda")
        cos, sin = position_embeddings
        position_embeddings = (cos.to("cuda"), sin.to("cuda"))
        attention_mask = attention_mask.to("cuda")

    tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
    captured = {}
    hidden_states = inputs_embeds

    with torch.inference_mode():
        for layer_idx, decoder_layer in enumerate(language_model.layers):
            if is_offloaded:
                pinned_params = {k: p for k, p in decoder_layer.named_parameters()}
                pinned_buffers = {k: b for k, b in decoder_layer.named_buffers() if b is not None}
                decoder_layer.to("cuda")
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                position_embeddings=position_embeddings,
            )
            if layer_idx in tap_set:
                captured[layer_idx] = hidden_states
            if is_offloaded:
                for name, p in pinned_params.items():
                    path, _, key = name.rpartition(".")
                    parent = decoder_layer.get_submodule(path) if path else decoder_layer
                    old = parent._parameters[key]
                    parent._parameters[key] = p
                    del old
                for name, b in pinned_buffers.items():
                    path, _, key = name.rpartition(".")
                    parent = decoder_layer.get_submodule(path) if path else decoder_layer
                    old = parent._buffers[key]
                    parent._buffers[key] = b
                    del old

    if is_offloaded:
        language_model._move_non_layers_to("cpu")

    selected = [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]
    stacked = torch.stack(selected, dim=0)
    stacked = stacked.permute(1, 2, 3, 0).reshape(batch_size, seq_len, -1)
    text_mask = attention_mask.to(stacked.dtype).unsqueeze(-1)
    return (stacked * text_mask).to(torch.float32)


def _decode(z, grid_h, grid_w, patch_size, autoencoder, latent_shift, latent_scale, dtype) -> list[Image.Image]:
    z = z * latent_scale + latent_shift
    batch_size = z.shape[0]
    ae_channels = z.shape[-1] // (patch_size * patch_size)
    z = z.view(batch_size, grid_h, grid_w, patch_size, patch_size, ae_channels)
    z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
    z = z.view(batch_size, ae_channels, grid_h * patch_size, grid_w * patch_size)
    z = z.to(dtype)
    decoded = autoencoder.decoder(z)
    decoded = decoded.float().clamp(-1.0, 1.0)
    decoded = ((decoded + 1.0) * 127.5).round().to(torch.uint8)
    decoded = decoded.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(arr) for arr in decoded]


def generate_image(
    root: str | Path,
    *,
    prompt: str = "A cat holding a sign that says hello world",
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 128,
    guidance_scale: float = 7.0,
    seed: int | None = None,
    text_quant_method: str | None = "sym-high",
    denoiser_quant_method: str | None = "sym-med",
    offloading: bool = False,
) -> list[Image.Image]:
    root = Path(root).expanduser().resolve()
    if not torch.cuda.is_available():
        print("CUDA not available, cannot run Ideogram generation")
        return []
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    ideogram_dir = root / "ideogram"
    text_encoder_dir = ideogram_dir / "text_encoder"
    tokenizer_dir = ideogram_dir / "tokenizer"
    vae_path = ideogram_dir / "vae" / "diffusion_pytorch_model.safetensors"

    patch_size = 2
    ae_scale_factor = 8
    max_text_tokens = 2048

    # ---- Phase 1: Build inputs ----
    print("Phase 1: Tokenizing and building inputs ...")
    p1 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    inputs = _build_inputs(
        tokenizer, prompt, height, width,
        patch_size=patch_size, ae_scale_factor=ae_scale_factor,
        max_text_tokens=max_text_tokens, device="cuda",
    )
    del tokenizer
    print(f"  grid={inputs['grid_h']}x{inputs['grid_w']}, "
          f"image_tokens={inputs['num_image_tokens']}, "
          f"max_text_tokens={inputs['max_text_tokens']}")
    print(f"  Done in {time.perf_counter() - p1:.3f}s")

    # ---- Phase 2: Encode text ----
    print("Phase 2: Encoding text with Qwen3-VL ...")
    p2 = time.perf_counter()
    try:
        language_model = load_ideogram_text_encoder(
            text_encoder_dir / "model.safetensors",
            quant_method=text_quant_method,
            config_path=text_encoder_dir,
            offloading=offloading,
        )
        if offloading:
            language_model.enable_block_offload("cuda")
        else:
            language_model = language_model.to("cuda")
        language_model.eval()
        llm_features = _encode_text(
            language_model, inputs["token_ids"], inputs["position_ids"], inputs["indicator"]
        )
        language_model = language_model.cpu()
        del language_model
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:
        traceback.print_exc()
        raise
    print(f"  Done in {time.perf_counter() - p2:.3f}s")

    # ---- Phase 3: Init latent ----
    print("Phase 3: Initializing latent and noise schedule ...")
    p3 = time.perf_counter()
    num_image_tokens = inputs["num_image_tokens"]
    max_text = inputs["max_text_tokens"]
    latent_dim = 128

    generator = torch.Generator(device="cuda")
    if seed is not None:
        generator.manual_seed(seed)
    z = torch.randn(1, num_image_tokens, latent_dim, dtype=torch.float32, device="cuda", generator=generator)
    text_z_padding = torch.zeros(1, max_text, latent_dim, dtype=torch.float32, device="cuda")

    schedule = get_schedule_for_resolution((height, width), known_mean=0.5, std=1.0)
    step_intervals = make_step_intervals(num_inference_steps).to("cuda")
    gw_per_step = torch.full((num_inference_steps,), guidance_scale, dtype=torch.float32, device="cuda")
    print(f"  steps={num_inference_steps}, guidance={guidance_scale}")
    print(f"  Done in {time.perf_counter() - p3:.3f}s")

    # ---- Phase 4: Denoise ----
    print("Phase 4: Denoising with Ideogram 4 transformer pair ...")
    p4 = time.perf_counter()
    cond_model, uncond_model = load_ideogram_transformer_pair(
        ideogram_dir, quant_method=denoiser_quant_method,
        offloading=offloading,
    )
    if offloading:
        n_pinned = pin_module_parameters(cond_model) + pin_module_parameters(uncond_model)
        print(f"  pinned {n_pinned} tensors to host memory")
        cond_model.enable_block_offload("cuda")
        uncond_model.enable_block_offload("cuda")
        cond_model.eval()
        uncond_model.eval()
    else:
        cond_model.eval()
        uncond_model.eval()

    neg_position_ids = inputs["position_ids"][:, max_text:]
    neg_segment_ids = inputs["segment_ids"][:, max_text:]
    neg_indicator = inputs["indicator"][:, max_text:]
    neg_llm_features = torch.zeros(
        1, num_image_tokens, llm_features.shape[-1],
        dtype=llm_features.dtype, device="cuda",
    )

    with torch.inference_mode():
        for i in range(num_inference_steps - 1, -1, -1):
            step_start = time.perf_counter()
            try:
                t = schedule(step_intervals[i + 1].unsqueeze(0)).float()
                s_val = schedule(step_intervals[i].unsqueeze(0)).item()

                if not offloading:
                    cond_model = cond_model.to("cuda")
                pos_z = torch.cat([text_z_padding, z], dim=1)
                pos_out = cond_model(
                    llm_features=llm_features,
                    x=pos_z,
                    t=t,
                    position_ids=inputs["position_ids"],
                    segment_ids=inputs["segment_ids"],
                    indicator=inputs["indicator"],
                )
                pos_v = pos_out[:, max_text:].clone()
                del pos_out
                if not offloading:
                    cond_model = cond_model.to("cpu")
                torch.cuda.empty_cache()

                if not offloading:
                    uncond_model = uncond_model.to("cuda")
                neg_out = uncond_model(
                    llm_features=neg_llm_features,
                    x=z,
                    t=t,
                    position_ids=neg_position_ids,
                    segment_ids=neg_segment_ids,
                    indicator=neg_indicator,
                )
                neg_v = neg_out.clone()
                del neg_out
                if not offloading:
                    uncond_model = uncond_model.to("cpu")
                torch.cuda.empty_cache()

                gw_i = gw_per_step[i]
                v = gw_i * pos_v + (1.0 - gw_i) * neg_v
                delta = s_val - t.item()
                z = z + v * delta

                gc.collect()
                torch.cuda.empty_cache()

                step_seconds = time.perf_counter() - step_start
                print(f"  * {num_inference_steps - i}/{num_inference_steps} steps", flush=True)
                print(f"    * done in {step_seconds:.3f}s", flush=True)
            except Exception:
                traceback.print_exc()
                print(f"  step {num_inference_steps - i}/{num_inference_steps} FAILED at i={i}")
                raise

    cond_model = cond_model.cpu()
    uncond_model = uncond_model.cpu()
    del cond_model, uncond_model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print(f"  Done in {time.perf_counter() - p4:.3f}s")

    # ---- Phase 5: Decode ----
    print("Phase 5: Decoding latents with VAE ...")
    p5 = time.perf_counter()
    shift, scale = get_latent_norm()
    latent_shift = shift.to("cuda")
    latent_scale = scale.to("cuda")

    ae = AutoEncoder(AutoEncoderParams())
    state_dict = convert_diffusers_state_dict(load_file(str(vae_path)))
    ae.load_state_dict(state_dict)
    del state_dict
    gc.collect()
    torch.cuda.empty_cache()
    ae = ae.to("cuda").eval()

    images = _decode(
        z, grid_h=inputs["grid_h"], grid_w=inputs["grid_w"],
        patch_size=patch_size, autoencoder=ae,
        latent_shift=latent_shift, latent_scale=latent_scale,
        dtype=torch.float32,
    )
    ae = ae.cpu()
    del ae
    torch.cuda.empty_cache()
    print(f"  Done in {time.perf_counter() - p5:.3f}s")

    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ideogram 4 text-to-image generation.")
    parser.add_argument("--root", required=True, help="Root folder containing ideogram/ subdirectory.")
    parser.add_argument("--prompt", default="A cat holding a sign that says hello world", help="Text prompt.")
    parser.add_argument("--output", default="output.png", help="Output image path.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--steps", type=int, default=128, help="Number of denoising steps.")
    parser.add_argument("--guidance-scale", type=float, default=7.0, help="CFG guidance scale.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed.")
    parser.add_argument("--text-quant-method", default="sym-high", help="Text encoder quantization method.")
    parser.add_argument("--denoiser-quant-method", default="sym-med", help="Denoiser quantization method.")
    parser.add_argument("--offload", action="store_true", help="Enable block offloading with pinned host memory. Keeps weights on CPU and streams blocks to GPU one at a time.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = generate_image(
        args.root,
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        text_quant_method=args.text_quant_method,
        denoiser_quant_method=args.denoiser_quant_method,
        offloading=args.offload,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(str(output))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
