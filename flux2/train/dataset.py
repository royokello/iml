from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple
import time
from PIL import Image
from diffusers import AutoencoderKLFlux2
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
import torch
from transformers import Qwen2TokenizerFast
from safetensors.torch import load_file, save_file

from flux2.gen import _encode_prompt_embeddings, _load_vae_scale_factor, _pack_latents, _patchify_latents, _prepare_latent_ids, _retrieve_latents
from flux2.text_encoder.loader import _load_flux2_text_encoder as load_flux2_text_encoder
from flux2.train.images import prepare_image_latent

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _list_image_files(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.iterdir() if _is_supported_image(path)),
        key=lambda path: path.name,
    )

def _encode_images(
    sample_ids: list[str],
    image_dir: Path,
    vae,
    processor,
    vae_scale_factor: int,
    *,
    cache_images: bool,
    load_targets: bool,
) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
    """Encode images from image_dir into VAE latents.

    For each sample_id, looks for {image_dir}/{sample_id}.{ext}. If not found,
    appends (None, None) for that slot. When load_targets is False, all entries
    are (None, None).

    Returns (latents, ids).
    """
    latents: list[torch.Tensor | None] = []
    latent_ids: list[torch.Tensor | None] = []

    if not load_targets:
        for _ in sample_ids:
            latents.append(None)
            latent_ids.append(None)
        return latents, latent_ids

    from PIL import Image

    for idx, sid in enumerate(sample_ids, start=1):
        img_path: Path | None = None
        for ext in IMAGE_SUFFIXES:
            candidate = image_dir / f"{sid}{ext}"
            if candidate.is_file():
                img_path = candidate
                break

        if img_path is None:
            latents.append(None)
            latent_ids.append(None)
            continue

        # Try cache
        img_cache_path = image_dir / f"{sid}.image.safetensors"
        if img_cache_path.is_file():
            tensors = load_file(str(img_cache_path), device="cpu")
            latents.append(tensors["latent"])
            latent_ids.append(tensors["id"])
            print(f" * [{idx}/{len(sample_ids)}] loaded image cache: {sid}")
            continue

        # Encode
        with Image.open(img_path) as im:
            rgb = im.convert("RGB")
        encode_start = time.perf_counter()
        image_latent, image_latent_id, _ = prepare_image_latent(
            vae, processor, rgb,
            device="cuda",
            vae_scale_factor=vae_scale_factor,
        )
        latents.append(image_latent)
        latent_ids.append(image_latent_id)
        if cache_images:
            save_file(
                {"latent": image_latent.contiguous(), "id": image_latent_id.contiguous()},
                str(img_cache_path),
            )
        print(f" * [{idx}/{len(sample_ids)}] {img_path.name} encoded in "
              f"{time.perf_counter() - encode_start:.3f}s")

    return latents, latent_ids


def _encode_ref_images(
    sample_ids: list[str],
    ref_dir: Path,
    vae,
    processor,
    vae_scale_factor: int,
    *,
    cache_images: bool,
    load_refs: bool,
) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
    """Encode reference images from {ref_dir}/{sid}/ subdirs into VAE latents.

    For each sample, looks for a subdir {ref_dir}/{sid}/. If found, encodes
    all images inside and concatenates them. Otherwise appends (None, None).

    Returns (latents, ids).
    """
    latents: list[torch.Tensor | None] = []
    latent_ids: list[torch.Tensor | None] = []

    if not load_refs:
        for _ in sample_ids:
            latents.append(None)
            latent_ids.append(None)
        return latents, latent_ids

    from PIL import Image

    for idx, sid in enumerate(sample_ids, start=1):
        ref_subdir = ref_dir / sid
        if not ref_subdir.is_dir():
            latents.append(None)
            latent_ids.append(None)
            continue

        ref_paths = sorted(
            (p for p in ref_subdir.iterdir()
             if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda p: p.name,
        )
        if not ref_paths:
            latents.append(None)
            latent_ids.append(None)
            continue

        sample_ref_latents = []
        sample_ref_ids = []
        for ref_path in ref_paths:
            ref_cache_path = ref_path.parent / f"{ref_path.stem}.safetensors"
            if ref_cache_path.is_file():
                ref_tensors = load_file(str(ref_cache_path), device="cpu")
                sample_ref_latents.append(ref_tensors["latent"])
                sample_ref_ids.append(ref_tensors["id"])
                print(f"   * ref [{idx}/{len(sample_ids)}] ({ref_path.name}) loaded cache")
                continue

            ref_encode_start = time.perf_counter()
            with Image.open(ref_path) as im:
                ref_rgb = im.convert("RGB")
            ref_latent, ref_id, _ = prepare_image_latent(
                vae, processor, ref_rgb,
                device="cuda",
                vae_scale_factor=vae_scale_factor,
            )
            sample_ref_latents.append(ref_latent)
            sample_ref_ids.append(ref_id)
            if cache_images:
                save_file(
                    {"latent": ref_latent.contiguous(), "id": ref_id.contiguous()},
                    str(ref_cache_path),
                )
            print(f"   * ref [{idx}/{len(sample_ids)}] ({ref_path.name}) encoded in "
                  f"{time.perf_counter() - ref_encode_start:.3f}s")

        latents.append(torch.cat(sample_ref_latents, dim=0))
        latent_ids.append(torch.cat(sample_ref_ids, dim=0))

    return latents, latent_ids


@dataclass
class Flux2Dataset:
    trigger: Tuple[torch.Tensor] | None
    captions: list[Tuple[torch.Tensor] | None]
    target_image: list[Tuple[torch.Tensor] | None]
    reference_images: list[Tuple[torch.Tensor] | None]
    target_ratios: list[float]
    
    def __init__(
        self,
        model_rootpath,
        text_quant_method,
        dataset_dirpath,
        trigger_str,
        target_resolution,
        reference_resolution,
        indices: Sequence[int] | None = None,
        cache_text: bool = False,
        cache_images: bool = False,
        load_captions: bool = True,
        load_target_images: bool = True,
        load_reference_images: bool = True,
        load_target_ratios: bool = False,
    ):
        print("Loading dataset ...")
        print(f" * Dataset DirPath: {dataset_dirpath}")
        print(f" * Trigger: {trigger_str}")
        print(f" * Flux2 Directory: {model_rootpath}")
        print(f" * Text Endoder Quant: {text_quant_method}")
        print(f" * Cache Text: {cache_text}")
        print(f" * Cache Images: {cache_images}")
        print(f" * Load Captions: {load_captions}")
        print(f" * Load Target Images: {load_target_images}")
        print(f" * Load Reference Images: {load_reference_images}")
        print(f" * Load Target Ratios: {load_target_ratios}")

        torch.cuda.empty_cache()

        vae_load_start = time.perf_counter()
        vae_path = model_rootpath / "vae"
        vae_scale_factor = _load_vae_scale_factor(vae_path)
        vae = AutoencoderKLFlux2.from_pretrained(str(vae_path), local_files_only=True)
        vae = vae.to("cuda", dtype=torch.float16)
        print(f"VAE loaded in {time.perf_counter() - vae_load_start:.3f}s")

        image_processor = Flux2ImageProcessor(vae_scale_factor=vae_scale_factor * 2)

        target_latents = []
        target_ratios: list[float] = []
        captions = []
        text_embeddings = []
        text_cache_paths: list[Path] = []
        all_ref_latents: list = []

        i = 0

        for target_filepath in dataset_dirpath.iterdir():
            if target_filepath.is_file() and target_filepath.suffix.lower() in IMAGE_SUFFIXES:
                if indices is not None and i not in indices:
                    i += 1
                    continue

                print(f" * [{i + 1}]")

                # --- Process refs first ---
                refs_dirpath = dataset_dirpath / f"{target_filepath.stem}"
                sample_refs: list[tuple[torch.Tensor, torch.Tensor]] = []
                has_good_ref = False

                if refs_dirpath.exists():
                    for ref_filepath in sorted(refs_dirpath.iterdir()):
                        if not (ref_filepath.is_file() and ref_filepath.suffix.lower() in IMAGE_SUFFIXES):
                            continue

                        ref_cache_path = ref_filepath.parent / f"{ref_filepath.stem}.safetensors"

                        # Check short side against reference_resolution
                        with Image.open(ref_filepath) as ref_img:
                            rw, rh = ref_img.size
                        if min(rw, rh) < reference_resolution:
                            continue

                        has_good_ref = True

                        if ref_cache_path.is_file():
                            ref_tensors = load_file(str(ref_cache_path), device="cpu")
                            ref_latent = (ref_tensors["latent"], ref_tensors["id"])
                            sample_refs.append(ref_latent)
                            print(f"   * ref ({ref_filepath.name}) loaded cache")
                        else:
                            ref_start = time.perf_counter()
                            with Image.open(ref_filepath) as ref_image:
                                ref_image_rgb = ref_image.convert("RGB")

                            w, h = ref_image_rgb.size
                            scale = reference_resolution / min(w, h)
                            new_w = max(16, round(w * scale / 16) * 16)
                            new_h = max(16, round(h * scale / 16) * 16)
                            new_size = (new_w, new_h)
                            ref_image_resized = ref_image_rgb.resize(new_size, Image.LANCZOS)
                            ref_loading_time = time.perf_counter() - ref_start

                            ref_encoding_start = time.perf_counter()
                            image_tensor = image_processor.preprocess(ref_image_resized)
                            image_tensor = image_tensor.to(device="cuda", dtype=torch.float16)

                            with torch.inference_mode():
                                latent = vae.encode(image_tensor)
                                latent = _retrieve_latents(latent)
                            latent = _patchify_latents(latent)

                            latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latent.device, latent.dtype)
                            latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
                                latent.device, latent.dtype
                            )
                            latent = (latent - latents_bn_mean) / latents_bn_std

                            ref_latent = (
                                _pack_latents(latent).squeeze(0).cpu(),
                                _prepare_latent_ids(torch, latent).squeeze(0).cpu()
                            )

                            ref_encoding_time = time.perf_counter() - ref_encoding_start
                            print(f"   * ref encoded in load={ref_loading_time:.3f}, enc={ref_encoding_time:.3f}, total={ref_loading_time + ref_encoding_time:.3f}s at {new_size}")

                            sample_refs.append(ref_latent)

                            del image_tensor
                            del latent
                            del latents_bn_mean
                            del latents_bn_std

                # Skip if refs exist but none met resolution threshold
                if refs_dirpath.exists() and not has_good_ref:
                    print(f"   * skipped: no ref meets reference_resolution ({reference_resolution})")
                    target_latents.append(None)
                    if load_target_ratios:
                        target_ratios.append(None)
                    captions.append(None)
                    text_embeddings.append(None)
                    text_cache_paths.append(None)
                    all_ref_latents.append(None)
                    i += 1
                    continue

                # --- Target ---
                target_cache_filepath = dataset_dirpath / f"{target_filepath.stem}.image.safetensors"

                if target_cache_filepath.exists():
                    if load_target_ratios:
                        with Image.open(target_filepath) as img:
                            w, h = img.size
                        target_ratios.append(w / h)

                else:
                    target_start = time.perf_counter()
                    with Image.open(target_filepath) as target_image:
                        target_image_rgb = target_image.convert("RGB")

                    w, h = target_image_rgb.size
                    if load_target_ratios:
                        target_ratios.append(w / h)
                    scale = target_resolution / min(w, h)
                    new_w = max(16, round(w * scale / 16) * 16)
                    new_h = max(16, round(h * scale / 16) * 16)
                    new_size = (new_w, new_h)
                    target_image_resized = target_image_rgb.resize(new_size, Image.LANCZOS)
                    target_loading_time = time.perf_counter() - target_start

                    target_encoding_start = time.perf_counter()
                    image_tensor = image_processor.preprocess(target_image_resized)
                    image_tensor = image_tensor.to(device="cuda", dtype=torch.float16)

                    with torch.inference_mode():
                        latent = vae.encode(image_tensor)
                        latent = _retrieve_latents(latent)
                    latent = _patchify_latents(latent)

                    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latent.device, latent.dtype)
                    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
                        latent.device, latent.dtype
                    )
                    latent = (latent - latents_bn_mean) / latents_bn_std

                    target_latent = (
                        _pack_latents(latent).squeeze(0).cpu(),
                        _prepare_latent_ids(torch, latent).squeeze(0).cpu()
                    )

                    target_latents.append(target_latent)
                    target_encoding_time = time.perf_counter() - target_encoding_start
                    print(f" * * target encoded in load={target_loading_time:.3f}, enc={target_encoding_time:.3f}, total={target_loading_time + target_encoding_time:.3f}s at {new_size}")

                    del image_tensor
                    del latent
                    del latents_bn_mean
                    del latents_bn_std

                # Append refs
                if sample_refs:
                    concat_latent = torch.cat([r[0] for r in sample_refs], dim=0)
                    concat_ids = torch.cat([r[1] for r in sample_refs], dim=0)
                    all_ref_latents.append((concat_latent.contiguous(), concat_ids.contiguous()))
                else:
                    all_ref_latents.append(None)

                # Text
                text_cache_filepath = dataset_dirpath / f"{target_filepath.stem}.text.safetensors"

                if text_cache_filepath.exists():
                    tensors = load_file(str(text_cache_filepath), device="cpu")
                    captions.append(None)
                    text_embeddings.append((tensors["embed"], tensors["id"]))
                    text_cache_paths.append(text_cache_filepath)
                else:
                    caption_cache_filepath = dataset_dirpath / f"{target_filepath.stem}.txt"
                    if caption_cache_filepath.is_file():
                        captions.append(caption_cache_filepath.read_text(encoding="utf-8").strip())
                    else:
                        captions.append(None)
                    text_embeddings.append(None)
                    text_cache_paths.append(text_cache_filepath)

                i += 1

        del vae
        del image_processor

        print("Encoding text ...")
        
        tokenizer_path = model_rootpath / "tokenizer"
        text_encoder_path = model_rootpath / "text_encoder"
        text_encoder_load_start = time.perf_counter()
        tokenizer = Qwen2TokenizerFast.from_pretrained(str(tokenizer_path))
        text_encoder = load_flux2_text_encoder(
            str(text_encoder_path),
            quant_method=text_quant_method,
        )
        text_encoder = text_encoder.to("cuda")
        print(f"Text Encoder loaded in {time.perf_counter() - text_encoder_load_start:.3f}s")

        encode_start = time.perf_counter()
        prompt_embeds, text_ids = _encode_prompt_embeddings(
            torch,
            tokenizer,
            text_encoder,
            prompt=trigger_str,
            device="cuda",
            max_length=512,
        )
        embed = prompt_embeds.squeeze(0).cpu().contiguous()
        text_id = text_ids.squeeze(0).cpu().contiguous()
        self.trigger_embedding = (embed, text_id)
        print(f" Trigger encoded in {time.perf_counter() - encode_start:.3f}s")

        for i, caption in enumerate(captions):
            if text_embeddings[i] is not None:
                print(f" * [{i + 1}/{len(captions)}] loaded from cache")
                continue
            if caption is None:
                print(f" * [{i + 1}/{len(captions)}] is None")
                continue

            encode_start = time.perf_counter()
            prompt_embeds, text_ids = _encode_prompt_embeddings(
                torch,
                tokenizer,
                text_encoder,
                prompt=caption,
                device="cuda",
                max_length=512,
            )
            embed = prompt_embeds.squeeze(0).cpu().contiguous()
            text_id = text_ids.squeeze(0).cpu().contiguous()
            text_embeddings[i] = (embed, text_id)
            print(f" * [{i + 1}/{len(captions)}] encoded in {time.perf_counter() - encode_start:.3f}s")

            if cache_text:
                save_file(
                    {"embed": embed.contiguous(), "id": text_id.contiguous()},
                    str(text_cache_paths[i]),
                )

        del tokenizer
        del text_encoder

        self.text_embeddings = text_embeddings
        self.target_latents = target_latents
        self.target_ratios = target_ratios
        self.ref_latents = all_ref_latents