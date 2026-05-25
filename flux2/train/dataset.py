from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import time
from PIL import Image
from diffusers import AutoencoderKLFlux2
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
import torch
from transformers import Qwen2TokenizerFast
from safetensors.torch import load_file, save_file

from flux2.gen import _encode_prompt_embeddings, _load_vae_scale_factor
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


def _load_single_dataset(dataset_dir: Path) -> dict[str, list[str | Path | None] | list[list[Path] | None]]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    target_images = _list_image_files(dataset_dir)
    if not target_images:
        raise FileNotFoundError(f"No supported target images found in dataset: {dataset_dir}")

    stems_seen: set[str] = set()
    duplicate_stems: list[str] = []
    for image_path in target_images:
        if image_path.stem in stems_seen:
            duplicate_stems.append(image_path.stem)
        stems_seen.add(image_path.stem)
    if duplicate_stems:
        duplicate_text = ", ".join(sorted(set(duplicate_stems)))
        raise ValueError(f"Dataset contains duplicate target image stems: {duplicate_text}")

    text_prompts: list[str | None] = []
    reference_images: list[list[Path] | None] = []
    for target_image in target_images:
        prompt_path = target_image.with_suffix(".txt")
        if prompt_path.is_file():
            text_prompts.append(prompt_path.read_text(encoding="utf-8").strip())
        else:
            text_prompts.append(None)

        references_dir = dataset_dir / target_image.stem
        if references_dir.is_dir():
            reference_images.append(_list_image_files(references_dir))
        else:
            reference_images.append(None)

    if not (len(target_images) == len(text_prompts) == len(reference_images)):
        raise ValueError("Dataset loader returned misaligned target, prompt, and reference lists.")

    return {
        "text_prompts": text_prompts,
        "target_images": target_images,
        "reference_images": reference_images,
    }


def _select_dataset_indices(
    dataset: dict[str, list[str | Path | None] | list[list[Path] | None]],
    indices: Sequence[int] | None,
) -> dict[str, list[str | Path | None] | list[list[Path] | None]]:
    if indices is None:
        return dataset

    dataset_size = len(dataset["target_images"])
    invalid_indices = [
        sample_index for sample_index in indices if sample_index < 0 or sample_index >= dataset_size
    ]
    if invalid_indices:
        invalid_text = ", ".join(str(sample_index) for sample_index in invalid_indices)
        raise ValueError(f"Dataset indices out of range for {dataset_size} images: {invalid_text}")

    return {
        "text_prompts": [dataset["text_prompts"][sample_index] for sample_index in indices],
        "target_images": [dataset["target_images"][sample_index] for sample_index in indices],
        "reference_images": [dataset["reference_images"][sample_index] for sample_index in indices],
    }


def _load_imgs_and_caps(
    dataset: dict[str, list[str | Path | None] | list[list[Path] | None]],
    *,
    collect_ratios: bool = False,
    load_target_image: bool = True,
) -> tuple[list[Image.Image | None], list[str], list[list[Image.Image]], list[str], list[Path], list[float]]:
    images: list[Image.Image | None] = []
    captions: list[str] = []
    references: list[list[Image.Image]] = []
    sample_ids: list[str] = []
    sample_paths: list[Path] = []
    ratios: list[float] = []

    for image_path, caption, reference_paths in zip(
        dataset["target_images"],
        dataset["text_prompts"],
        dataset["reference_images"],
    ):
        with Image.open(image_path) as image:
            if collect_ratios:
                width, height = image.size
                ratios.append(width / height)
            if load_target_image:
                images.append(image.convert("RGB"))
            else:
                images.append(None)

        captions.append(caption or "")
        sample_ids.append(image_path.stem)
        sample_paths.append(image_path)

        sample_references: list[Image.Image] = []
        if reference_paths is not None:
            for reference_path in reference_paths:
                with Image.open(reference_path) as reference_image:
                    sample_references.append(reference_image.convert("RGB"))
        references.append(sample_references)

    return images, captions, references, sample_ids, sample_paths, ratios


def _encode_text(
    text: str,
    tokenizer,
    text_encoder,
    text_embeds_list,
    text_ids_list,
    cache_path: Path | None = None,
):
    prompt_embeds, text_ids = _encode_prompt_embeddings(
        torch,
        tokenizer,
        text_encoder,
        prompt=text,
        device="cuda",
        max_length=512,
    )
    embed = prompt_embeds.squeeze(0).cpu().contiguous()
    text_id = text_ids.squeeze(0).cpu().contiguous()
    text_embeds_list.append(embed)
    text_ids_list.append(text_id)

    if cache_path is not None:
        save_file({"embed": embed, "id": text_id}, str(cache_path))


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
    text_embeds_list: list[torch.Tensor | None]
    text_ids_list: list[torch.Tensor | None]
    base_target_latents: list[torch.Tensor | None]
    base_target_latent_ids: list[torch.Tensor | None]
    high_target_latents: list[torch.Tensor | None]
    high_target_latent_ids: list[torch.Tensor | None]
    base_ref_latents: list[torch.Tensor | None]
    base_ref_latent_ids: list[torch.Tensor | None]
    high_ref_latents: list[torch.Tensor | None]
    high_ref_latent_ids: list[torch.Tensor | None]
    target_image_ratios: list[float]
    ratios: list[float]
    trigger_embed: torch.Tensor | None = None
    trigger_id: torch.Tensor | None = None

    def __init__(
        self,
        model_rootpath,
        text_quant_method,
        dataset_dirpath,
        trigger,
        indices: Sequence[int] | None = None,
        cache_text: bool = False,
        cache_images: bool = False,
        ratios: bool = False,
        load_target_image: bool = True,
        load_reference_image: bool = True,
    ):
        print("Loading dataset ...")
        print(f" * Dataset DirPath: {dataset_dirpath / 'base'}")
        print(f" * Trigger: {trigger}")
        print(f" * Flux2 Directory: {model_rootpath}")
        print(f" * Text Endoder Quant: {text_quant_method}")
        print(f" * Cache Text: {cache_text}")
        print(f" * Cache Images: {cache_images}")
        print(f" * Collect Ratios: {ratios}")
        print(f" * Load Target Image: {load_target_image}")
        print(f" * Load Reference Image: {load_reference_image}")

        load_start = time.perf_counter()
        base_dir = dataset_dirpath / "base"
        self._dataset_dir = Path(dataset_dirpath)
        dataset = _select_dataset_indices(_load_single_dataset(base_dir), indices)
        image_count = len(dataset["target_images"])
        caption_count = sum(caption is not None for caption in dataset["text_prompts"])
        if trigger is None or trigger.strip() == "":
            raise ValueError("Trigger is required.")

        tgts, caps, references, sample_ids, _, target_image_ratios = _load_imgs_and_caps(
            dataset,
            collect_ratios=ratios,
            load_target_image=load_target_image,
        )
        print(f"Loaded images and captions in {time.perf_counter() - load_start:.3f}s")

        text_items = list(zip(sample_ids, caps))
        text_embeds_list = []
        text_ids_list = []
        text_cache_dir = base_dir

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
        _encode_text(trigger, tokenizer, text_encoder, text_embeds_list, text_ids_list)
        self.trigger_embed = text_embeds_list[0]
        self.trigger_id = text_ids_list[0]
        text_embeds_list.clear()
        text_ids_list.clear()
        print(f" * trigger encoded in {time.perf_counter() - encode_start:.3f}s")

        for sample_index, (sample_id, text) in enumerate(text_items):
            if not text:
                text_embeds_list.append(None)
                text_ids_list.append(None)
                continue

            cache_path = text_cache_dir / f"{sample_id}.text.safetensors"
            if cache_path.is_file():
                tensors = load_file(str(cache_path), device="cpu")
                text_embeds_list.append(tensors["embed"])
                text_ids_list.append(tensors["id"])
                print(f" * [{sample_index + 1}/{len(text_items)}] loaded text cache: {sample_id}")
                continue

            encode_start = time.perf_counter()
            _encode_text(
                text,
                tokenizer,
                text_encoder,
                text_embeds_list,
                text_ids_list,
                cache_path=cache_path if cache_text else None,
            )
            print(f" * [{sample_index + 1}/{len(text_items)}] encoded: {sample_id} in {time.perf_counter() - encode_start:.3f}s")

        del tokenizer
        del text_encoder

        base_target_latents = []
        base_target_latent_ids = []
        high_target_latents = []
        high_target_latent_ids = []
        base_ref_latents = []
        base_ref_latent_ids = []
        high_ref_latents = []
        high_ref_latent_ids = []

        if load_target_image or load_reference_image:
            torch.cuda.empty_cache()

            vae_load_start = time.perf_counter()
            vae_path = model_rootpath / "vae"
            vae_scale_factor = _load_vae_scale_factor(vae_path)
            vae = AutoencoderKLFlux2.from_pretrained(str(vae_path), local_files_only=True)
            vae = vae.to("cuda", dtype=torch.float16)
            print(f"VAE loaded in {time.perf_counter() - vae_load_start:.3f}s")

            image_processor = Flux2ImageProcessor(vae_scale_factor=vae_scale_factor * 2)

            print("Encoding base target images ...")
            base_target_latents, base_target_latent_ids = _encode_images(
                sample_ids, base_dir, vae, image_processor, vae_scale_factor,
                cache_images=cache_images,
                load_targets=load_target_image,
            )

            print("Encoding high target images ...")
            high_target_latents, high_target_latent_ids = _encode_images(
                sample_ids, self._dataset_dir / "high", vae, image_processor, vae_scale_factor,
                cache_images=cache_images,
                load_targets=load_target_image,
            )

            print("Encoding base reference images ...")
            base_ref_latents, base_ref_latent_ids = _encode_ref_images(
                sample_ids, base_dir, vae, image_processor, vae_scale_factor,
                cache_images=cache_images,
                load_refs=load_reference_image,
            )

            print("Encoding high reference images ...")
            high_ref_latents, high_ref_latent_ids = _encode_ref_images(
                sample_ids, self._dataset_dir / "high", vae, image_processor, vae_scale_factor,
                cache_images=cache_images,
                load_refs=load_reference_image,
            )

            del vae
            del image_processor
        else:
            print("Skipping image encoding ...")
            for _ in range(len(tgts)):
                base_target_latents.append(None)
                base_target_latent_ids.append(None)
                high_target_latents.append(None)
                high_target_latent_ids.append(None)
                base_ref_latents.append(None)
                base_ref_latent_ids.append(None)
                high_ref_latents.append(None)
                high_ref_latent_ids.append(None)

        self.text_embeds_list = text_embeds_list
        self.text_ids_list = text_ids_list
        self.base_target_latents = base_target_latents
        self.base_target_latent_ids = base_target_latent_ids
        self.high_target_latents = high_target_latents
        self.high_target_latent_ids = high_target_latent_ids
        self.base_ref_latents = base_ref_latents
        self.base_ref_latent_ids = base_ref_latent_ids
        self.high_ref_latents = high_ref_latents
        self.high_ref_latent_ids = high_ref_latent_ids
        self.target_image_ratios = target_image_ratios
        self.ratios = target_image_ratios

        high_tgt_count = sum(1 for h in high_target_latents if h is not None)
        high_ref_count = sum(1 for h in high_ref_latents if h is not None)
        if high_tgt_count or high_ref_count:
            print(f" * high encodings: {high_tgt_count} targets, {high_ref_count} refs")
