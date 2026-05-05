from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import time
from PIL import Image
from diffusers import AutoencoderKLFlux2
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
import torch
from transformers import Qwen2TokenizerFast

from flux2.gen import _encode_prompt_embeddings, _load_vae_scale_factor, _prepare_image_ids
from flux2.text_encoder.loader import _load_flux2_text_encoder as load_flux2_text_encoder
from flux2.train.images import image_latent_resolution, prepare_image_latent

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _list_image_files(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.iterdir() if _is_supported_image(path)),
        key=lambda path: path.name,
    )


def _load_single_dataset(dataset_dir: Path) -> dict[str, list[str | Path | None] | list[list[Path]]]:
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
    reference_images: list[list[Path]] = []
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
            reference_images.append([])

    if not (len(target_images) == len(text_prompts) == len(reference_images)):
        raise ValueError("Dataset loader returned misaligned target, prompt, and reference lists.")

    return {
        "text_prompts": text_prompts,
        "target_images": target_images,
        "reference_images": reference_images,
    }


def _select_dataset_indices(
    dataset: dict[str, list[str | Path | None] | list[list[Path]]],
    indices: Sequence[int] | None,
) -> dict[str, list[str | Path | None] | list[list[Path]]]:
    if indices is None:
        return dataset

    dataset_size = len(dataset["target_images"])
    selected_indices = [
        sample_index for sample_index in indices if sample_index >= 0 and sample_index < dataset_size
    ]

    return {
        "text_prompts": [dataset["text_prompts"][sample_index] for sample_index in selected_indices],
        "target_images": [dataset["target_images"][sample_index] for sample_index in selected_indices],
        "reference_images": [dataset["reference_images"][sample_index] for sample_index in selected_indices],
    }

@dataclass
class Encodings:
    prompt_embeds: Any
    text_ids: Any
    target_image_latents: list[Any]
    target_image_latent_ids: list[Any]
    target_image_resolutions: list[tuple[int, int]]
    reference_latents: list[Any]
    reference_latent_ids: list[Any]

    def __init__(
        self,
        prompt_embeds=None,
        text_ids=None,
        target_image_latents=None,
        target_image_latent_ids=None,
        target_image_resolutions=None,
        reference_latents=None,
        reference_latent_ids=None,
    ):
        self.prompt_embeds = prompt_embeds
        self.text_ids = text_ids
        self.target_image_latents = target_image_latents if target_image_latents is not None else []
        self.target_image_latent_ids = target_image_latent_ids if target_image_latent_ids is not None else []
        self.target_image_resolutions = target_image_resolutions if target_image_resolutions is not None else []
        self.reference_latents = reference_latents if reference_latents is not None else []
        self.reference_latent_ids = reference_latent_ids if reference_latent_ids is not None else []


@dataclass
class Flux2Dataset:
    shared_prompt_embeds: Any
    shared_text_ids: Any
    low_res_encodings: Encodings
    high_res_encodings: Encodings

    def __init__(
        self,
        model_rootpath,
        text_quant_method,
        dataset_dirpath,
        trigger,
        indices: Sequence[int] | None = None,
    ):
        low_res_encodings = Encodings()
        high_res_encodings = Encodings()

        dataset = {
            "low": _select_dataset_indices(_load_single_dataset(dataset_dirpath / "images" / "low"), indices),
            "high": _select_dataset_indices(_load_single_dataset(dataset_dirpath / "images" / "high"), indices),
        }

        tokenizer_path = model_rootpath / "tokenizer"
        text_encoder_path = model_rootpath / "text_encoder"
        tokenizer = Qwen2TokenizerFast.from_pretrained(str(tokenizer_path))
        text_encoder = load_flux2_text_encoder(
            str(text_encoder_path),
            quant_method=text_quant_method,
        )
        text_encoder = text_encoder.to("cuda")
        
        shared_prompt_embeds = None
        shared_text_ids = None

        if trigger is None:
            for res in ["low", "high"]:
                encodings = low_res_encodings if res == "low" else high_res_encodings
                prompt_embeds_list = []
                text_ids_list = []
                text_prompts = dataset[res]["text_prompts"]
                for caption_index, caption in enumerate(text_prompts, start=1):

                    encode_start = time.perf_counter()
                    sample_prompt_embeds, sample_text_ids = _encode_prompt_embeddings(
                        torch,
                        tokenizer,
                        text_encoder,
                        prompt=caption,
                        device="cuda",
                        max_length=512,
                    )
                    prompt_embeds_list.append(sample_prompt_embeds.squeeze(0).cpu())
                    text_ids_list.append(sample_text_ids.squeeze(0).cpu())

                    print(f"  * {res} {caption_index} / {len(text_prompts)} : {time.perf_counter() - encode_start:.3f}s")

                encodings.prompt_embeds = torch.stack(prompt_embeds_list, dim=0)
                encodings.text_ids = torch.stack(text_ids_list, dim=0)
                print(f"  * cached {res} caption encodings on cpu: {tuple(encodings.prompt_embeds.shape)}")
        else:
            encode_start = time.perf_counter()
            shared_prompt_embeds, shared_text_ids = _encode_prompt_embeddings(
                torch,
                tokenizer,
                text_encoder,
                prompt=trigger,
                device="cuda",
                max_length=512,
            )
            print(f"  * trigger encode time: {time.perf_counter() - encode_start:.3f}s embeddings: {tuple(shared_prompt_embeds.shape)} text ids: {tuple(shared_text_ids.shape)}")

        del tokenizer
        del text_encoder

        torch.cuda.empty_cache()

        vae_load_start = time.perf_counter()
        vae_path = model_rootpath / "vae"
        vae_scale_factor = _load_vae_scale_factor(vae_path)
        vae = AutoencoderKLFlux2.from_pretrained(str(vae_path), local_files_only=True)
        vae = vae.to("cuda", dtype=torch.float16)
        print(f"  * vae loaded in {time.perf_counter() - vae_load_start:.3f}s")

        image_latent_prep_start = time.perf_counter()
        image_processor = Flux2ImageProcessor(vae_scale_factor=vae_scale_factor * 2)

        for res in ["low", "high"]:
            encodings = low_res_encodings if res == "low" else high_res_encodings

            target_image_latents = []
            target_image_latent_ids = []
            target_image_resolutions = []
            for image_index, image_path in enumerate(dataset[res]["target_images"], start=1):
                
                image_encode_start = time.perf_counter()
                with Image.open(image_path) as target_image:
                    target_image_pil = target_image.convert("RGB")
                    target_image_resolutions.append(
                        image_latent_resolution(target_image_pil, vae_scale_factor=vae_scale_factor)
                    )
                    image_latent, image_latent_id, _ = prepare_image_latent(
                        vae,
                        image_processor,
                        target_image_pil,
                        device="cuda",
                        vae_scale_factor=vae_scale_factor,
                    )
                target_image_latents.append(image_latent)
                target_image_latent_ids.append(image_latent_id)
                print(
                    f"  * {image_index} / {len(dataset[res]['target_images'])} "
                    f"encoded in {time.perf_counter() - image_encode_start:.3f}s"
                )

            reference_latents = []
            reference_latent_ids = []
            for sample_index, sample_references in enumerate(dataset[res]["reference_images"], start=1):
                if not sample_references:
                    reference_latents.append(None)
                    reference_latent_ids.append(None)
                    continue

                print(f"  * sample {sample_index} references: {len(sample_references)} ...")
                reference_encode_start = time.perf_counter()
                encoded_reference_latents = []
                raw_reference_latents = []
                for reference_index, reference_path in enumerate(sample_references, start=1):
                    print(f"    * reference {reference_index} / {len(sample_references)} ...")
                    with Image.open(reference_path) as reference_image:
                        reference_latent, _, raw_reference_latent = prepare_image_latent(
                            vae,
                            image_processor,
                            reference_image.convert("RGB"),
                            device="cuda",
                            vae_scale_factor=vae_scale_factor,
                        )
                    encoded_reference_latents.append(reference_latent)
                    raw_reference_latents.append(raw_reference_latent)

                reference_latents.append(torch.cat(encoded_reference_latents, dim=0))
                reference_latent_ids.append(_prepare_image_ids(torch, raw_reference_latents).squeeze(0).cpu())
                print(f"  reference encoding took {time.perf_counter() - reference_encode_start:.3f}s")

            encodings.target_image_latents = target_image_latents
            encodings.target_image_latent_ids = target_image_latent_ids
            encodings.target_image_resolutions = target_image_resolutions
            encodings.reference_latents = reference_latents
            encodings.reference_latent_ids = reference_latent_ids

        self.shared_prompt_embeds = shared_prompt_embeds
        self.shared_text_ids = shared_text_ids
        self.low_res_encodings = low_res_encodings
        self.high_res_encodings = high_res_encodings
