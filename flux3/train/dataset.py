"""Training dataset for Flux 3 Klein.

The project layout is the same as the Flux 2 dataset. The cache files carry a
`flux3` tag, so one project can hold the caches of both models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from flux2.train.dataset import _is_supported_image, _list_image_files
from flux3.inputs import encode_image_latent, encode_prompt, prepare_image_ids, prepare_text_ids, resize_reference

_LATENT_CHANNELS = 128
_TRIGGER_CACHE_NAME = ".trigger.flux3.text.safetensors"


@dataclass
class _Sample:
    """One entry of the sample list, before any image or text is loaded."""

    name: str
    target_path: Path
    ratio: float
    ref_paths: list[Path] = field(default_factory=list)


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _list_samples(
    dataset_dirpath: Path,
    *,
    indices: Sequence[int] | None,
    target_resolution: int,
    reference_resolution: int,
    target_upscale: bool,
    ref_upscale: bool,
) -> list[_Sample]:
    """Return the samples to train on, in name order.

    A target below `target_resolution` is skipped, unless `target_upscale` is
    set. A reference below `reference_resolution` is skipped, unless
    `ref_upscale` is set. A sample whose reference folder exists, but has no
    usable reference, is skipped.
    """
    target_paths = _list_image_files(dataset_dirpath)
    index_set = None if indices is None else set(indices)

    samples: list[_Sample] = []
    for position, target_path in enumerate(target_paths):
        if index_set is not None and position not in index_set:
            continue

        width, height = _image_size(target_path)
        if min(width, height) < target_resolution and not target_upscale:
            print(f" * [{position + 1}] {target_path.name} skipped: target below target_resolution ({target_resolution})")
            continue

        refs_dirpath = dataset_dirpath / target_path.stem
        ref_paths: list[Path] = []
        if refs_dirpath.is_dir():
            for ref_path in sorted(refs_dirpath.iterdir(), key=lambda path: path.name):
                if not _is_supported_image(ref_path):
                    continue
                ref_width, ref_height = _image_size(ref_path)
                if min(ref_width, ref_height) < reference_resolution and not ref_upscale:
                    continue
                ref_paths.append(ref_path)

        if refs_dirpath.exists() and not ref_paths:
            print(
                f" * [{position + 1}] {target_path.name} skipped: "
                f"no ref meets reference_resolution ({reference_resolution})"
            )
            continue

        samples.append(
            _Sample(
                name=target_path.stem,
                target_path=target_path,
                ratio=width / height,
                ref_paths=ref_paths,
            )
        )
    return samples


class _VaeHolder:
    """Load the VAE on the first use. It stays loaded until `free` runs."""

    def __init__(self, vae_path: Path) -> None:
        self._vae_path = vae_path
        self._vae = None

    def get(self):
        if self._vae is None:
            from flux3.gen import _load_vae

            start = time.perf_counter()
            self._vae = _load_vae(self._vae_path).to("cuda", dtype=torch.float16)
            print(f"VAE loaded in {time.perf_counter() - start:.3f}s")
        return self._vae

    def free(self) -> None:
        self._vae = None
        torch.cuda.empty_cache()


def _tokens_to_cache(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return tokens.view(height, width, _LATENT_CHANNELS).permute(2, 0, 1).contiguous()


def _cache_to_tokens(latent: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    _, height, width = latent.shape
    return latent.permute(1, 2, 0).reshape(height * width, _LATENT_CHANNELS).contiguous(), height, width


def _load_tokens(
    image_path: Path,
    short_side: int,
    cache_path: Path,
    *,
    use_cache: bool,
    vae_holder: _VaeHolder,
    label: str,
    layer: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the latent tokens and the ids of an image, from the cache or the VAE.

    `layer` is the `l` coordinate of the ids: 0 for a target, -1, -2, ... for references.
    """
    if use_cache and cache_path.is_file():
        tokens, height, width = _cache_to_tokens(load_file(str(cache_path), device="cpu")["latent"])
        print(f"   * {label} ({image_path.name}) loaded cache")
        return tokens, prepare_image_ids(height, width, layer=layer)

    start = time.perf_counter()
    with Image.open(image_path) as image:
        resized = resize_reference(image.convert("RGB"), short_side)
    tokens, height, width = encode_image_latent(vae_holder.get(), resized, torch.device("cuda"))
    tokens = tokens.to("cpu", torch.float16).contiguous()
    if use_cache:
        save_file({"latent": _tokens_to_cache(tokens, height, width)}, str(cache_path))
    print(f"   * {label} ({image_path.name}) encoded in {time.perf_counter() - start:.3f}s at {resized.size}")
    return tokens, prepare_image_ids(height, width, layer=layer)


def _read_text_cache(path: Path, prompt: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return the cached text embeds, or None if the cache is missing or stale."""
    if not path.is_file():
        return None
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("max_length") != str(max_length) or metadata.get("prompt") != prompt:
        return None
    embed = load_file(str(path), device="cpu")["embed"]
    return embed, prepare_text_ids(embed.shape[0])


def _write_text_cache(path: Path, embed: torch.Tensor, prompt: str, max_length: int) -> None:
    save_file(
        {"embed": embed.contiguous()},
        str(path),
        metadata={"max_length": str(max_length), "prompt": prompt},
    )


class Flux3Dataset:
    """The training data of a Flux 3 Klein project.

    Layout: `<dataset_dirpath>/<name>.<ext>` for targets, `<name>.txt` for
    captions, and `<name>/<ref>.<ext>` for references. The targets are sorted by
    name.

    Attributes, one entry per sample:
        text_embeddings: `(embeds[S, 20480], txt_ids[S, 4])`, or None when the
            sample has no caption. Use `trigger_embedding` then.
        target_latents: `(tokens[h * w, 128], img_ids[h * w, 4])`, or None when
            `load_target_images` is False.
        ref_latents: `(cond_tokens[L, 128], cond_ids[L, 4])`, or None when the
            sample has no reference or `load_reference_images` is False.
        target_ratios: width / height of the target file, or None when
            `load_target_ratios` is False.
        sample_names: The file stem of the target.

    One value for the dataset:
        trigger_embedding: `(embeds[S, 20480], txt_ids[S, 4])` of the trigger.
    """

    def __init__(
        self,
        model_root: Path,
        text_quant_method: str | None,
        dataset_dirpath: Path,
        trigger_str: str,
        target_resolution: int,
        reference_resolution: int,
        *,
        indices: Sequence[int] | None = None,
        cache_text: bool = False,
        cache_images: bool = False,
        load_captions: bool = True,
        load_target_images: bool = True,
        load_reference_images: bool = True,
        load_target_ratios: bool = False,
        target_upscale: bool = False,
        ref_upscale: bool = False,
        max_length: int = 512,
    ) -> None:
        """Load the dataset.

        Args:
            model_root: The `<root>/flux3/model` folder.
            text_quant_method: The text encoder quantization, or None for fp16.
            dataset_dirpath: The folder with the target images.
            trigger_str: The prompt of a sample without a caption.
            target_resolution: The short side of the targets in pixels.
            reference_resolution: The short side of the references in pixels.
            indices: The positions in the sorted target list to keep. None keeps all.
            cache_text: Read and write the text caches.
            cache_images: Read and write the latent caches.
            load_captions: Read the caption files. If False, every sample uses the trigger.
            load_target_images: Encode the targets. If False, `target_latents[i]` is None.
            load_reference_images: Encode the references. If False, `ref_latents[i]` is None.
            load_target_ratios: Fill `target_ratios`.
            target_upscale: Keep targets below `target_resolution`.
            ref_upscale: Keep references below `reference_resolution`.
            max_length: The maximum number of text tokens.
        """
        model_root = Path(model_root)
        dataset_dirpath = Path(dataset_dirpath)
        print("Loading dataset ...")
        print(f" * Dataset DirPath: {dataset_dirpath}")
        print(f" * Trigger: {trigger_str}")
        print(f" * Flux3 Directory: {model_root}")
        print(f" * Text Encoder Quant: {text_quant_method}")
        print(f" * Cache Text: {cache_text}")
        print(f" * Cache Images: {cache_images}")
        print(f" * Load Captions: {load_captions}")
        print(f" * Load Target Images: {load_target_images}")
        print(f" * Load Reference Images: {load_reference_images}")
        print(f" * Load Target Ratios: {load_target_ratios}")
        print(f" * Target Upscale: {target_upscale}")
        print(f" * Ref Upscale: {ref_upscale}")

        samples = _list_samples(
            dataset_dirpath,
            indices=indices,
            target_resolution=target_resolution,
            reference_resolution=reference_resolution,
            target_upscale=target_upscale,
            ref_upscale=ref_upscale,
        )
        print(f" * Samples: {len(samples)}")

        # 1. Images.
        torch.cuda.empty_cache()
        vae_holder = _VaeHolder(model_root / "vae")
        target_latents: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        ref_latents: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        for number, sample in enumerate(samples, start=1):
            print(f" * [{number}/{len(samples)}] {sample.name}")

            if load_target_images:
                target_latents.append(
                    _load_tokens(
                        sample.target_path,
                        target_resolution,
                        dataset_dirpath / f"{sample.name}.flux3.{target_resolution}.safetensors",
                        use_cache=cache_images,
                        vae_holder=vae_holder,
                        label="target",
                    )
                )
            else:
                target_latents.append(None)

            refs: list[tuple[torch.Tensor, torch.Tensor]] = []
            if load_reference_images:
                for ref_path in sample.ref_paths:
                    tokens, ids = _load_tokens(
                        ref_path,
                        reference_resolution,
                        ref_path.parent / f"{ref_path.stem}.flux3.{reference_resolution}.safetensors",
                        use_cache=cache_images,
                        vae_holder=vae_holder,
                        label="ref",
                        layer=-(1 + len(refs)),
                    )
                    refs.append((tokens, ids))
            if refs:
                ref_latents.append(
                    (torch.cat([tokens for tokens, _ in refs], dim=0), torch.cat([ids for _, ids in refs], dim=0))
                )
            else:
                ref_latents.append(None)
        vae_holder.free()

        # 2. Text.
        captions: list[str | None] = []
        text_embeddings: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        text_cache_paths: list[Path] = []
        for sample in samples:
            text_cache_paths.append(dataset_dirpath / f"{sample.name}.flux3.text.safetensors")
            caption_path = dataset_dirpath / f"{sample.name}.txt"
            if load_captions and caption_path.is_file():
                caption = caption_path.read_text(encoding="utf-8").strip()
                captions.append(caption)
                cached = _read_text_cache(text_cache_paths[-1], caption, max_length) if cache_text else None
                text_embeddings.append(cached)
            else:
                captions.append(None)
                text_embeddings.append(None)

        trigger_cache_path = dataset_dirpath / _TRIGGER_CACHE_NAME
        trigger_embedding = _read_text_cache(trigger_cache_path, trigger_str, max_length) if cache_text else None

        pending = [i for i, caption in enumerate(captions) if caption is not None and text_embeddings[i] is None]
        if pending or trigger_embedding is None:
            from transformers import Qwen2TokenizerFast

            from flux3.text_encoder.loader import load_flux3_text_encoder

            print("Encoding text ...")
            start = time.perf_counter()
            tokenizer = Qwen2TokenizerFast.from_pretrained(str(model_root / "text_encoder"))
            text_encoder = load_flux3_text_encoder(model_root / "text_encoder", text_quant_method)
            text_encoder = text_encoder.to("cuda")
            print(f"Text Encoder loaded in {time.perf_counter() - start:.3f}s")

            device = torch.device("cuda")

            def encode(prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
                embeds, _ = encode_prompt(tokenizer, text_encoder, prompt, device=device, max_length=max_length)
                embed = embeds.squeeze(0).to("cpu", torch.float16).contiguous()
                return embed, prepare_text_ids(embed.shape[0])

            if trigger_embedding is None:
                start = time.perf_counter()
                trigger_embedding = encode(trigger_str)
                if cache_text:
                    _write_text_cache(trigger_cache_path, trigger_embedding[0], trigger_str, max_length)
                print(f" Trigger encoded in {time.perf_counter() - start:.3f}s")

            for number, i in enumerate(pending, start=1):
                start = time.perf_counter()
                text_embeddings[i] = encode(captions[i])
                if cache_text:
                    _write_text_cache(text_cache_paths[i], text_embeddings[i][0], captions[i], max_length)
                print(f" * [{number}/{len(pending)}] {samples[i].name} encoded in {time.perf_counter() - start:.3f}s")

            del tokenizer
            del text_encoder
            torch.cuda.empty_cache()
        else:
            print("Text: every embedding loaded from cache")

        self.text_embeddings = text_embeddings
        self.target_latents = target_latents
        self.ref_latents = ref_latents
        self.target_ratios = [sample.ratio if load_target_ratios else None for sample in samples]
        self.sample_names = [sample.name for sample in samples]
        self.trigger_embedding = trigger_embedding
