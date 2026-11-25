"""Training dataset for Krea 2.

The project layout matches the Flux 2 and Flux 3 trainers: images and their
`.txt` captions live in `<project>/images`. The cache files carry a `krea2` tag,
so one project can hold the caches of several models.

The text cache is large: every caption stores 12 taps of a 2560-wide hidden
state over 512 tokens, 31.5 MB in fp16. A 100-image project is about 3.1 GB of
disk. Encoding is cached by default because the text encoder is 3.7 GB and one
caption costs about 2.3 s.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from krea_2.config import PATCH_SIZE, VAE_SCALE_FACTOR
from krea_2.inputs import encode_latents, prepare_position_ids

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
DEFAULT_TARGET_RESOLUTION = 512
ALIGN = VAE_SCALE_FACTOR * PATCH_SIZE
TEXT_EMBEDS_KEY = "embeds"
TEXT_MASK_KEY = "mask"
LATENT_KEY = "latent"


def target_size(
    width: int,
    height: int,
    target_resolution: int = DEFAULT_TARGET_RESOLUTION,
) -> tuple[int, int]:
    """Return the training size for an image, keeping its aspect ratio.

    The short side becomes `target_resolution` and both sides round to a
    multiple of 16, the size of one image token. Batch size is one, so the
    aspect ratio never needs bucketing.

    Args:
        width: The source width in pixels.
        height: The source height in pixels.
        target_resolution: The short side of the result.

    Returns:
        The `(width, height)` of the resized image.
    """
    scale = target_resolution / min(width, height)
    new_width = max(ALIGN, round(width * scale / ALIGN) * ALIGN)
    new_height = max(ALIGN, round(height * scale / ALIGN) * ALIGN)
    return new_width, new_height


def _list_image_files(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name,
    )


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


class _VaeHolder:
    """Load the VAE on the first use. It stays loaded until `free` runs."""

    def __init__(self, vae_path: Path) -> None:
        self._vae_path = vae_path
        self._vae = None

    def get(self):
        if self._vae is None:
            from krea_2.vae.loader import load_krea2_vae

            start = time.perf_counter()
            self._vae = load_krea2_vae(self._vae_path).to("cuda", dtype=torch.float16)
            print(f"VAE loaded in {time.perf_counter() - start:.3f}s")
        return self._vae

    def free(self) -> None:
        self._vae = None
        torch.cuda.empty_cache()


def _resize(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def _load_latent(
    image_path: Path,
    size: tuple[int, int],
    cache_path: Path,
    *,
    write_cache: bool,
    vae_holder: _VaeHolder,
    label: str,
) -> tuple[torch.Tensor, int, int]:
    """Return the latent tokens and grid size of one image.

    An existing cache is always read, whether or not `write_cache` is set.
    """
    grid_h = size[1] // ALIGN
    grid_w = size[0] // ALIGN
    if cache_path.is_file():
        tokens = load_file(str(cache_path), device="cpu")[LATENT_KEY]
        print(f"   * {label} ({image_path.name}) loaded cache")
        return tokens, grid_h, grid_w

    start = time.perf_counter()
    with Image.open(image_path) as image:
        resized = _resize(image, size)
    tokens, encoded_h, encoded_w = encode_latents(vae_holder.get(), resized, device=torch.device("cuda"))
    if (encoded_h, encoded_w) != (grid_h, grid_w):
        raise RuntimeError(
            f"VAE grid {(encoded_h, encoded_w)} does not match the target size {size} of {image_path.name}."
        )
    tokens = tokens.to("cpu", torch.float16).contiguous()
    if write_cache:
        save_file({LATENT_KEY: tokens}, str(cache_path))
    print(f"   * {label} ({image_path.name}) encoded in {time.perf_counter() - start:.3f}s at {size}")
    return tokens, grid_h, grid_w


def _read_text_cache(path: Path, prompt: str, max_length: int, text_positions: str):
    """Return the cached text embeds and mask, or None if the cache is missing or stale."""
    if not path.is_file():
        return None
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if (
        metadata.get("prompt") != prompt
        or metadata.get("max_length") != str(max_length)
        or metadata.get("text_positions") != text_positions
    ):
        return None
    contents = load_file(str(path), device="cpu")
    return contents[TEXT_EMBEDS_KEY], contents[TEXT_MASK_KEY]


def _write_text_cache(
    path: Path,
    embeds: torch.Tensor,
    mask: torch.Tensor,
    prompt: str,
    max_length: int,
    text_positions: str,
) -> None:
    save_file(
        {TEXT_EMBEDS_KEY: embeds.contiguous(), TEXT_MASK_KEY: mask.contiguous()},
        str(path),
        metadata={
            "prompt": prompt,
            "max_length": str(max_length),
            "text_positions": text_positions,
        },
    )


class Krea2Dataset:
    """The training data of a Krea 2 project.

    Layout: `<dataset_dirpath>/<name>.<ext>` for the images and `<name>.txt`
    for the captions. The images are sorted by name.

    Attributes, one entry per sample:
        text_embeddings: `(embeds[S, 12, 2560], mask[S])` in fp16, or None when
            the sample has no caption. Use `trigger_embedding` then.
        target_latents: `(tokens[h * w, 64], position_ids[S + h * w, 3])`.
        sample_names: The file stem of the image.

    One value for the dataset:
        trigger_embedding: `(embeds[S, 12, 2560], mask[S])` of the trigger.

    Raises:
        ValueError: If the image folder has no supported image.
    """

    def __init__(
        self,
        model_root: Path,
        text_quant_method: str | None,
        dataset_dirpath: Path,
        trigger_str: str,
        target_resolution: int,
        *,
        indices: Sequence[int] | None = None,
        cache_text: bool = True,
        cache_images: bool = True,
        max_length: int = 512,
        text_positions: str = "diffusers",
    ) -> None:
        """Load the dataset.

        Args:
            model_root: The `<root>/krea_2/model` folder.
            text_quant_method: The text encoder quantization, or None for fp16.
            dataset_dirpath: The folder with the images.
            trigger_str: The prompt of a sample without a caption.
            target_resolution: The short side of the images in pixels.
            indices: The positions in the sorted image list to keep. None keeps all.
            cache_text: Write the text caches. An existing cache is always read.
            cache_images: Write the latent caches. An existing cache is always read.
            max_length: The number of text tokens.
            text_positions: `diffusers` or `absolute`, as in `encode_prompt`.

        Raises:
            ValueError: If the folder has no supported image.
        """
        model_root = Path(model_root)
        dataset_dirpath = Path(dataset_dirpath)
        print("Loading dataset ...")
        print(f" * Dataset DirPath: {dataset_dirpath}")
        print(f" * Trigger: {trigger_str}")
        print(f" * Krea 2 Directory: {model_root}")
        print(f" * Text Encoder Quant: {text_quant_method}")
        print(f" * Cache Text: {cache_text}")
        print(f" * Cache Images: {cache_images}")
        print(f" * Target Resolution: {target_resolution}")
        print(f" * Max Length: {max_length}")

        image_paths = _list_image_files(dataset_dirpath) if dataset_dirpath.is_dir() else []
        if indices is not None:
            index_set = set(indices)
            image_paths = [path for position, path in enumerate(image_paths) if position in index_set]
        if not image_paths:
            raise ValueError(f"No supported images found in {dataset_dirpath}.")
        print(f" * Samples: {len(image_paths)}")

        # 1. Text. Read the caches first, because the position ids need the
        # text length. Encode only what is missing or stale.
        captions: list[str | None] = []
        text_embeddings: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        text_cache_paths: list[Path] = []
        for image_path in image_paths:
            text_cache_paths.append(dataset_dirpath / f"{image_path.stem}.krea2.text.safetensors")
            caption_path = dataset_dirpath / f"{image_path.stem}.txt"
            if caption_path.is_file():
                caption = caption_path.read_text(encoding="utf-8").strip()
                captions.append(caption)
                text_embeddings.append(
                    _read_text_cache(text_cache_paths[-1], caption, max_length, text_positions)
                )
            else:
                captions.append(None)
                text_embeddings.append(None)

        trigger_cache_path = dataset_dirpath / ".trigger.krea2.text.safetensors"
        trigger_embedding = _read_text_cache(trigger_cache_path, trigger_str, max_length, text_positions)

        pending = [i for i, caption in enumerate(captions) if caption is not None and text_embeddings[i] is None]
        if pending or trigger_embedding is None:
            from transformers import Qwen2TokenizerFast

            from krea_2.inputs import encode_prompt
            from krea_2.text_encoder.loader import load_krea2_text_encoder

            print("Encoding text ...")
            start = time.perf_counter()
            tokenizer = Qwen2TokenizerFast.from_pretrained(str(model_root / "text_encoder"))
            text_encoder = load_krea2_text_encoder(model_root / "text_encoder", text_quant_method)
            text_encoder = text_encoder.to("cuda")
            print(f"Text Encoder loaded in {time.perf_counter() - start:.3f}s")
            device = torch.device("cuda")

            def encode(prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
                embeds, mask = encode_prompt(
                    tokenizer,
                    text_encoder,
                    prompt,
                    device=device,
                    max_length=max_length,
                    text_positions=text_positions,
                )
                return embeds[0].to("cpu", torch.float16).contiguous(), mask[0].to("cpu").contiguous()

            if trigger_embedding is None:
                start = time.perf_counter()
                trigger_embedding = encode(trigger_str)
                if cache_text:
                    _write_text_cache(
                        trigger_cache_path,
                        trigger_embedding[0],
                        trigger_embedding[1],
                        trigger_str,
                        max_length,
                        text_positions,
                    )
                print(f" * trigger encoded in {time.perf_counter() - start:.3f}s")

            for number, i in enumerate(pending, start=1):
                start = time.perf_counter()
                text_embeddings[i] = encode(captions[i])
                if cache_text:
                    _write_text_cache(
                        text_cache_paths[i],
                        text_embeddings[i][0],
                        text_embeddings[i][1],
                        captions[i],
                        max_length,
                        text_positions,
                    )
                print(f" * [{number}/{len(pending)}] {image_paths[i].stem} encoded in {time.perf_counter() - start:.3f}s")

            del tokenizer
            del text_encoder
            torch.cuda.empty_cache()
        else:
            print("Text: every embedding loaded from cache")

        # 2. Images.
        torch.cuda.empty_cache()
        vae_holder = _VaeHolder(model_root / "vae")
        target_latents: list[tuple[torch.Tensor, torch.Tensor]] = []
        for number, image_path in enumerate(image_paths, start=1):
            print(f" * [{number}/{len(image_paths)}] {image_path.stem}")
            width, height = _image_size(image_path)
            size = target_size(width, height, target_resolution)
            tokens, grid_h, grid_w = _load_latent(
                image_path,
                size,
                dataset_dirpath / f"{image_path.stem}.krea2.{size[1]}x{size[0]}.latents.safetensors",
                write_cache=cache_images,
                vae_holder=vae_holder,
                label="target",
            )
            target_latents.append((tokens, (grid_h, grid_w)))
        vae_holder.free()

        # 3. Position ids, one table per sample because the image grid differs.
        positions: list[torch.Tensor] = []
        for i, (_, (grid_h, grid_w)) in enumerate(target_latents):
            embedding = text_embeddings[i] if text_embeddings[i] is not None else trigger_embedding
            positions.append(prepare_position_ids(embedding[0].shape[0], grid_h, grid_w, torch.device("cpu")))

        self.text_embeddings = text_embeddings
        self.target_latents = [
            (tokens, position_ids) for (tokens, _), position_ids in zip(target_latents, positions)
        ]
        self.sample_names = [path.stem for path in image_paths]
        self.trigger_embedding = trigger_embedding

    def __len__(self) -> int:
        return len(self.sample_names)


__all__ = ["Krea2Dataset", "target_size"]
