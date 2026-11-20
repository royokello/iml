from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from sdxl.train.text import POOLED_KEY, PROMPT_EMBEDS_KEY, encode_caption

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
VAE_SCALE_FACTOR = 8
ALIGN = 32
DEFAULT_TARGET_RESOLUTION = 512

LATENT_KEY = "latent"
CAPTION_METADATA_KEY = "caption"


def target_size(
    width: int,
    height: int,
    target_resolution: int = DEFAULT_TARGET_RESOLUTION,
) -> tuple[int, int]:
    """Return the training size for an image, keeping its aspect ratio.

    The short side becomes `target_resolution` and both sides round to a
    multiple of 32, the same rule `flux2/train/dataset.py` applies with a
    multiple of 16.

    Args:
        width: The image width.
        height: The image height.
        target_resolution: The short side of the result.

    Returns:
        The `(width, height)` pair.

    Raises:
        ValueError: If a side is not positive, or the resolution is not
            positive.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size {width}x{height}.")
    if target_resolution <= 0:
        raise ValueError(f"Invalid target resolution {target_resolution}.")

    scale = target_resolution / min(width, height)
    new_width = max(ALIGN, round(width * scale / ALIGN) * ALIGN)
    new_height = max(ALIGN, round(height * scale / ALIGN) * ALIGN)
    return new_width, new_height


def _list_image_files(directory: Path) -> list[Path]:
    """Return the image files in a folder, sorted by name.

    Args:
        directory: The dataset folder.

    Returns:
        The image paths. The list is empty when the folder has no image.
    """
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _image_size(path: Path) -> tuple[int, int]:
    """Return the pixel size of an image.

    Args:
        path: The image file.

    Returns:
        The `(width, height)` pair.
    """
    with Image.open(path) as image:
        return image.size


def _prepare_pixels(
    path: Path,
    size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Resize an image and return it in `[-1, 1]`.

    Args:
        path: The image file.
        size: The `(width, height)` pair to resize to.
        device: The tensor device.

    Returns:
        A float32 tensor with shape `(1, 3, height, width)`.
    """
    with Image.open(path) as image:
        resized = image.convert("RGB").resize(size, Image.LANCZOS)

    array = np.asarray(resized, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


class SDXLDataset:
    """A cached SDXL training set: one latent, one text embedding per image.

    The text encoders and the VAE are loaded only when something must be
    encoded, and they are freed before the constructor returns.
    """

    def __init__(
        self,
        model_root: str | Path,
        dataset_dir: str | Path,
        target_resolution: int = DEFAULT_TARGET_RESOLUTION,
        target_upscale: bool = False,
        cache_images: bool = False,
        cache_text: bool = False,
        device: torch.device | str = "cuda",
    ):
        """Build the training set.

        Args:
            model_root: The model folder, holding `vae` and the tokenizers.
            dataset_dir: The folder with `<id>.png` and optional `<id>.txt`.
            target_resolution: The short side of every encoded image.
            target_upscale: Encode an image whose short side is below
                `target_resolution`. Without it, such an image is skipped.
            cache_images: Write and reuse `<id>.<res>.safetensors` latents.
            cache_text: Write and reuse `<id>.text.safetensors` embeddings.
            device: Where the encoders run during the cache pass.

        Raises:
            FileNotFoundError: If the dataset folder is missing.
            ValueError: If the folder has no usable image.
        """
        self.model_root = Path(model_root).expanduser().resolve()
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.device = torch.device(device)
        self.target_resolution = target_resolution
        self.target_upscale = target_upscale
        self.cache_images = cache_images
        self.cache_text = cache_text

        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {self.dataset_dir}")

        image_paths = _list_image_files(self.dataset_dir)
        if not image_paths:
            raise ValueError(f"No images found in {self.dataset_dir}")

        selected: list[Path] = []
        self.sample_sizes: list[tuple[int, int]] = []
        skipped = 0
        for path in image_paths:
            width, height = _image_size(path)
            if min(width, height) < target_resolution and not target_upscale:
                skipped += 1
                continue
            selected.append(path)
            self.sample_sizes.append(target_size(width, height, target_resolution))

        if not selected:
            raise ValueError(
                f"No image in {self.dataset_dir} has a short side of at least "
                f"{target_resolution} pixels. {skipped} image(s) skipped. "
                "Use --target-upscale to encode them anyway."
            )

        self.names = [path.stem for path in selected]
        print(
            f"Dataset: {len(selected)} image(s) at target resolution "
            f"{target_resolution}, {skipped} skipped",
            flush=True,
        )

        self.latents = self._build_latents(selected)
        self.prompt_embeds, self.pooled = self._build_text(selected)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.latents)

    def time_ids(self, index: int, dtype: torch.dtype) -> torch.Tensor:
        """Return the SDXL added time ids for one sample.

        Args:
            index: The sample index.
            dtype: The tensor dtype. Use the dtype of `prompt_embeds`.

        Returns:
            A `(1, 6)` tensor holding `[height, width, 0, 0, height, width]`.
        """
        width, height = self.sample_sizes[index]
        values = [height, width, 0, 0, height, width]
        return torch.tensor([values], dtype=dtype)

    def _latent_cache_path(self, image_path: Path) -> Path:
        """Return the latent cache path for one image.

        Args:
            image_path: The image file.

        Returns:
            `<dataset>/<id>.<target resolution>.safetensors`.
        """
        return self.dataset_dir / f"{image_path.stem}.{self.target_resolution}.safetensors"

    def _build_latents(self, image_paths: list[Path]) -> list[torch.Tensor]:
        """Return one fp16 latent per image, encoding only what is not cached.

        Args:
            image_paths: The image files, in dataset order.

        Returns:
            One CPU fp16 tensor per image, shape `(4, height / 8, width / 8)`.
        """
        latents: list[torch.Tensor] = []
        vae = None

        for index, image_path in enumerate(image_paths):
            width, height = self.sample_sizes[index]
            cache_path = self._latent_cache_path(image_path)

            if self.cache_images and cache_path.is_file():
                cached = load_file(str(cache_path), device="cpu").get(LATENT_KEY)
                if cached is not None and cached.ndim == 4 and cached.shape[0] == 1:
                    cached = cached.squeeze(0)
                if cached is not None and cached.ndim == 3 and tuple(cached.shape[-2:]) == (
                    height // VAE_SCALE_FACTOR,
                    width // VAE_SCALE_FACTOR,
                ):
                    latents.append(cached.to(dtype=torch.float16))
                    print(f"    [{index + 1}/{len(image_paths)}] {image_path.name}: cache")
                    continue

            if vae is None:
                vae = self._load_vae()

            start = time.perf_counter()
            pixels = _prepare_pixels(image_path, (width, height), self.device)
            with torch.no_grad():
                latent = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
            latent = latent.squeeze(0).to(device="cpu", dtype=torch.float16).contiguous()

            if self.cache_images:
                save_file({LATENT_KEY: latent}, str(cache_path))

            latents.append(latent)
            print(
                f"    [{index + 1}/{len(image_paths)}] {image_path.name}: "
                f"{width}x{height} in {time.perf_counter() - start:.2f}s"
            )

        if vae is not None:
            del vae
            torch.cuda.empty_cache()

        return latents

    def _build_text(self, image_paths: list[Path]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return one embedding and one pooled vector per image.

        Args:
            image_paths: The image files, in dataset order.

        Returns:
            Two lists of CPU fp16 tensors: shapes `(77, 2048)` and `(1280,)`.
        """
        prompt_embeds: list[torch.Tensor] = []
        pooled_embeds: list[torch.Tensor] = []
        tokenizer = None
        tokenizer_2 = None
        text_encoder = None
        text_encoder_2 = None

        for index, image_path in enumerate(image_paths):
            caption = _read_caption(image_path)
            cache_path = self.dataset_dir / f"{image_path.stem}.text.safetensors"

            if self.cache_text and cache_path.is_file() and _caption_matches(cache_path, caption):
                tensors = load_file(str(cache_path), device="cpu")
                prompt_embeds.append(tensors[PROMPT_EMBEDS_KEY].to(dtype=torch.float16))
                pooled_embeds.append(tensors[POOLED_KEY].to(dtype=torch.float16))
                print(f"    [{index + 1}/{len(image_paths)}] {image_path.name}: text cache")
                continue

            if text_encoder is None:
                tokenizer, tokenizer_2, text_encoder, text_encoder_2 = self._load_encoders()

            start = time.perf_counter()
            embed, pooled = encode_caption(
                caption,
                tokenizer,
                tokenizer_2,
                text_encoder,
                text_encoder_2,
                device=self.device,
            )

            if self.cache_text:
                save_file(
                    {PROMPT_EMBEDS_KEY: embed, POOLED_KEY: pooled},
                    str(cache_path),
                    metadata={CAPTION_METADATA_KEY: caption},
                )

            prompt_embeds.append(embed)
            pooled_embeds.append(pooled)
            print(
                f"    [{index + 1}/{len(image_paths)}] {image_path.name}: "
                f"text in {time.perf_counter() - start:.2f}s"
            )

        if text_encoder is not None:
            del tokenizer, tokenizer_2, text_encoder, text_encoder_2
            torch.cuda.empty_cache()

        return prompt_embeds, pooled_embeds

    def _load_vae(self) -> AutoencoderKL:
        """Load the VAE in fp32 with tiling.

        Returns:
            The VAE on `self.device`.
        """
        print("Loading VAE ...", flush=True)
        vae = AutoencoderKL.from_pretrained(
            str(self.model_root / "vae"),
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        vae.enable_tiling()
        return vae.to(device=self.device, dtype=torch.float32).eval()

    def _load_encoders(self):
        """Load both tokenizers and both text encoders in fp16.

        Returns:
            `(tokenizer, tokenizer_2, text_encoder, text_encoder_2)`.
        """
        print("Loading text encoders ...", flush=True)
        tokenizer = CLIPTokenizer.from_pretrained(
            str(self.model_root / "tokenizer"), local_files_only=True
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            str(self.model_root / "tokenizer_2"), local_files_only=True
        )
        text_encoder = (
            CLIPTextModel.from_pretrained(
                str(self.model_root / "text_encoder"),
                dtype=torch.float16,
                local_files_only=True,
            )
            .to(device=self.device, dtype=torch.float16)
            .eval()
        )
        text_encoder_2 = (
            CLIPTextModelWithProjection.from_pretrained(
                str(self.model_root / "text_encoder_2"),
                dtype=torch.float16,
                local_files_only=True,
            )
            .to(device=self.device, dtype=torch.float16)
            .eval()
        )
        return tokenizer, tokenizer_2, text_encoder, text_encoder_2


def _read_caption(image_path: Path) -> str:
    """Return the caption of an image, or an empty string.

    Args:
        image_path: The image file.

    Returns:
        The stripped text of the sibling `.txt` file.
    """
    caption_path = image_path.with_suffix(".txt")
    if not caption_path.is_file():
        return ""
    return caption_path.read_text(encoding="utf-8").strip()


def _caption_matches(cache_path: Path, caption: str) -> bool:
    """Return True when a text cache belongs to a caption.

    Args:
        cache_path: The `.text.safetensors` file.
        caption: The current caption.

    Returns:
        True when the file metadata holds the same caption.
    """
    with safe_open(str(cache_path), framework="pt") as handle:
        metadata = handle.metadata() or {}
    return metadata.get(CAPTION_METADATA_KEY) == caption


__all__ = ["DEFAULT_TARGET_RESOLUTION", "SDXLDataset", "target_size"]
