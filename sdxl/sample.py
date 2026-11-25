#!/usr/bin/env python
"""Render training samples for saved SDXL LoRA checkpoints.

Run from the repository root:

    python -m sdxl.sample --root "/iml_root" --project "/iml_root/sdxl/loras/niji" --sample-indices 1,2,3

The project folder uses the Flux layout: `images/` holds the captions, `models/`
holds `<step>.safetensors`, and the images go to `samples/`. The model is loaded
once. Each checkpoint is loaded into the same LoRA modules, so only the weights
change between checkpoints.

Index `n` is the n-th file of `sorted(<project>/images)`, the same order the
trainer and `steps.csv` use.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from PIL.PngImagePlugin import PngInfo

from flux2.sample import _list_step_checkpoints, _parse_sample_indices
from sdxl.config import DEFAULT_MODEL_NAME, resolve_model_root

DEFAULT_SAMPLE_SEED = 19930625
DEFAULT_SAMPLE_STEPS = 32
DEFAULT_CFG = 5.0
DEFAULT_TARGET_RESOLUTION = 512
DEFAULT_PROMPT_RESOLUTION = 1024
DEFAULT_METHOD = "q8"


@dataclass
class Sample:
    """One captioned image to render."""

    index: int
    name: str
    prompt: str
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Render training samples for saved SDXL LoRA checkpoints."
    )
    parser.add_argument("--root", type=Path, required=True, help="Data root. The model is read from <root>/sdxl/<name>.")
    parser.add_argument("--name", default=DEFAULT_MODEL_NAME, help=f"Model folder name. Default: {DEFAULT_MODEL_NAME}.")
    parser.add_argument("--project", type=Path, required=True, help="Training project directory. Holds images and models.")
    parser.add_argument("--prompt", type=str, help="Prompt that replaces the dataset captions.")
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_PROMPT_RESOLUTION,
        help=f"Prompt-mode sample width. Default: {DEFAULT_PROMPT_RESOLUTION}.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_PROMPT_RESOLUTION,
        help=f"Prompt-mode sample height. Default: {DEFAULT_PROMPT_RESOLUTION}.",
    )
    parser.add_argument(
        "--target-res",
        dest="target_res",
        type=int,
        default=DEFAULT_TARGET_RESOLUTION,
        help=f"Short side of a dataset sample. Default: {DEFAULT_TARGET_RESOLUTION}.",
    )
    parser.add_argument("--start", type=int, default=1, help="First step checkpoint number to render.")
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_SAMPLE_STEPS,
        help=f"Denoising steps. Default: {DEFAULT_SAMPLE_STEPS}.",
    )
    parser.add_argument("--cfg", type=float, default=DEFAULT_CFG, help=f"CFG scale. Default: {DEFAULT_CFG}.")
    parser.add_argument("--negative", default="", help="Negative prompt. Default: empty.")
    parser.add_argument(
        "--sample-seed",
        "--sample_seed",
        dest="sample_seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help=f"Base seed. Each sample mixes it with its file name. Default: {DEFAULT_SAMPLE_SEED}.",
    )
    parser.add_argument(
        "--sample-indices",
        dest="sample_indices",
        type=str,
        default=None,
        help="Comma-separated 1-based dataset indices. Default: every image.",
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        help=f"Quantization method of the UNet checkpoint. Default: {DEFAULT_METHOD}.",
    )
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="Read the fp16 UNet checkpoint instead of the quantized one.",
    )
    parser.add_argument(
        "--stream-denoiser",
        dest="stream_denoiser",
        action="store_true",
        help=(
            "Keep the quantized weights in pinned host memory and copy each layer to the GPU "
            "for its own multiply. Needs a quantized checkpoint."
        ),
    )
    return parser.parse_args()


def _read_caption(image_path: Path) -> str:
    """Return the caption of an image.

    Args:
        image_path: The image file.

    Returns:
        The caption text.

    Raises:
        FileNotFoundError: If the `.txt` file is missing.
    """
    caption_path = image_path.with_suffix(".txt")
    if not caption_path.is_file():
        raise FileNotFoundError(f"Caption file not found: {caption_path}")
    return caption_path.read_text(encoding="utf-8").strip()


def _image_size(path: Path) -> tuple[int, int]:
    """Return the pixel size of an image.

    Args:
        path: The image file.

    Returns:
        The `(width, height)` pair.
    """
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def _collect_samples(
    *,
    images_dir: Path,
    prompt: str,
    sample_indices: list[int] | None,
    target_res: int,
    prompt_width: int,
    prompt_height: int,
    prompt_mode: bool,
) -> list[Sample]:
    """Return the samples to render, one per selected dataset index.

    Args:
        images_dir: The folder that holds the images and captions.
        prompt: The prompt that replaces every caption. Empty outside prompt
            mode.
        sample_indices: The 1-based dataset indices to render. None means every
            image.
        target_res: The short side of a dataset sample.
        prompt_width: The width in prompt mode.
        prompt_height: The height in prompt mode.
        prompt_mode: True when a `--prompt` replaces the captions.

    Returns:
        The samples, in index order.

    Raises:
        ValueError: If an index is outside the dataset.
    """
    from sdxl.train.dataset import IMAGE_SUFFIXES, target_size

    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No images in {images_dir}")

    indices = (
        list(range(1, len(image_paths) + 1)) if sample_indices is None else sample_indices
    )
    for index in indices:
        if index < 1 or index > len(image_paths):
            raise ValueError(
                f"Sample index {index} is outside the dataset. {images_dir} holds "
                f"{len(image_paths)} image(s), so the indices run 1 to {len(image_paths)}."
            )

    samples: list[Sample] = []
    for index in indices:
        image_path = image_paths[index - 1]
        if prompt_mode:
            width, height = prompt_width, prompt_height
        else:
            width, height = target_size(*_image_size(image_path), target_res)
        samples.append(
            Sample(
                index=index,
                name=image_path.stem,
                prompt=prompt if prompt_mode else _read_caption(image_path),
                width=width,
                height=height,
            )
        )
    return samples


def _output_name(sample_dir: Path, step: int, sample: Sample) -> Path:
    """Return the image path of one sample.

    Args:
        sample_dir: The `samples` folder.
        step: The checkpoint step.
        sample: The sample to render.

    Returns:
        The path that holds this checkpoint's sample.
    """
    return sample_dir / f"{step:06d}_{sample.index}_{sample.width}x{sample.height}.png"


def main() -> None:
    """Render every checkpoint sample and save it as a PNG."""
    args = parse_args()

    prompt = None if args.prompt is None else args.prompt.strip()
    if prompt == "":
        raise ValueError("--prompt must not be empty.")
    prompt_mode = prompt is not None

    project_path = args.project.expanduser().resolve()
    images_dir = project_path / "images"
    checkpoints_dir = project_path / "models"
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"Expected models directory at {checkpoints_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Expected images directory at {images_dir}")

    checkpoints = [
        (step_number, checkpoint_path)
        for step_number, checkpoint_path in _list_step_checkpoints(checkpoints_dir)
        if step_number >= args.start
    ]
    if not checkpoints:
        raise FileNotFoundError(
            f"No step checkpoints found at or after step {args.start} in {checkpoints_dir}"
        )

    parsed_indices = _parse_sample_indices(args.sample_indices)

    print("1. Read dataset ...", flush=True)
    dataset_start = time.perf_counter()
    samples = _collect_samples(
        images_dir=images_dir,
        prompt=prompt or "",
        sample_indices=parsed_indices,
        target_res=args.target_res,
        prompt_width=args.width,
        prompt_height=args.height,
        prompt_mode=prompt_mode,
    )
    sample_dir = project_path / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    print(f"  * mode: {'prompt' if prompt_mode else 'captioned'}", flush=True)
    print(f"  * selected samples: {', '.join(str(sample.index) for sample in samples)}", flush=True)
    print(
        "  * sample resolutions: "
        + ", ".join(f"{sample.index}={sample.width}x{sample.height}" for sample in samples),
        flush=True,
    )
    print(f"  * done in {time.perf_counter() - dataset_start:.3f}s", flush=True)
    if not samples:
        print("  * no samples requested", flush=True)
        return
    print(f"  * checkpoints: {len(checkpoints)}", flush=True)

    import torch

    from diffusers import AutoencoderKL
    from diffusers.schedulers import EulerAncestralDiscreteScheduler
    from flux2.lora.qmodel import inject_trainable_lora_modules
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    from sdxl.denoiser.loader import load_sdxl_denoiser
    from sdxl.lora.config import SDXL_LORA_TARGETS
    from sdxl.pipeline import StableDiffusionXLPipeline
    from sdxl.train.main import _load_lora_state, _read_metadata
    from utils.quant.streaming import move_unstreamed_modules_to
    from utils.seed import image_seed

    if not torch.cuda.is_available():
        raise RuntimeError("SDXL sampling needs a CUDA device.")
    device = torch.device("cuda")

    model_root = resolve_model_root(args.root, args.name)
    metadata = _read_metadata(checkpoints[0][1])
    if "rank" not in metadata or "alpha" not in metadata:
        raise RuntimeError(
            f"Checkpoint {checkpoints[0][1].name} has no rank/alpha metadata. "
            "Sample a checkpoint written by sdxl.train.main."
        )
    rank = int(metadata["rank"])
    alpha = int(metadata["alpha"])

    print("2. Load the model ...", flush=True)
    model_start = time.perf_counter()
    print(f"  * lora rank/alpha from checkpoint: r={rank}, a={alpha}", flush=True)

    tokenizer = CLIPTokenizer.from_pretrained(str(model_root / "tokenizer"), local_files_only=True)
    tokenizer_2 = CLIPTokenizer.from_pretrained(str(model_root / "tokenizer_2"), local_files_only=True)
    text_encoder = (
        CLIPTextModel.from_pretrained(str(model_root / "text_encoder"), dtype=torch.float16, local_files_only=True)
        .to(device=device, dtype=torch.float16)
        .eval()
    )
    text_encoder_2 = (
        CLIPTextModelWithProjection.from_pretrained(
            str(model_root / "text_encoder_2"), dtype=torch.float16, local_files_only=True
        )
        .to(device=device, dtype=torch.float16)
        .eval()
    )
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(str(model_root / "scheduler"), local_files_only=True)
    scheduler.set_timesteps(args.steps, device=device)

    unet = load_sdxl_denoiser(
        model_root,
        quant_method=None if args.no_quant else args.method,
        stream_denoiser=args.stream_denoiser,
    )
    injected = inject_trainable_lora_modules(
        unet,
        target_linear_names=SDXL_LORA_TARGETS,
        rank=rank,
        alpha=alpha,
    )
    if args.stream_denoiser:
        move_unstreamed_modules_to(unet, device)
    else:
        unet = unet.to(device)
    print(f"  * injected lora modules: {len(injected)}", flush=True)

    vae = AutoencoderKL.from_pretrained(str(model_root / "vae"), torch_dtype=torch.float32, local_files_only=True)
    vae.enable_tiling()
    vae.to(device="cpu", dtype=torch.float32).eval()

    pipe = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=scheduler,
    )
    print(f"  * done in {time.perf_counter() - model_start:.3f}s", flush=True)

    print("3. Render checkpoint samples ...", flush=True)
    render_start = time.perf_counter()
    print(f"  * sample output dir: {sample_dir}", flush=True)
    for step_number, checkpoint_path in checkpoints:
        print(f"  * step {step_number}: {checkpoint_path.name}", flush=True)
        step_start = time.perf_counter()
        matched = _load_lora_state(unet, checkpoint_path)
        print(f"    * loaded {matched} lora modules", flush=True)

        for sample in samples:
            output_path = _output_name(sample_dir, step_number, sample)
            if output_path.is_file():
                print(f"    * skipped sample {sample.index}: {output_path}", flush=True)
                continue

            seed = image_seed(args.sample_seed, sample.name)
            generator = torch.Generator(device=device).manual_seed(seed)
            print(
                f"    * denoise sample {sample.index} at {sample.width}x{sample.height}, seed {seed} ...",
                flush=True,
            )
            images = pipe.generate(
                prompt=sample.prompt,
                height=sample.height,
                width=sample.width,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                negative_prompt=args.negative,
                generator=generator,
                gpu_device=device,
            )
            image = images[0] if isinstance(images, (list, tuple)) else images

            pnginfo = PngInfo()
            pnginfo.add_text(
                "sdxl",
                json.dumps(
                    {
                        "checkpoint": str(checkpoint_path),
                        "step": step_number,
                        "sample_index": sample.index,
                        "sample_name": sample.name,
                        "seed": seed,
                        "steps": args.steps,
                        "cfg": args.cfg,
                        "width": sample.width,
                        "height": sample.height,
                        "method": "fp16" if args.no_quant else args.method,
                        "prompt": sample.prompt,
                        "negative": args.negative,
                        "scheduler": "EulerAncestralDiscreteScheduler",
                    }
                ),
            )
            image.save(output_path, pnginfo=pnginfo)
            print(f"    * saved {output_path}", flush=True)

        torch.cuda.empty_cache()
        print(f"    * step sample time: {time.perf_counter() - step_start:.3f}s", flush=True)

    del unet, vae
    torch.cuda.empty_cache()
    print(f"  * sampling done in {time.perf_counter() - render_start:.3f}s", flush=True)


if __name__ == "__main__":
    main()
