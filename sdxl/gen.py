#!/usr/bin/env python
"""Generate one image from a prepared SDXL model folder.

The model folder comes from `python -m sdxl.convert`. The UNet is read from
`<model>/unet/<method>_quant.safetensors` by default, so run
`python -m sdxl.denoiser.quant` first. Pass `--no-quant` to use the fp16 file.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL.PngImagePlugin import PngInfo
from diffusers import AutoencoderKL
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from sdxl.config import DEFAULT_MODEL_NAME, resolve_model_root, resolve_outputs_dir
from sdxl.denoiser.loader import load_sdxl_denoiser
from sdxl.pipeline import StableDiffusionXLPipeline

DEFAULT_PROMPT = (
    "A propaganda poster depicting a cat dressed as french emperor napoleon "
    "holding a piece of cheese."
)
DEFAULT_SEED = 19930625
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_STEPS = 28
DEFAULT_CFG = 5.0
DEFAULT_METHOD = "sym-high"


def _coerce_dimension(value: int) -> int:
    """Round a pixel size down to a multiple of 8.

    Args:
        value: The requested size.

    Returns:
        The size rounded down to a multiple of 8, and at least 64.
    """
    return max(64, (value // 8) * 8)


def _output_path(output_dir: str | Path, seed: int) -> Path:
    """Return a new image path inside the output folder.

    Args:
        output_dir: The output folder. It is created if missing.
        seed: The noise seed, used in the file name.

    Returns:
        A path that does not exist yet.
    """
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    for index in range(1, 100):
        candidate = directory / f"{stamp}-{seed:08d}-{index:02d}.png"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot pick a free file name in {directory}.")


@torch.inference_mode()
def generate_image(
    *,
    root: str | Path,
    name: str = DEFAULT_MODEL_NAME,
    output_dir: str | Path | None = None,
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = "",
    seed: int = DEFAULT_SEED,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    steps: int = DEFAULT_STEPS,
    cfg: float = DEFAULT_CFG,
    method: str = DEFAULT_METHOD,
    no_quant: bool = False,
    offload: bool = False,
) -> Path:
    """Generate one image and save it as a PNG.

    Args:
        root: The data root.
        name: The model folder name. Defaults to `base`.
        output_dir: The folder that receives the PNG. Defaults to
            `<root>/sdxl/outputs`.
        prompt: The positive prompt.
        negative_prompt: The negative prompt.
        seed: The noise seed.
        height: The image height in pixels.
        width: The image width in pixels.
        steps: The number of denoising steps.
        cfg: The classifier-free guidance scale.
        method: The quantization method. The UNet is read from
            `<root>/sdxl/<name>/unet/<method>_quant.safetensors`.
        no_quant: Read the fp16 UNet checkpoint instead of the quantized one.
        offload: Stream `down_blocks`, `mid_block`, and `up_blocks` to the GPU
            one at a time. Peak UNet memory is the largest block (`up_blocks`,
            about 2.6 GB) instead of the whole checkpoint (about 5.1 GB).

    Returns:
        The path of the saved PNG.

    Raises:
        RuntimeError: If no CUDA device is available.
        FileNotFoundError: If the model folder or a checkpoint is missing.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("SDXL generation needs a CUDA device.")

    model_root = resolve_model_root(root, name)
    if not model_root.is_dir():
        raise FileNotFoundError(f"Model folder not found: {model_root}")
    if output_dir is None:
        output_dir = resolve_outputs_dir(root)

    height = _coerce_dimension(height)
    width = _coerce_dimension(width)
    device = torch.device("cuda")

    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    print("Loading tokenizers ...", flush=True)
    tokenizer = CLIPTokenizer.from_pretrained(
        str(model_root / "tokenizer"), local_files_only=True
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        str(model_root / "tokenizer_2"), local_files_only=True
    )

    print("Loading text encoders ...", flush=True)
    text_encoder = (
        CLIPTextModel.from_pretrained(
            str(model_root / "text_encoder"),
            dtype=torch.float16,
            local_files_only=True,
        )
        .to(device=device, dtype=torch.float16)
        .eval()
    )
    text_encoder_2 = (
        CLIPTextModelWithProjection.from_pretrained(
            str(model_root / "text_encoder_2"),
            dtype=torch.float16,
            local_files_only=True,
        )
        .to(device=device, dtype=torch.float16)
        .eval()
    )

    print("Loading scheduler ...", flush=True)
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        str(model_root / "scheduler"), local_files_only=True
    )
    scheduler.set_timesteps(steps, device=device)

    print("Loading UNet ...", flush=True)
    unet = load_sdxl_denoiser(model_root, quant_method=None if no_quant else method)
    if offload:
        unet.enable_block_offload(device)
        print("    offload: streaming down_blocks, mid_block, up_blocks", flush=True)

    print("Loading VAE ...", flush=True)
    vae = AutoencoderKL.from_pretrained(
        str(model_root / "vae"),
        torch_dtype=torch.float32,
        local_files_only=True,
    )
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

    print(f"Generating {width}x{height}, {steps} steps, cfg {cfg}, seed {seed} ...", flush=True)
    images = pipe.generate(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=cfg,
        negative_prompt=negative_prompt,
        generator=generator,
        gpu_device=device,
    )
    image = images[0] if isinstance(images, (list, tuple)) else images

    output_path = _output_path(output_dir, seed)
    metadata = {
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "width": width,
        "height": height,
        "method": "none" if no_quant else method,
        "offload": offload,
        "prompt": prompt,
        "negative": negative_prompt,
        "scheduler": "EulerAncestralDiscreteScheduler",
    }
    pnginfo = PngInfo()
    pnginfo.add_text("sdxl", json.dumps(metadata))
    image.save(output_path, pnginfo=pnginfo)
    print(f"Saved {output_path}", flush=True)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate an image from a prepared SDXL model folder.")
    parser.add_argument(
        "--root",
        required=True,
        help="Data root. The model is read from <root>/sdxl/<name>.",
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_MODEL_NAME,
        help=f"Model folder name. Default: {DEFAULT_MODEL_NAME}.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output folder. Default: <root>/sdxl/outputs.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Positive prompt.")
    parser.add_argument("--negative", default="", help="Negative prompt. Default: empty.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Noise seed. Default: {DEFAULT_SEED}.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help=f"Image width. Default: {DEFAULT_WIDTH}.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help=f"Image height. Default: {DEFAULT_HEIGHT}.")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help=f"Denoising steps. Default: {DEFAULT_STEPS}.")
    parser.add_argument("--cfg", type=float, default=DEFAULT_CFG, help=f"CFG scale. Default: {DEFAULT_CFG}.")
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
        "--offload",
        action="store_true",
        help=(
            "Stream down_blocks, mid_block, and up_blocks to the GPU one at a time. "
            "Use this when the UNet does not fit in VRAM."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the generator from the command line."""
    args = parse_args()
    generate_image(
        root=args.root,
        name=args.name,
        output_dir=args.output,
        prompt=args.prompt,
        negative_prompt=args.negative,
        seed=args.seed,
        height=args.height,
        width=args.width,
        steps=args.steps,
        cfg=args.cfg,
        method=args.method,
        no_quant=args.no_quant,
        offload=args.offload,
    )


if __name__ == "__main__":
    main()
