#!/usr/bin/env python
"""Quantize an SDXL checkpoint into a prepared model folder.

One command does the whole job. It reads a single-file `.safetensors`
checkpoint and writes `<root>/sdxl/<name>`: the quantized UNet, both text
encoders, the VAE, both tokenizers, and the scheduler. Later tools read that
folder.

The UNet is converted and quantized in memory. The quantized file holds every
UNet tensor, so the fp16 UNet checkpoint is never written unless
`--keep-fp16` is set. That saves one 5.1 GB file and a full read.

The default method is `q8`. Any other name from `utils.quant` is passed
through as-is.

Diffusers 0.38 converts single-file CLIP weights through
`CLIPTextModel.text_model`. Transformers 5 removed that attribute, so the
text encoders are converted and loaded here instead of through
`StableDiffusionXLPipeline.from_single_file`.
"""

from __future__ import annotations

import argparse
import gc
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.loaders.single_file_utils import (
    convert_ldm_clip_checkpoint,
    convert_open_clip_checkpoint,
)
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from sdxl.config import DEFAULT_MODEL_NAME, MODEL_COMPONENTS, resolve_model_root, validate_model_root
from sdxl.denoiser.targets import build_sdxl_denoiser_target_tensors
from utils.quant.model import _build_quantized_or_fp16_tensors
from utils.quant.targets import build_mixed_target_config

DEFAULT_METHOD = "q8"

CONFIG_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
CONDITIONER_PREFIX = "conditioner."
CLIP_G_PREFIX = "conditioner.embedders.1.model."
MAX_SHARD_SIZE = "10GB"

# The SDXL noise schedule, copied from the reference scheduler config.
SCHEDULER_BETA_START = 0.00085
SCHEDULER_BETA_END = 0.012
SCHEDULER_BETA_SCHEDULE = "scaled_linear"
SCHEDULER_STEPS_OFFSET = 1
SCHEDULER_TIMESTEP_SPACING = "leading"
SCHEDULER_PREDICTION_TYPE = "epsilon"


def _is_non_empty(directory: Path) -> bool:
    """Return True when the folder exists and holds at least one entry.

    Args:
        directory: The folder to check.

    Returns:
        True when the folder has entries.
    """
    return directory.is_dir() and any(directory.iterdir())


def _format_bytes(value: int) -> str:
    """Format a byte count with a binary unit.

    Args:
        value: The byte count.

    Returns:
        The size, for example `5.1GiB`.
    """
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}TiB"


def _folder_bytes(directory: Path) -> int:
    """Return the total size of the files in a folder.

    Args:
        directory: The folder to measure. Subfolders are included.

    Returns:
        The byte count.
    """
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _load_conditioner_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Read the CLIP weights from a single-file checkpoint.

    Args:
        checkpoint_path: The SDXL checkpoint.

    Returns:
        The tensors whose names start with `conditioner.`.
    """
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        keys = [key for key in handle.keys() if key.startswith(CONDITIONER_PREFIX)]
        return {key: handle.get_tensor(key) for key in keys}


def _apply_state(model: torch.nn.Module, state: dict[str, torch.Tensor], label: str) -> None:
    """Load a state dict and fail on any key mismatch.

    Transformers 5 keeps the `text_model.` prefix on
    `CLIPTextModelWithProjection` but drops it on `CLIPTextModel`. A key is
    therefore accepted with or without that prefix.

    Args:
        model: The target model.
        state: The converted state dict.
        label: The component name, used in the error message.

    Raises:
        RuntimeError: If a key is missing or unexpected.
    """
    model_keys = set(model.state_dict())
    matched: dict[str, torch.Tensor] = {}
    unmatched: list[str] = []
    for key, tensor in state.items():
        target = key if key in model_keys else key.removeprefix("text_model.")
        if target in model_keys:
            matched[target] = tensor
        else:
            unmatched.append(key)

    missing, unexpected = model.load_state_dict(matched, strict=False)
    if missing or unexpected or unmatched:
        raise RuntimeError(
            f"{label} state mismatch. "
            f"Missing: {sorted(missing)}. "
            f"Unexpected: {sorted(unexpected)}. "
            f"Unmatched source keys: {sorted(unmatched)}."
        )


def _load_clip_l(checkpoint: dict[str, torch.Tensor]) -> CLIPTextModel:
    """Build CLIP-L from the checkpoint weights.

    Args:
        checkpoint: The `conditioner.*` tensors.

    Returns:
        The text encoder in fp16 eval mode.
    """
    config = CLIPTextConfig.from_pretrained(CONFIG_REPO, subfolder="text_encoder")
    model = CLIPTextModel(config)
    _apply_state(model, convert_ldm_clip_checkpoint(checkpoint), "text_encoder")
    return model.eval().half()


def _load_clip_g(checkpoint: dict[str, torch.Tensor]) -> CLIPTextModelWithProjection:
    """Build CLIP-G from the checkpoint weights.

    Args:
        checkpoint: The `conditioner.*` tensors.

    Returns:
        The text encoder with projection in fp16 eval mode.
    """
    config = CLIPTextConfig.from_pretrained(CONFIG_REPO, subfolder="text_encoder_2")
    model = CLIPTextModelWithProjection(config)
    state = convert_open_clip_checkpoint(model, checkpoint, prefix=CLIP_G_PREFIX)
    _apply_state(model, state, "text_encoder_2")
    return model.eval().half()


def _quantize_unet(
    unet: UNet2DConditionModel,
    out_dir: Path,
    method: str,
) -> Path:
    """Quantize the UNet weights in memory and save them to one file.

    The saved file holds every tensor of the UNet. Linear weights named by the
    quantization targets are quantized, and the rest are copied and downcast to
    fp16, so the file is complete and the fp16 checkpoint is not needed.

    Args:
        unet: The converted UNet, in fp16.
        out_dir: The `unet` folder of the model.
        method: A quantization method name, for example `q8`.

    Returns:
        The path of the quantized checkpoint.
    """
    state = unet.state_dict()
    linear_names = [
        name for name, tensor in state.items() if name.endswith(".weight") and tensor.ndim == 2
    ]
    target_tensors = build_sdxl_denoiser_target_tensors(linear_names)
    targets = build_mixed_target_config(method, target_tensors)
    methods_by_name = {
        name: target_method
        for target_method, target_names in targets.items()
        for name in target_names
    }

    print(f"Quantizing UNet weights with {method} ...", flush=True)
    result: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        result.update(
            _build_quantized_or_fp16_tensors(name, tensor, methods_by_name.get(name))
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{method.replace('-', '_')}_quant.safetensors"
    print(f"Saving {output_path} ...", flush=True)
    save_file(
        result,
        str(output_path),
        metadata={
            "model": "sdxl",
            "component": "denoiser",
            "method": method,
            "application": "IML",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return output_path


def quantize_checkpoint(
    input_path: str | Path,
    output_path: str | Path,
    *,
    method: str = DEFAULT_METHOD,
    keep_fp16: bool = False,
    overwrite: bool = False,
) -> Path:
    """Quantize an SDXL checkpoint into a prepared model folder.

    Args:
        input_path: The single-file `.safetensors` checkpoint.
        output_path: The model folder. It is created if missing.
        method: A quantization method name. The default is `q8`.
        keep_fp16: Also write the fp16 UNet checkpoint.
        overwrite: Replace the output folder when it is not empty.

    Returns:
        The prepared model folder.

    Raises:
        FileNotFoundError: If the checkpoint is missing.
        ValueError: If the checkpoint is not a `.safetensors` file, or if the
            output path is an existing file.
        FileExistsError: If the output folder is not empty and `overwrite` is
            false.
        RuntimeError: If CUDA is not available, because the quantizers run on
            the GPU.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Quantization needs a CUDA device.")

    in_path = Path(input_path).expanduser().resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Input checkpoint not found: {in_path}")
    if in_path.suffix.lower() != ".safetensors":
        raise ValueError(f"Expected a .safetensors checkpoint: {in_path}")

    out_path = Path(output_path).expanduser().resolve()
    if out_path.is_file():
        raise ValueError(f"Output path is a file, not a folder: {out_path}")
    if _is_non_empty(out_path):
        if not overwrite:
            raise FileExistsError(
                f"Output folder is not empty: {out_path}. Pass --overwrite to replace it."
            )
        print(f"Removing existing output folder: {out_path}", flush=True)
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    print(
        "Quantizing SDXL checkpoint:\n"
        f"  input={in_path}\n"
        f"  output={out_path}\n"
        f"  method={method}",
        flush=True,
    )

    print("UNet ...", flush=True)
    unet = UNet2DConditionModel.from_single_file(
        str(in_path),
        config=CONFIG_REPO,
        subfolder="unet",
        torch_dtype=torch.float16,
    )
    _quantize_unet(unet, out_path / "unet", method)
    if keep_fp16:
        unet.save_pretrained(
            str(out_path / "unet"),
            safe_serialization=True,
            max_shard_size=MAX_SHARD_SIZE,
        )
    else:
        (out_path / "unet").mkdir(parents=True, exist_ok=True)
        unet.save_config(str(out_path / "unet"))
    del unet
    gc.collect()

    print("VAE ...", flush=True)
    vae = AutoencoderKL.from_single_file(
        str(in_path),
        config=CONFIG_REPO,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    vae.save_pretrained(str(out_path / "vae"), safe_serialization=True)
    del vae
    gc.collect()

    print("Text encoders ...", flush=True)
    checkpoint = _load_conditioner_state(in_path)
    text_encoder = _load_clip_l(checkpoint)
    text_encoder.save_pretrained(str(out_path / "text_encoder"), safe_serialization=True)
    del text_encoder
    gc.collect()

    text_encoder_2 = _load_clip_g(checkpoint)
    text_encoder_2.save_pretrained(str(out_path / "text_encoder_2"), safe_serialization=True)
    del text_encoder_2, checkpoint
    gc.collect()

    print("Tokenizers ...", flush=True)
    for name in ("tokenizer", "tokenizer_2"):
        tokenizer = CLIPTokenizer.from_pretrained(CONFIG_REPO, subfolder=name)
        tokenizer.save_pretrained(str(out_path / name))

    scheduler = EulerAncestralDiscreteScheduler(
        beta_start=SCHEDULER_BETA_START,
        beta_end=SCHEDULER_BETA_END,
        beta_schedule=SCHEDULER_BETA_SCHEDULE,
        steps_offset=SCHEDULER_STEPS_OFFSET,
        timestep_spacing=SCHEDULER_TIMESTEP_SPACING,
        prediction_type=SCHEDULER_PREDICTION_TYPE,
    )
    scheduler.save_pretrained(str(out_path / "scheduler"))

    validate_model_root(out_path)

    print("Prepared layout:", flush=True)
    for name in MODEL_COMPONENTS:
        print(f"  {name:<16}{_format_bytes(_folder_bytes(out_path / name))}", flush=True)
    print(f"  {'total':<16}{_format_bytes(_folder_bytes(out_path))}", flush=True)
    return out_path


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Quantize an SDXL checkpoint into a prepared model folder."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the SDXL .safetensors checkpoint.",
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Data root. The model folder becomes <root>/sdxl/<name>.",
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_MODEL_NAME,
        help=f"Model folder name under <root>/sdxl. Default: {DEFAULT_MODEL_NAME}.",
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        help=f"Quantization method. Default: {DEFAULT_METHOD}.",
    )
    parser.add_argument(
        "--keep-fp16",
        action="store_true",
        help="Write the fp16 UNet checkpoint as well, for --no-quant generation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output folder when it is not empty.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the quantizer from the command line."""
    args = parse_args()
    output = resolve_model_root(args.root, args.name)
    target = quantize_checkpoint(
        args.input,
        output,
        method=args.method,
        keep_fp16=args.keep_fp16,
        overwrite=args.overwrite,
    )
    print(f"Done: {target}")


if __name__ == "__main__":
    main()
