#!/usr/bin/env python
"""Convert a single-file SDXL checkpoint into a prepared model folder.

The output folder holds the UNet, both text encoders, the VAE, both
tokenizers, and the scheduler. Later tools read this folder.

Diffusers 0.38 converts single-file CLIP weights through
`CLIPTextModel.text_model`. Transformers 5 removed that attribute, so the
text encoders are converted and loaded here instead of through
`StableDiffusionXLPipeline.from_single_file`.
"""

from __future__ import annotations

import argparse
import gc
import shutil
from pathlib import Path

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.loaders.single_file_utils import (
    convert_ldm_clip_checkpoint,
    convert_open_clip_checkpoint,
)
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from safetensors import safe_open
from transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from sdxl.config import MODEL_COMPONENTS, validate_model_root

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


def convert_checkpoint(
    input_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Convert an SDXL checkpoint into a prepared model folder.

    Args:
        input_path: The single-file `.safetensors` checkpoint.
        output_path: The model folder. It is created if missing.
        overwrite: Replace the output folder when it is not empty.

    Returns:
        The prepared model folder.

    Raises:
        FileNotFoundError: If the checkpoint is missing.
        ValueError: If the checkpoint is not a `.safetensors` file, or if the
            output path is an existing file.
        FileExistsError: If the output folder is not empty and `overwrite` is
            false.
    """
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
        "Converting SDXL checkpoint:\n"
        f"  input={in_path}\n"
        f"  output={out_path}",
        flush=True,
    )

    print("UNet ...", flush=True)
    unet = UNet2DConditionModel.from_single_file(
        str(in_path),
        config=CONFIG_REPO,
        subfolder="unet",
        torch_dtype=torch.float16,
    )
    unet.save_pretrained(
        str(out_path / "unet"),
        safe_serialization=True,
        max_shard_size=MAX_SHARD_SIZE,
    )
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
        description="Convert a single-file SDXL checkpoint into a prepared model folder."
    )
    parser.add_argument("--input", required=True, help="Path to the SDXL .safetensors checkpoint.")
    parser.add_argument(
        "--output",
        required=True,
        help="Model folder. It is created if missing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output folder when it is not empty.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the conversion from the command line."""
    args = parse_args()
    target = convert_checkpoint(
        args.input,
        args.output,
        overwrite=args.overwrite,
    )
    print(f"Done: {target}")


if __name__ == "__main__":
    main()
