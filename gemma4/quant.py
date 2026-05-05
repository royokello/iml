#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gemma4.config import (
    _GEMMA4_QUANT_CONFIGS,
    _GEMMA4_QUANT_METHODS,
    _KV_PROJECTION_WEIGHT_SUFFIXES,
    _LANGUAGE_PER_LAYER_TOKEN_EMBED_WEIGHT,
    _LANGUAGE_TOKEN_EMBED_WEIGHT,
    _NUM_LANGUAGE_KV_PROJECTION_LAYERS,
    _NUM_LANGUAGE_LAYERS,
)
from utils.quant.model import quantize_model_tensors

_MODEL_DIR = "gemma4"
_CHECKPOINT_NAME = "model.safetensors"
_DEFAULT_QUANT_METHOD = "high"

_AUDIO_PREFIXES = (
    "model.audio_tower.",
    "model.embed_audio.",
)
_VISION_PREFIXES = (
    "model.vision_tower.",
    "model.embed_vision.",
)
_LANGUAGE_PREFIXES = (
    "model.language_model.",
)


def _matches_any_prefix(name: str, prefixes: Iterable[str]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def _to_fp16(tensor: torch.Tensor) -> torch.Tensor:
    if torch.is_floating_point(tensor) and tensor.dtype != torch.float16:
        return tensor.to(dtype=torch.float16)
    return tensor


def _build_language_targets(
    config: Mapping[str, str | Mapping[str, tuple[str, ...]]],
) -> dict[str, list[str]]:
    targets_by_method: dict[str, list[str]] = {}
    for target, quant_method in (
        ("token_embed", config.get("token_embed")),
        ("per_layer_token_embed", config.get("per_layer_token_embed")),
    ):
        if quant_method is None:
            continue
        target_name = (
            _LANGUAGE_TOKEN_EMBED_WEIGHT
            if target == "token_embed"
            else _LANGUAGE_PER_LAYER_TOKEN_EMBED_WEIGHT
        )
        targets_by_method.setdefault(quant_method, []).append(target_name)

    linears = config.get("linears", {})
    if not isinstance(linears, Mapping):
        raise TypeError("Gemma 4 quant config 'linears' must be a mapping of quant methods to suffixes.")

    for quant_method, suffixes in linears.items():
        tensors = targets_by_method.setdefault(quant_method, [])
        for layer_idx in range(_NUM_LANGUAGE_LAYERS):
            layer_prefix = f"model.language_model.layers.{layer_idx}."
            for suffix in suffixes:
                if suffix in _KV_PROJECTION_WEIGHT_SUFFIXES and layer_idx >= _NUM_LANGUAGE_KV_PROJECTION_LAYERS:
                    continue
                tensors.append(layer_prefix + suffix)
    return targets_by_method


def _target_count(targets_by_method: Mapping[str, list[str]]) -> int:
    return sum(len(tensors) for tensors in targets_by_method.values())


def _format_target_counts(targets_by_method: Mapping[str, list[str]]) -> str:
    return ", ".join(
        f"{len(tensors)} {quant_method}" for quant_method, tensors in targets_by_method.items()
    )


def _quant_metadata(targets_by_method: Mapping[str, list[str]]) -> dict[str, str]:
    metadata = {
        "quant_target_count": str(_target_count(targets_by_method)),
    }
    for quant_method, tensors in targets_by_method.items():
        key = quant_method.replace("-", "_")
        metadata[f"{key}_targets"] = str(len(tensors))
        metadata[f"{key}_quant_targets"] = ",".join(tensors)
    return metadata


def _save_fp16_component(
    input_path: Path,
    output_path: Path,
    *,
    component: str,
    prefixes: tuple[str, ...],
) -> Path:
    print(f"Saving {component} tensors as fp16 to {output_path} ...")
    save_start = time.perf_counter()
    state_dict: dict[str, torch.Tensor] = {}

    with safe_open(str(input_path), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            if _matches_any_prefix(name, prefixes):
                state_dict[name] = _to_fp16(handle.get_tensor(name))

    if not state_dict:
        raise ValueError(f"No {component} tensors found in {input_path}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        state_dict,
        str(output_path),
        metadata={
            "component": component,
            "dtype": "fp16",
        },
    )
    save_seconds = time.perf_counter() - save_start
    print(f"Saved {len(state_dict)} {component} tensors in {save_seconds:.3f}s")
    return output_path


def _save_language_quantized(
    input_path: Path,
    output_path: Path,
    *,
    method: str,
    targets_by_method: dict[str, list[str]],
) -> Path:
    print(
        f"Applying Gemma 4 {method} quantization to "
        f"{_format_target_counts(targets_by_method)} language tensors on cuda ..."
    )
    quantize_start = time.perf_counter()
    quantized_state_dict = quantize_model_tensors(
        files=input_path,
        targets=targets_by_method,
        inclusion_prefix=_LANGUAGE_PREFIXES,
        exclusion_prefix=_AUDIO_PREFIXES + _VISION_PREFIXES,
    )
    state_dict = {
        name: tensor
        for name, tensor in quantized_state_dict.items()
        if _matches_any_prefix(name, _LANGUAGE_PREFIXES)
    }

    if not state_dict:
        raise ValueError(f"No language tensors found in {input_path}.")

    quantize_seconds = time.perf_counter() - quantize_start
    print(f"Quantized language tensors with {method} config in {quantize_seconds:.3f}s")

    print(f"Saving quantized language tensors to {output_path} ...")
    save_start = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        state_dict,
        str(output_path),
        metadata={
            "component": "language",
            "method": method,
            **_quant_metadata(targets_by_method),
        },
    )
    save_seconds = time.perf_counter() - save_start
    print(f"Saved {len(state_dict)} language tensors in {save_seconds:.3f}s")
    return output_path


def quantize_gemma4(
    root: str | Path,
    input_path: str | Path | None = None,
    *,
    method: str = _DEFAULT_QUANT_METHOD,
    media: bool = False,
) -> tuple[Path | None, Path | None, Path]:
    method = method.strip().lower()
    try:
        quant_config = _GEMMA4_QUANT_CONFIGS[method]
    except KeyError as exc:
        allowed = ", ".join(_GEMMA4_QUANT_METHODS)
        raise ValueError(
            f"Unsupported Gemma 4 quantization method: {method!r}. Expected one of: {allowed}."
        ) from exc

    gemma4_path = Path(root) / "gemma4_2b"
    checkpoint_path = (
        Path(input_path).expanduser().resolve()
        if input_path is not None
        else gemma4_path / _CHECKPOINT_NAME
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Gemma 4 checkpoint not found: {checkpoint_path}")

    targets_by_method = _build_language_targets(quant_config)

    print(f"Splitting Gemma 4 checkpoint: {checkpoint_path}")
    print(
        "Found "
        f"{_format_target_counts(targets_by_method)} "
        "language tensors to quantize."
    )

    audio_path: Path | None = None
    vision_path: Path | None = None
    if media:
        audio_path = _save_fp16_component(
            checkpoint_path,
            gemma4_path / "audio.safetensors",
            component="audio",
            prefixes=_AUDIO_PREFIXES,
        )
        vision_path = _save_fp16_component(
            checkpoint_path,
            gemma4_path / "vision.safetensors",
            component="vision",
            prefixes=_VISION_PREFIXES,
        )

    language_path = _save_language_quantized(
        checkpoint_path,
        gemma4_path / f"language_{method}_quant.safetensors",
        method=method,
        targets_by_method=targets_by_method,
    )
    return audio_path, vision_path, language_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split Gemma 4 E2B-IT weights and quantize the language component."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder containing the gemma4 model subdirectory.",
    )
    parser.add_argument(
        "--input",
        help=(
            "Source model.safetensors file. "
            "Defaults to <root>/gemma4/model.safetensors."
        ),
    )
    parser.add_argument(
        "--method",
        choices=_GEMMA4_QUANT_METHODS,
        default=_DEFAULT_QUANT_METHOD,
        help="Gemma 4 quantization config to apply.",
    )
    parser.add_argument(
        "--media",
        action="store_true",
        help="Also extract audio and vision tensors as fp16 safetensors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path, vision_path, language_path = quantize_gemma4(
        args.root,
        input_path=args.input,
        method=args.method,
        media=args.media,
    )
    if audio_path is not None:
        print(f"Saved audio weights to {audio_path}")
    if vision_path is not None:
        print(f"Saved vision weights to {vision_path}")
    print(f"Saved quantized language weights to {language_path}")


if __name__ == "__main__":
    main()
