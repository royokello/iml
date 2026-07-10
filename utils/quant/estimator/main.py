from __future__ import annotations

import argparse
from collections.abc import Callable
from collections.abc import Mapping
from pathlib import Path

from flux2.denoiser.targets import (
    _build_flux2_denoiser_target_tensors as _build_flux2_denoiser_targets,
)
from flux2.text_encoder.target import _build_target_tensors as _build_flux2_text_encoder_targets
from ideogram.text_encoder.target import _build_ideogram_text_encoder_targets
from ideogram.denoiser.targets import _build_ideogram_denoiser_target_tensors
from krea_2.text_encoder.target import _build_krea2_text_encoder_targets
from krea_2.denoiser.targets import _build_krea2_denoiser_target_tensors
from anima.denoiser.targets import _build_anima_denoiser_target_tensors
from gemma4.config import (
    _GEMMA4_QUANT_CONFIGS,
    _HIGH_LINEAR_WEIGHT_SUFFIXES,
    _KV_PROJECTION_WEIGHT_SUFFIXES,
    _LANGUAGE_PER_LAYER_TOKEN_EMBED_WEIGHT,
    _LANGUAGE_TOKEN_EMBED_WEIGHT,
    _LOW_LINEAR_WEIGHT_SUFFIXES,
    _NUM_LANGUAGE_KV_PROJECTION_LAYERS,
    _NUM_LANGUAGE_LAYERS,
)
from utils.quant.estimator.utils import estimate_quantized_safetensors_size
from utils.quant.name import mixed_quant_methods, quant_method_sort_key
from utils.quant.targets import build_mixed_target_config
from wan22.quant.targets import (
    _build_wan22_denoiser_mixed_target_tensors,
    _build_wan22_text_encoder_mixed_target_tensors,
)

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
_WAN22_TEXT_ENCODER_CHECKPOINT_NAME = "t5_umt5-xxl-enc-bf16.pth"
_WAN22_DENOISER_SHARDS = 3

def _single_file(filename: str) -> Callable[[Path], list[Path]]:
    return lambda source: [source / filename]


def _build_flux2_text_encoder_files(source: Path, version: str) -> list[Path]:
    shard_count = 2 if version == "4b" else 4
    return [
        source / f"model-{index:05d}-of-{shard_count:05d}.safetensors"
        for index in range(1, shard_count + 1)
    ]


def _build_flux2_denoiser_files(source: Path, version: str) -> list[Path]:
    if version == "4b":
        return [source / "diffusion_pytorch_model.safetensors"]
    return [
        source / "diffusion_pytorch_model-00001-of-00002.safetensors",
        source / "diffusion_pytorch_model-00002-of-00002.safetensors",
    ]


def _build_ideogram_text_encoder_files(source: Path) -> list[Path]:
    return [source / "model.safetensors"]


def _build_ideogram_denoiser_files(source: Path) -> list[Path]:
    return [source / "diffusion_pytorch_model.safetensors"]


def _build_krea2_text_encoder_files(source: Path) -> list[Path]:
    return [source / "model.safetensors"]


def _build_krea2_denoiser_files(source: Path) -> list[Path]:
    return sorted(source.glob("*.safetensors"))


def _build_anima_denoiser_files(source: Path) -> list[Path]:
    return [source / "anima-base-v1.0.safetensors"]


def _build_wan22_denoiser_files(source: Path) -> list[Path]:
    return [
        source / f"diffusion_pytorch_model-{index:05d}-of-{_WAN22_DENOISER_SHARDS:05d}.safetensors"
        for index in range(1, _WAN22_DENOISER_SHARDS + 1)
    ]


_GEMMA4_LINEAR_SUFFIXES_BY_GROUP = {
    "high": _HIGH_LINEAR_WEIGHT_SUFFIXES,
    "low": _LOW_LINEAR_WEIGHT_SUFFIXES,
}


def _build_language_targets(
    config: Mapping[str, str | Mapping[str, str]],
) -> dict[str, list[str]]:
    methods: set[str] = set()
    for val in (config.get("token_embed"), config.get("per_layer_token_embed")):
        if val is not None:
            methods.add(val)
    linears = config.get("linears", {})
    if not isinstance(linears, Mapping):
        raise TypeError("Gemma 4 quant config 'linears' must be a mapping of target groups to quant methods.")
    methods.update(linears.values())
    targets_by_method: dict[str, list[str]] = {m: [] for m in methods}

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
        targets_by_method[quant_method].append(target_name)

    for target_group, quant_method in linears.items():
        try:
            suffixes = _GEMMA4_LINEAR_SUFFIXES_BY_GROUP[target_group]
        except KeyError as exc:
            raise ValueError(f"Unsupported Gemma 4 linear target group: {target_group!r}.") from exc
        tensors = targets_by_method[quant_method]
        for layer_idx in range(_NUM_LANGUAGE_LAYERS):
            layer_prefix = f"model.language_model.layers.{layer_idx}."
            for suffix in suffixes:
                if suffix in _KV_PROJECTION_WEIGHT_SUFFIXES and layer_idx >= _NUM_LANGUAGE_KV_PROJECTION_LAYERS:
                    continue
                tensors.append(layer_prefix + suffix)
    return targets_by_method


def _sorted_target_configs(
    configs: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, list[str]]]:
    return {method: configs[method] for method in sorted(configs, key=quant_method_sort_key)}


def _mixed_target_configs(
    builder: Callable[[], dict[str, list[str]]],
) -> dict[str, dict[str, list[str]]]:
    target_tensors = builder()
    configs: dict[str, dict[str, list[str]]] = {}
    for method in mixed_quant_methods():
        configs[method] = build_mixed_target_config(method, target_tensors)
    configs["fp16"] = build_mixed_target_config("fp16", target_tensors)
    return _sorted_target_configs(configs)


def _gemma4_target_configs() -> dict[str, dict[str, list[str]]]:
    return _sorted_target_configs(
        {
            method: _build_language_targets(config)
            for method, config in _GEMMA4_QUANT_CONFIGS.items()
        }
    )


def _flux2_text_target_configs() -> dict[str, dict[str, list[str]]]:
    return _mixed_target_configs(_build_flux2_text_encoder_targets)


def _flux2_4b_denoiser_target_configs() -> dict[str, dict[str, list[str]]]:
    return _mixed_target_configs(lambda: _build_flux2_denoiser_targets("4b"))


def _flux2_9b_denoiser_target_configs() -> dict[str, dict[str, list[str]]]:
    return _mixed_target_configs(lambda: _build_flux2_denoiser_targets("9b"))


def _wan22_text_encoder_target_configs() -> dict[str, dict[str, list[str]]]:
    return _mixed_target_configs(_build_wan22_text_encoder_mixed_target_tensors)


def _wan22_denoiser_target_configs() -> dict[str, dict[str, list[str]]]:
    return _mixed_target_configs(_build_wan22_denoiser_mixed_target_tensors)


def _ideogram_text_target_configs() -> dict[str, dict[str, list[str]]]:
    return _mixed_target_configs(_build_ideogram_text_encoder_targets)


def _ideogram_denoiser_target_configs() -> dict[str, dict[str, list[str]]]:
    return _mixed_target_configs(_build_ideogram_denoiser_target_tensors)


def _krea2_text_target_configs() -> dict[str, dict[str, list[str]]]:
    return _mixed_target_configs(_build_krea2_text_encoder_targets)


def _krea2_denoiser_target_configs() -> dict[str, dict[str, list[str]]]:
    return _mixed_target_configs(_build_krea2_denoiser_target_tensors)


def _model_specs() -> dict[str, dict[str, object]]:
    flux2_4b_text = {
        "files": lambda source: _build_flux2_text_encoder_files(source, "4b"),
        "target_configs": _flux2_text_target_configs,
    }
    flux2_9b_text = {
        "files": lambda source: _build_flux2_text_encoder_files(source, "9b"),
        "target_configs": _flux2_text_target_configs,
    }
    wan22_text = {
        "files": _single_file(_WAN22_TEXT_ENCODER_CHECKPOINT_NAME),
        "target_configs": _wan22_text_encoder_target_configs,
    }

    return {
        "ideogram_text_encoder": {
            "files": _build_ideogram_text_encoder_files,
            "target_configs": _ideogram_text_target_configs,
        },
        "ideogram_denoiser": {
            "files": _build_ideogram_denoiser_files,
            "target_configs": _ideogram_denoiser_target_configs,
        },
        "krea2_text_encoder": {
            "files": _build_krea2_text_encoder_files,
            "target_configs": _krea2_text_target_configs,
        },
        "krea2_denoiser": {
            "files": _build_krea2_denoiser_files,
            "target_configs": _krea2_denoiser_target_configs,
        },
        "anima_denoiser": {
            "files": _build_anima_denoiser_files,
            "target_configs": lambda: _mixed_target_configs(_build_anima_denoiser_target_tensors),
        },
        "flux2_4b_text_encoder": flux2_4b_text,
        "flux2_9b_text_encoder": flux2_9b_text,
        "flux2_4b_denoiser": {
            "files": lambda source: _build_flux2_denoiser_files(source, "4b"),
            "target_configs": _flux2_4b_denoiser_target_configs,
        },
        "flux2_9b_denoiser": {
            "files": lambda source: _build_flux2_denoiser_files(source, "9b"),
            "target_configs": _flux2_9b_denoiser_target_configs,
        },
        "gemma4_2b": {
            "files": _single_file("model.safetensors"),
            "target_configs": _gemma4_target_configs,
            "inclusion_prefix": _LANGUAGE_PREFIXES,
            "exclusion_prefix": _AUDIO_PREFIXES + _VISION_PREFIXES,
        },
        "wan22_5b_text_encoder": wan22_text,
        "wan22_5b_denoiser": {
            "files": _build_wan22_denoiser_files,
            "target_configs": _wan22_denoiser_target_configs,
        },
    }


def _methods_by_name(targets_by_method: dict[str, list[str]]) -> dict[str, str]:
    methods: dict[str, str] = {}
    for method, names in targets_by_method.items():
        for name in names:
            methods[name] = method
    return methods


def _read_tensor_metadata(paths: list[Path]) -> list[tuple[str, str, tuple[int, ...]]]:
    tensors: list[tuple[str, str, tuple[int, ...]]] = []
    for path in paths:
        if path.suffix == ".safetensors":
            from utils.quant.estimator.safetensors import get_safetensors_tensor_metadata

            tensors.extend(get_safetensors_tensor_metadata(path))
        elif path.suffix == ".pth":
            from utils.quant.estimator.torch import get_torch_tensor_metadata

            tensors.extend(get_torch_tensor_metadata(path))
        else:
            raise ValueError(f"Unsupported checkpoint file extension: {path}")
    return tensors


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{size} B"
        value /= 1024
    return f"{size} B"


def parse_args() -> argparse.Namespace:
    specs = _model_specs()
    parser = argparse.ArgumentParser(description="Estimate quantized model safetensors size.")
    parser.add_argument(
        "--model",
        choices=sorted(specs),
        required=True,
        help="Model/component preset to estimate.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Folder containing the source checkpoint file or shard files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = _model_specs()
    spec = specs[args.model]
    files = spec["files"]
    target_configs = spec["target_configs"]
    if not callable(files) or not callable(target_configs):
        raise TypeError(f"Invalid estimator spec for {args.model}")

    source = Path(args.source).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")

    paths = files(source)
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Source directory is missing expected checkpoint file(s): {', '.join(missing)}. "
            f"Directory checked: {source}"
        )

    tensors = _read_tensor_metadata(paths)
    estimates: list[tuple[str, dict[str, list[str]], int]] = []
    for method, targets_by_method in target_configs().items():
        size = estimate_quantized_safetensors_size(
            tensors,
            _methods_by_name(targets_by_method),
            spec.get("inclusion_prefix"),
            spec.get("exclusion_prefix"),
        )
        estimates.append((method, targets_by_method, size))

    print(f"model: {args.model}")
    print(f"source: {source}")
    print(f"files: {len(paths)}")
    print("estimates:")
    for method, targets_by_method, size in estimates:
        target_summary = ", ".join(
            f"{len(names)} {target_method}"
            for target_method, names in targets_by_method.items()
        )
        print(
            f"  {method}: targets={target_summary}; "
            f"estimated_size_bytes={size}; estimated_size={_format_bytes(size)}"
        )


if __name__ == "__main__":
    main()
