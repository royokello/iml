from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from flux2.denoiser.quant import (
    _build_checkpoint_files as _build_flux2_denoiser_files,
)
from flux2.denoiser.quant import (
    _build_target_tensors as _build_flux2_denoiser_targets,
)
from flux2.text_encoder.quant import (
    _build_checkpoint_files as _build_flux2_text_encoder_files,
)
from flux2.text_encoder.quant import (
    _build_target_tensors as _build_flux2_text_encoder_targets,
)
from gemma4.quant import (
    _AUDIO_PREFIXES,
    _GEMMA4_QUANT_CONFIGS,
    _LANGUAGE_PREFIXES,
    _VISION_PREFIXES,
    _build_language_targets,
)
from utils.quant.estimator.safetensors import get_safetensors_tensor_metadata
from utils.quant.estimator.torch import get_torch_tensor_metadata
from utils.quant.estimator.utils import estimate_quantized_safetensors_size
from wan22.quant.denoiser import (
    _build_checkpoint_files as _build_wan22_denoiser_files,
)
from wan22.quant.denoiser import (
    _build_target_tensors as _build_wan22_denoiser_targets,
)
from wan22.quant.text_encoder import (
    _CHECKPOINT_NAME as WAN22_TEXT_ENCODER_CHECKPOINT_NAME,
)
from wan22.quant.text_encoder import (
    _build_target_tensors as _build_wan22_text_encoder_targets,
)

_GEMMA4_METHOD = "high"
_DEFAULT_METHOD = "sym-high"


def _single_file(filename: str) -> Callable[[Path], list[Path]]:
    return lambda source: [source / filename]


def _standard_targets(
    builder: Callable[[], list[str]],
    method: str | None,
) -> dict[str, list[str]]:
    return {method or _DEFAULT_METHOD: builder()}


def _gemma4_targets(method: str | None) -> dict[str, list[str]]:
    method = (method or _GEMMA4_METHOD).strip().lower()
    if method != _GEMMA4_METHOD:
        raise ValueError("Gemma 4 estimator only supports --method high.")
    return _build_language_targets(_GEMMA4_QUANT_CONFIGS[method])


def _flux2_text_targets(method: str | None) -> dict[str, list[str]]:
    return _standard_targets(_build_flux2_text_encoder_targets, method)


def _flux2_4b_denoiser_targets(method: str | None) -> dict[str, list[str]]:
    return _standard_targets(lambda: _build_flux2_denoiser_targets("4b"), method)


def _flux2_9b_denoiser_targets(method: str | None) -> dict[str, list[str]]:
    return _standard_targets(lambda: _build_flux2_denoiser_targets("9b"), method)


def _wan22_text_encoder_targets(method: str | None) -> dict[str, list[str]]:
    return _standard_targets(_build_wan22_text_encoder_targets, method)


def _wan22_denoiser_targets(method: str | None) -> dict[str, list[str]]:
    return _standard_targets(_build_wan22_denoiser_targets, method)


def _model_specs() -> dict[str, dict[str, object]]:
    flux2_4b_text = {
        "files": lambda source: _build_flux2_text_encoder_files(source, "4b"),
        "targets": _flux2_text_targets,
    }
    flux2_9b_text = {
        "files": lambda source: _build_flux2_text_encoder_files(source, "9b"),
        "targets": _flux2_text_targets,
    }
    wan22_text = {
        "files": _single_file(WAN22_TEXT_ENCODER_CHECKPOINT_NAME),
        "targets": _wan22_text_encoder_targets,
    }

    return {
        "flux2_4b_text_encoder": flux2_4b_text,
        "flux2_9b_text_encoder": flux2_9b_text,
        "flux2_4b_denoiser": {
            "files": lambda source: _build_flux2_denoiser_files(source, "4b"),
            "targets": _flux2_4b_denoiser_targets,
        },
        "flux2_9b_denoiser": {
            "files": lambda source: _build_flux2_denoiser_files(source, "9b"),
            "targets": _flux2_9b_denoiser_targets,
        },
        "gemma4_2b": {
            "files": _single_file("model.safetensors"),
            "targets": _gemma4_targets,
            "inclusion_prefix": _LANGUAGE_PREFIXES,
            "exclusion_prefix": _AUDIO_PREFIXES + _VISION_PREFIXES,
        },
        "wan22_5b_text_encoder": wan22_text,
        "wan22_5b_denoiser": {
            "files": _build_wan22_denoiser_files,
            "targets": _wan22_denoiser_targets,
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
            tensors.extend(get_safetensors_tensor_metadata(path))
        elif path.suffix == ".pth":
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
    parser.add_argument(
        "--method",
        help=f"Quantization method. Defaults to {_DEFAULT_METHOD}; Gemma 4 defaults to high.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = _model_specs()
    spec = specs[args.model]
    files = spec["files"]
    targets = spec["targets"]
    if not callable(files) or not callable(targets):
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

    targets_by_method = targets(args.method)
    size = estimate_quantized_safetensors_size(
        _read_tensor_metadata(paths),
        _methods_by_name(targets_by_method),
        spec.get("inclusion_prefix"),
        spec.get("exclusion_prefix"),
    )

    print(f"model: {args.model}")
    print(f"source: {source}")
    print(f"files: {len(paths)}")
    print(
        "targets: "
        + ", ".join(f"{len(names)} {method}" for method, names in targets_by_method.items())
    )
    print(f"estimated_size_bytes: {size}")
    print(f"estimated_size: {_format_bytes(size)}")


if __name__ == "__main__":
    main()
