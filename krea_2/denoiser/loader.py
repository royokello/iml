from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch
from safetensors import safe_open

from flux2.denoiser.loader import _list_checkpoint_keys, _materialize_meta_tensors
from krea_2.denoiser.targets import _build_krea2_denoiser_target_tensors
from krea_2.models.denoiser.transformer import Krea2Transformer2DModel
from utils.loaders.sharded import load_local_sharded_checkpoint, stream_local_sharded_checkpoint
from utils.loaders.single import load_local_single_checkpoint, stream_local_single_checkpoint
from utils.quant.linear import QuantizedLinear
from utils.quant.name import convert_quant_ranks
from utils.quant.replace import replace_targeted_linear_modules, target_tensors_to_linear_names
from utils.quant.streaming import StreamedQuantizedLinear, pin_quantized_buffers

__all__ = ["load_krea2_denoiser"]

_VARIANTS = ("turbo", "raw")
_SHARD_PATTERN = "diffusion_pytorch_model-*.safetensors"
_PLAIN_METHODS = ("fp16", "fp32", "bf16")


def _build_model_from_config(model_dir: Path, *, device: str = "meta") -> Krea2Transformer2DModel:
    with (model_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    init_parameters = inspect.signature(Krea2Transformer2DModel.__init__).parameters
    init_kwargs = {key: value for key, value in config.items() if key in init_parameters and key != "self"}

    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        with torch.device(device):
            model = Krea2Transformer2DModel(**init_kwargs)
    finally:
        torch.set_default_dtype(default_dtype)
    return model


def _check_shard_keys(model: torch.nn.Module, checkpoint_dir: Path) -> None:
    checkpoint_keys: set[str] = set()
    for shard_path in sorted(checkpoint_dir.glob(_SHARD_PATTERN)):
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            checkpoint_keys.update(handle.keys())

    model_keys = set(model.state_dict().keys())
    unexpected = sorted(checkpoint_keys - model_keys)
    missing = sorted(model_keys - checkpoint_keys)
    if unexpected or missing:
        raise RuntimeError(
            f"Denoiser checkpoint in {checkpoint_dir} does not match the model. "
            f"Unexpected keys: {', '.join(unexpected) or 'none'}. "
            f"Missing keys: {', '.join(missing) or 'none'}."
        )


def _cast_float_tensors_except_quantized(module: torch.nn.Module, dtype: torch.dtype) -> None:
    if isinstance(module, QuantizedLinear):
        return

    for parameter_name, parameter in module.named_parameters(recurse=False):
        if parameter is not None and parameter.is_floating_point() and parameter.dtype != dtype:
            module._parameters[parameter_name] = torch.nn.Parameter(
                parameter.to(dtype=dtype),
                requires_grad=parameter.requires_grad,
            )

    for buffer_name, buffer in module.named_buffers(recurse=False):
        if buffer is not None and buffer.is_floating_point() and buffer.dtype != dtype:
            module._buffers[buffer_name] = buffer.to(dtype=dtype)

    for child in module.children():
        _cast_float_tensors_except_quantized(child, dtype)


def _materialize_meta_parameters(model: torch.nn.Module) -> None:
    """Replace meta parameters with empty CPU tensors.

    A streamed load assigns with `parameter.data = tensor`, and PyTorch rejects
    that when the parameter is on the meta device. The quantized layers already
    hold real CPU buffers, so only the non-quantized parameters are still meta.
    For Krea 2 that is about 0.6 GiB.
    """
    for module in model.modules():
        for name, parameter in list(module._parameters.items()):
            if parameter is not None and parameter.is_meta:
                module._parameters[name] = torch.nn.Parameter(
                    torch.empty(parameter.shape, dtype=parameter.dtype),
                    requires_grad=parameter.requires_grad,
                )


def load_krea2_denoiser(
    path: str | Path,
    quant_method: str | None = None,
    *,
    distilled: bool = True,
    offloading: bool = False,
    stream_load: bool | None = None,
    stream_denoiser: bool = False,
) -> Krea2Transformer2DModel:
    """Load the Krea 2 denoiser.

    Args:
        path: The `denoiser` folder, or a `denoiser/<variant>` folder.
        quant_method: A quantization method name, or None for plain weights.
            `"fp16"` reads `<variant>/fp16_quant.safetensors`.
        distilled: True for the turbo variant, False for raw.
        offloading: Build and keep the weights in host memory.
        stream_load: Copy tensors into the model one at a time. Defaults to
            `offloading`.
        stream_denoiser: Keep the quantized weights in pinned host memory and
            copy each layer onto the GPU for its own multiply.

    Returns:
        The denoiser in eval mode with float weights in fp16.

    Raises:
        ValueError: If `path` points to a variant that does not match
            `distilled`, or if `stream_denoiser` has no quantized checkpoint.
        FileNotFoundError: If a folder or checkpoint is missing. The loader
            never quantizes on the fly.
        RuntimeError: If the checkpoint keys do not match the model, or if a
            tensor stays on the meta device.
    """
    variant = "turbo" if distilled else "raw"
    if stream_denoiser and quant_method is None:
        raise ValueError(
            "stream_denoiser needs a quantized checkpoint. "
            "Pass a quant method, or drop --stream-denoiser."
        )
    linear_cls = StreamedQuantizedLinear if stream_denoiser else QuantizedLinear

    if quant_method is not None:
        quant_method_key = quant_method.replace("-", "_")
        rank_1_method, rank_2_method = convert_quant_ranks(quant_method, 2)
    else:
        quant_method_key = rank_1_method = rank_2_method = None

    target_tensors = _build_krea2_denoiser_target_tensors()
    rank_1_linear = target_tensors_to_linear_names(target_tensors["rank_1"])
    rank_2_linear = target_tensors_to_linear_names(target_tensors["rank_2"])

    model_dir = Path(path).expanduser().resolve()
    if model_dir.is_file():
        model_dir = model_dir.parent

    if model_dir.name in _VARIANTS and model_dir.parent.name == "denoiser":
        if model_dir.name != variant:
            raise ValueError(
                f"Denoiser variant mismatch: path points to {model_dir.name!r}, distilled={distilled!r}."
            )
        checkpoint_dir = model_dir
        model_dir = model_dir.parent
    else:
        checkpoint_dir = model_dir / variant

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Denoiser directory not found: {model_dir}")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Denoiser checkpoint directory not found: {checkpoint_dir}")

    # A streamed denoiser keeps only its quantized buffers in host memory, so
    # build on meta and let the streamed load materialize the rest. Building on
    # CPU would allocate the full fp16 model first (~26 GB for Krea 2).
    build_device = "cpu" if (offloading and not stream_denoiser) else "meta"
    model = _build_model_from_config(model_dir, device=build_device)
    if stream_load is None:
        stream_load = offloading

    quantized_checkpoint_path = None
    if quant_method is not None:
        quantized_checkpoint_path = checkpoint_dir / f"{quant_method_key}_quant.safetensors"
        if not quantized_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Quantized checkpoint not found: {quantized_checkpoint_path}. Run the denoiser quantizer first."
            )
        print(f"    using saved quantized checkpoint: {quantized_checkpoint_path.name}")

    if quantized_checkpoint_path is None:
        single_path = checkpoint_dir / "diffusion_pytorch_model.safetensors"
        if single_path.is_file():
            if stream_load:
                incompatible = stream_local_single_checkpoint(model, single_path, offloading=offloading)
            else:
                incompatible = load_local_single_checkpoint(model, single_path)
        elif any(checkpoint_dir.glob(_SHARD_PATTERN)):
            _check_shard_keys(model, checkpoint_dir)
            print(f"    loading sharded checkpoint from {checkpoint_dir}")
            if stream_load:
                stream_local_sharded_checkpoint(
                    model,
                    checkpoint_dir,
                    index_filename=None,
                    shard_pattern=_SHARD_PATTERN,
                    offloading=offloading,
                )
            else:
                load_local_sharded_checkpoint(
                    model,
                    checkpoint_dir,
                    index_filename=None,
                    shard_pattern=_SHARD_PATTERN,
                )
            incompatible = {"unexpected_keys": [], "missing_keys": []}
        else:
            raise FileNotFoundError(f"Denoiser checkpoint not found in {checkpoint_dir}")
    else:
        checkpoint_keys = _list_checkpoint_keys(quantized_checkpoint_path)
        if rank_1_method not in _PLAIN_METHODS:
            replace_targeted_linear_modules(
                model,
                method=rank_1_method,
                target_linear_names=rank_1_linear,
                quantize_weights=False,
                linear_cls=linear_cls,
                checkpoint_keys=checkpoint_keys,
            )
        if rank_2_method not in _PLAIN_METHODS:
            replace_targeted_linear_modules(
                model,
                method=rank_2_method,
                target_linear_names=rank_2_linear,
                quantize_weights=False,
                linear_cls=linear_cls,
                checkpoint_keys=checkpoint_keys,
            )

        if stream_denoiser:
            # The model was built on meta: see build_device above. The streamed
            # load assigns through `parameter.data`, which meta rejects.
            _materialize_meta_parameters(model)

        if stream_load:
            incompatible = stream_local_single_checkpoint(
                model, quantized_checkpoint_path, offloading=offloading
            )
        else:
            incompatible = load_local_single_checkpoint(model, quantized_checkpoint_path)

    if not offloading:
        _materialize_meta_tensors(model)

    unexpected = incompatible.get("unexpected_keys") or []
    missing = incompatible.get("missing_keys") or []
    if unexpected or missing:
        raise RuntimeError(
            "Denoiser checkpoint does not match the model. "
            f"Unexpected keys: {', '.join(sorted(unexpected)) or 'none'}. "
            f"Missing keys: {', '.join(sorted(missing)) or 'none'}."
        )

    _cast_float_tensors_except_quantized(model, torch.float16)

    if stream_denoiser:
        pin_quantized_buffers(model)

    model.eval()
    return model
