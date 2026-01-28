from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

from flask import render_template, request

try:
    from safetensors.torch import safe_open, save_file
except ImportError:  # pragma: no cover - optional dependency
    safe_open = None
    save_file = None

ALLOWED_EXTENSIONS = {".safetensors"}


def _bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def _cast_safetensors_to_fp16(model_path: Path, output_dir: Path) -> Dict[str, Any]:
    if safe_open is None or save_file is None:
        raise RuntimeError("safetensors is not installed in this environment.")

    if model_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Only .safetensors files are supported.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required to cast tensors to FP16.") from exc

    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"{model_path.stem}_fp16{model_path.suffix}"
    output_path = output_dir / output_name
    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")

    casted = 0
    tensors: Dict[str, "torch.Tensor"] = {}
    with safe_open(model_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if tensor.is_floating_point() and tensor.dtype != torch.float16:
                tensor = tensor.to(dtype=torch.float16)
                casted += 1
            tensors[key] = tensor

    save_file(tensors, str(output_path), metadata=metadata)

    input_size = model_path.stat().st_size
    output_size = output_path.stat().st_size

    return {
        "output_path": str(output_path),
        "tensor_count": len(tensors),
        "cast_count": casted,
        "input_size_mb": f"{_bytes_to_mb(input_size):.2f}",
        "output_size_mb": f"{_bytes_to_mb(output_size):.2f}",
    }


def register(
    app,
    resolve_path: Callable[[str], Path],
    resolve_dir: Callable[[str], Path],
) -> None:
    @app.route("/fp16", methods=["GET", "POST"])
    def fp16_page():
        context: Dict[str, Any] = {}
        if request.method == "POST":
            source = request.form.get("source", "").strip()
            output_dir = request.form.get("output", "").strip()

            context["last_input"] = {
                "source": source,
                "output": output_dir,
            }

            if not source or not output_dir:
                context["error"] = "Please provide a model path and output directory."
            else:
                try:
                    model_path = resolve_path(source)
                    output_path = resolve_dir(output_dir)
                    context["result"] = _cast_safetensors_to_fp16(model_path, output_path)
                except Exception as exc:
                    context["error"] = f"FP16 conversion failed: {exc}"

        return render_template("fp16.html", **context)
