from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from flask import jsonify

try:
    from safetensors.torch import safe_open
except ImportError:  # pragma: no cover - optional dependency
    safe_open = None

ALLOWED_EXTENSIONS = {".safetensors"}


def _summarize_safetensors(model_path: Path) -> Dict[str, Any]:
    if safe_open is None:
        raise RuntimeError("safetensors is not installed in this environment.")

    tensors: List[Dict[str, Any]] = []
    dtype_counts: Counter[str] = Counter()
    total_elements = 0
    user_metadata: Dict[str, Any] = {}

    with safe_open(model_path, framework="pt", device="cpu") as handle:
        user_metadata = handle.metadata() or {}
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            dtype = str(tensor.dtype).replace("torch.", "")
            numel = tensor.numel()
            tensors.append(
                {
                    "name": key,
                    "shape": list(tensor.shape),
                    "dtype": dtype,
                }
            )
            dtype_counts[dtype] += 1
            total_elements += numel

    metadata = {
        "tensor_count": len(tensors),
        "total_elements": total_elements,
        "dtypes": dict(dtype_counts),
        "format": "safetensors",
        "_metadata": user_metadata,
    }
    return {"metadata": metadata, "tensors": tensors}


def _load_model_summary(model_path: Path) -> Dict[str, Dict[str, List[Dict[str, object]]]]:
    if model_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Only .safetensors files are supported.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    file_size = model_path.stat().st_size
    modified_at = datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()

    summary = _summarize_safetensors(model_path)

    summary["metadata"].update(
        {
            "source": str(model_path),
            "file_size_bytes": file_size,
            "modified_at": modified_at,
        }
    )
    return summary


def register(app, resolve_path) -> None:
    @app.route("/inspect/<path:model_filepath>")
    def inspect_model(model_filepath: str):
        """
        Inspect a model by path (relative to the configured root).
        """
        try:
            model_path = resolve_path(model_filepath)
            summary = _load_model_summary(model_path)
            return jsonify(summary)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500
        except Exception as exc:  # pragma: no cover - catch-all for unexpected errors
            return jsonify({"error": f"Unexpected error: {exc}"}), 500
