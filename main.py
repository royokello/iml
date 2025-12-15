import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from video.analysis import run_analysis

from flask import Flask, jsonify, render_template, request

try:
    from safetensors.torch import safe_open
except ImportError:  # pragma: no cover - optional dependency
    safe_open = None

ALLOWED_EXTENSIONS = {".safetensors"}

app = Flask(__name__)
ROOT_DIR: Path | None = None


def _ensure_root_configured() -> Path:
    if ROOT_DIR is None:
        raise RuntimeError("Model root directory is not configured.")
    return ROOT_DIR


def _resolve_path(relative_path: str) -> Path:
    """
    Only allow absolute filesystem paths. Relative paths are rejected.
    """
    candidate = Path(relative_path).expanduser()
    if not candidate.is_absolute():
        raise ValueError("Please provide an absolute path to the safetensors file.")
    return candidate.resolve()


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


def _list_directory(relative_path: str) -> Dict[str, Any]:
    """
    Return directory entries within the root for browsing from the UI.
    """
    base = _resolve_path(relative_path)
    if not base.exists():
        raise FileNotFoundError(f"Path not found: {base}")
    if not base.is_dir():
        raise ValueError("Requested path is not a directory.")

    dirs = []
    files = []
    for child in sorted(base.iterdir(), key=lambda p: p.name.lower()):
        if child.name.startswith("."):
            continue

        rel = child.relative_to(_ensure_root_configured()).as_posix()
        if child.is_dir():
            dirs.append({"name": child.name, "path": rel})
        elif child.is_file():
            files.append(
                {
                    "name": child.name,
                    "path": rel,
                    "size": child.stat().st_size,
                    "ext": child.suffix.lower(),
                }
            )
    return {"path": relative_path, "dirs": dirs, "files": files}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/inspect")
def inspect():
    return render_template("inspect.html")


@app.route("/quality", methods=["GET", "POST"])
def quality_page():
    context = {}
    if request.method == "POST":
        source = request.form.get("source", "").strip()
        resolution = int(request.form.get("resolution", 768))
        qualities_str = request.form.get("qualities", "")
        # Parse qualities
        crfs = []
        for q in qualities_str.split(","):
            q = q.strip()
            if q.isdigit():
                crfs.append(int(q))
        
        sample_len = float(request.form.get("sample_len", 8.0))
        num_samples = int(request.form.get("num_samples", 32))
        
        context["last_input"] = {
            "source": source,
            "resolution": resolution,
            "qualities": qualities_str,
            "sample_len": sample_len,
            "num_samples": num_samples
        }
        
        if not source or not crfs:
            context["error"] = "Please provide valid source and qualities."
        else:
            try:
                results = run_analysis(
                    _ensure_root_configured(),
                    source,
                    resolution,
                    0, # start_crf unused in this signature
                    crfs,
                    sample_len,
                    num_samples
                )
                if "error" in results:
                    context["error"] = results["error"]
                else:
                    context["results"] = results
            except Exception as e:
                context["error"] = f"Analysis failed: {str(e)}"

    return render_template("quality.html", **context)


@app.route("/inspect/<path:model_filepath>")
def inspect_model(model_filepath: str):
    """
    Inspect a model by path (relative to the configured root).
    """
    try:
        model_path = _resolve_path(model_filepath)
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


@app.route("/inspect/browse")
def browse_models():
    """
    Browse the root directory to pick a model file.
    """
    rel_path = request.args.get("path", "").strip()
    try:
        listing = _list_directory(rel_path)
        return jsonify(
            {
                "path": listing["path"],
                "dirs": listing["dirs"],
                "files": listing["files"],
                "root": str(_ensure_root_configured()),
                "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
            }
        )
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Unexpected error: {exc}"}), 500


def main(root: str):
    global ROOT_DIR
    ROOT_DIR = Path(root).resolve()
    print(f"Starting ML Bench ... scanning models under: {ROOT_DIR}")
    app.run(host="0.0.0.0", port=5051, debug=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    args = parser.parse_args()
    main(args.root)
