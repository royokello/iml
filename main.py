import argparse
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from video.analysis import run_analysis
from video.compress import run_batch_compression

from flask import Flask, jsonify, render_template, request, send_file

try:
    from safetensors.torch import safe_open
except ImportError:  # pragma: no cover - optional dependency
    safe_open = None

ALLOWED_EXTENSIONS = {".safetensors"}

app = Flask(__name__)
ROOT_DIR: Path | None = None
SCAN_RESULT: List[Dict[str, Any]] = []  # Stores [{"folder_path": str, "images": [str]}]


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


@app.route("/compress", methods=["GET", "POST"])
def compress_page():
    context = {}
    if request.method == "POST":
        source = request.form.get("source", "").strip()
        output_dir = request.form.get("output", "").strip()
        resolution = int(request.form.get("resolution", 768))
        crf = int(request.form.get("crf", 25))

        context["last_input"] = {
            "source": source,
            "output": output_dir,
            "resolution": resolution,
            "crf": crf
        }
        
        if not source or not output_dir:
            context["error"] = "Please provide valid source path and output directory."
        else:
            try:
                # Call compression logic
                results_list = run_batch_compression(
                    _ensure_root_configured(),
                    source,
                    output_dir,
                    resolution,
                    crf
                )
                
                # Check for global error in first item
                if results_list and results_list[0].get("error") and len(results_list) == 1:
                     context["error"] = results_list[0]["error"]
                elif not results_list:
                     context["error"] = "No results returned."
                else:
                    # Process results for display
                    total_source_size = 0
                    total_output_size = 0
                    processed_results = []
                    
                    for r in results_list:
                        if r.get("success"):
                            s_size = r.get("source_size", 0)
                            o_size = r.get("output_size", 0)
                            total_source_size += s_size
                            total_output_size += o_size
                            
                            r["source_size_mb"] = f"{(s_size / 1048576):.2f}"
                            r["output_size_mb"] = f"{(o_size / 1048576):.2f}"
                            r["filename"] = r.get("source", "").split("\\")[-1] # Basic name extraction
                        else:
                            r["filename"] = r.get("source", "Unknown").split("\\")[-1]
                        processed_results.append(r)

                    context["results"] = processed_results
                    context["summary"] = {
                        "total_files": len(processed_results),
                        "total_source_mb": f"{(total_source_size / 1048576):.2f}",
                        "total_output_mb": f"{(total_output_size / 1048576):.2f}",
                        "saved_mb": f"{((total_source_size - total_output_size) / 1048576):.2f}"
                    }
                    
            except Exception as e:
                context["error"] = f"Compression failed: {str(e)}"

    return render_template("compress.html", **context)


@app.route("/viewer")
def viewer():
    return render_template("viewer.html")


@app.route("/api/scan", methods=["POST"])
def api_scan():
    global SCAN_RESULT
    data = request.get_json()
    path = data.get("path", "").strip()
    
    if not path:
        return jsonify({"error": "Path is required"}), 400
    
    path_obj = Path(path)
    if not path_obj.exists() or not path_obj.is_dir():
        return jsonify({"error": "Invalid directory path"}), 400
    
    # Scan directory recursively
    SCAN_RESULT = []
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    
    for root, dirs, files in os.walk(path):
        # Sort directories for consistent ordering
        dirs.sort()
        
        # Filter and sort image files
        images = sorted([
            os.path.join(root, f) for f in files
            if Path(f).suffix.lower() in image_extensions
        ])
        
        if images:
            SCAN_RESULT.append({
                "folder_path": root,
                "images": images
            })
    
    # Calculate totals
    total_images = sum(len(folder["images"]) for folder in SCAN_RESULT)
    
    # Return metadata only
    folders_meta = [
        {"path": folder["folder_path"], "count": len(folder["images"])}
        for folder in SCAN_RESULT
    ]
    
    return jsonify({
        "folders": folders_meta,
        "total_images": total_images,
        "total_folders": len(SCAN_RESULT)
    })


@app.route("/api/view")
def api_view():
    f = request.args.get("f", type=int)
    i = request.args.get("i", type=int)
    
    if f is None or i is None:
        return jsonify({"error": "Missing folder or image index"}), 400
    
    if f < 0 or f >= len(SCAN_RESULT):
        return jsonify({"error": "Invalid folder index"}), 400
    
    folder = SCAN_RESULT[f]
    if i < 0 or i >= len(folder["images"]):
        return jsonify({"error": "Invalid image index"}), 400
    
    image_path = folder["images"][i]
    return send_file(image_path)


@app.route("/api/meta")
def api_meta():
    f = request.args.get("f", type=int)
    i = request.args.get("i", type=int)
    
    if f is None or i is None:
        return jsonify({"error": "Missing folder or image index"}), 400
    
    if f < 0 or f >= len(SCAN_RESULT):
        return jsonify({"error": "Invalid folder index"}), 400
    
    folder = SCAN_RESULT[f]
    if i < 0 or i >= len(folder["images"]):
        return jsonify({"error": "Invalid image index"}), 400
    
    image_path = folder["images"][i]
    
    # Calculate global index
    global_index = sum(len(SCAN_RESULT[j]["images"]) for j in range(f)) + i + 1
    total_images = sum(len(folder["images"]) for folder in SCAN_RESULT)
    
    return jsonify({
        "filename": os.path.basename(image_path),
        "folder": folder["folder_path"],
        "global_index": global_index,
        "total_images": total_images,
        "folder_index": f,
        "image_index": i,
        "folder_count": len(folder["images"])
    })




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
