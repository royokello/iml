import argparse
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from video.analysis import run_analysis
from video.animate import create_image_animation
from video.compress import run_batch_compression
from video.grid import create_video_grids

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


def _grid_output_dir() -> Path:
    return _ensure_root_configured() / "outputs" / "grids"


def _animation_output_dir() -> Path:
    return _ensure_root_configured() / "animations"


def _sdxl_output_dir() -> Path:
    return _ensure_root_configured() / "sdxl"


def _parse_pair(value: str, label: str) -> tuple[int, int]:
    if not value:
        raise ValueError(f"{label} is required.")

    cleaned = value.lower().replace(":", "x")
    parts = [part.strip() for part in cleaned.split("x") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"{label} must look like 2x2.")

    try:
        first = int(parts[0])
        second = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"{label} must contain integers.") from exc

    if first <= 0 or second <= 0:
        raise ValueError(f"{label} must use positive numbers.")

    return first, second


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


@app.route("/grids", methods=["GET", "POST"])
def grids_page():
    context = {}
    if request.method == "POST":
        video_path = request.form.get("video", "").strip()
        grid_format = request.form.get("grid_format", "2x2").strip()
        cell_ratio = request.form.get("cell_ratio", "1x1").strip()
        alignment = request.form.get("alignment", "center").strip().lower()
        frame_interval_raw = request.form.get("frame_interval", "1").strip()
        cell_height_raw = request.form.get("cell_height", "384").strip()

        context["last_input"] = {
            "video": video_path,
            "grid_format": grid_format,
            "cell_ratio": cell_ratio,
            "alignment": alignment,
            "frame_interval": frame_interval_raw,
            "cell_height": cell_height_raw,
        }

        if not video_path:
            context["error"] = "Please provide a video path."
        else:
            try:
                rows, cols = _parse_pair(grid_format, "Grid format")
                ratio_w, ratio_h = _parse_pair(cell_ratio, "Cell ratio")
                frame_interval = float(frame_interval_raw)
                cell_height = int(cell_height_raw)

                result = create_video_grids(
                    video_path=video_path,
                    output_dir=_grid_output_dir(),
                    grid_rows=rows,
                    grid_cols=cols,
                    cell_ratio=(ratio_w, ratio_h),
                    cell_height=cell_height,
                    frame_interval_sec=frame_interval,
                    alignment=alignment,
                )
                context["result"] = result
            except Exception as exc:
                context["error"] = f"Grid generation failed: {exc}"

    return render_template("grids.html", **context)


@app.route("/animate-grid", methods=["GET", "POST"])
def animate_grid_page():
    context: Dict[str, Any] = {}
    if request.method == "POST":
        image_path = request.form.get("image_path", "").strip()
        length_raw = request.form.get("length", "4").strip()
        context["last_input"] = {
            "image_path": image_path,
            "length": length_raw,
        }

        try:
            length_secs = float(length_raw)
            if length_secs <= 0:
                raise ValueError("Length must be greater than 0.")
        except ValueError as exc:
            context["error"] = f"Invalid length: {exc}"
            return render_template("animate.html", **context)

        if not image_path:
            context["error"] = "Please provide an absolute path to the grid image."
        else:
            try:
                video_result = create_image_animation(
                    image_path=image_path,
                    output_root=_animation_output_dir(),
                    length_seconds=length_secs,
                )
                context["result"] = {
                    "run_path": str(_animation_output_dir()),
                    "length": length_secs,
                    "video": video_result,
                    "frame_count": video_result["frame_count"],
                    "cell_count": video_result["cells"],
                }
            except Exception as exc:
                context["error"] = f"Failed to create animation: {exc}"

    return render_template("animate.html", **context)


@app.route("/sdxl", methods=["GET", "POST"])
def sdxl_page():
    fallback_defaults = {
        "prompt": "A propaganda poster depicting a cat dressed as french emperor napoleon holding a piece of cheese.",
        "negative": "",
        "seed": 19930625,
        "height": 512,
        "width": 512,
        "steps": 20,
        "cfg": 5.0,
    }

    sdxl_inference = None
    defaults = dict(fallback_defaults)
    load_error = None

    try:
        from sdxl import _inference as sdxl_inference

        defaults = {
            "prompt": sdxl_inference.DEFAULT_PROMPT,
            "negative": sdxl_inference.DEFAULT_NEGATIVE,
            "seed": sdxl_inference.DEFAULT_SEED,
            "height": sdxl_inference.DEFAULT_HEIGHT,
            "width": sdxl_inference.DEFAULT_WIDTH,
            "steps": sdxl_inference.DEFAULT_STEPS,
            "cfg": sdxl_inference.DEFAULT_CFG,
        }
    except Exception as exc:
        load_error = f"Unable to load SDXL inference dependencies: {exc}"

    model_base = _ensure_root_configured() / "sdxl" / "base"
    context = {"defaults": defaults, "model_base": str(model_base)}

    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()
        negative = request.form.get("negative", "").strip()
        seed_raw = request.form.get("seed", "").strip()
        width_raw = request.form.get("width", "").strip()
        height_raw = request.form.get("height", "").strip()
        steps_raw = request.form.get("steps", "").strip()
        cfg_raw = request.form.get("cfg", "").strip()

        context["last_input"] = {
            "prompt": prompt,
            "negative": negative,
            "seed": seed_raw,
            "width": width_raw,
            "height": height_raw,
            "steps": steps_raw,
            "cfg": cfg_raw,
        }

        if load_error:
            context["error"] = load_error
            return render_template("sdxl.html", **context)

        if not model_base.is_dir():
            context["error"] = f"SDXL base directory not found: {model_base}"
            return render_template("sdxl.html", **context)

        try:
            seed = int(seed_raw) if seed_raw else defaults["seed"]
            width = int(width_raw) if width_raw else defaults["width"]
            height = int(height_raw) if height_raw else defaults["height"]
            steps = int(steps_raw) if steps_raw else defaults["steps"]
            cfg = float(cfg_raw) if cfg_raw else defaults["cfg"]
        except ValueError as exc:
            context["error"] = f"Invalid numeric value: {exc}"
            return render_template("sdxl.html", **context)

        try:
            output_dir = _sdxl_output_dir()
            output_paths = sdxl_inference.generate_images(
                output_dir=str(output_dir),
                base_dir=str(model_base),
                prompt=prompt or defaults["prompt"],
                negative=negative,
                seed=seed,
                height=height,
                width=width,
                steps=steps,
                cfg=cfg,
            )
        except Exception as exc:
            context["error"] = f"SDXL generation failed: {exc}"
        else:
            context["result"] = {
                "output_dir": str(output_dir),
                "images": [{"filename": Path(path).name, "path": path} for path in output_paths],
            }
    elif load_error:
        context["error"] = load_error

    return render_template("sdxl.html", **context)


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
    video_extensions = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
    media_extensions = image_extensions | video_extensions
    
    for root, dirs, files in os.walk(path):
        # Sort directories for consistent ordering
        dirs.sort()
        
        # Filter and sort media files (images and videos)
        media_files = sorted([
            os.path.join(root, f) for f in files
            if Path(f).suffix.lower() in media_extensions
        ])
        
        if media_files:
            SCAN_RESULT.append({
                "folder_path": root,
                "images": media_files  # Keep 'images' key for backward compatibility
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
    
    media_path = folder["images"][i]
    
    # Detect file type and set appropriate mimetype
    ext = Path(media_path).suffix.lower()
    video_extensions = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
    
    if ext in video_extensions:
        # Set mimetype for videos
        mimetype_map = {
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".mkv": "video/x-matroska"
        }
        return send_file(media_path, mimetype=mimetype_map.get(ext, "video/mp4"))
    else:
        return send_file(media_path)


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


@app.route("/api/grid-image")
def api_grid_image():
    run_dir = request.args.get("run", "").strip()
    filename = request.args.get("file", "").strip()
    if not run_dir or not filename:
        return jsonify({"error": "Missing run or file name"}), 400

    try:
        datetime.strptime(run_dir, "%Y-%m-%d-%H-%M-%S")
    except ValueError:
        return jsonify({"error": "Invalid run name"}), 400

    if Path(filename).name != filename:
        return jsonify({"error": "Invalid file name"}), 400

    try:
        image_path = _grid_output_dir() / run_dir / filename
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    if not image_path.is_file():
        return jsonify({"error": "File not found"}), 404

    return send_file(image_path, mimetype="image/png")


@app.route("/api/grid-video")
def api_grid_video():
    run_dir = request.args.get("run", "").strip()
    filename = request.args.get("file", "").strip()
    if not run_dir or not filename:
        return jsonify({"error": "Missing run or file name"}), 400

    try:
        datetime.strptime(run_dir, "%Y-%m-%d-%H-%M-%S")
    except ValueError:
        return jsonify({"error": "Invalid run name"}), 400

    if Path(filename).name != filename or not filename.lower().endswith(".mp4"):
        return jsonify({"error": "Invalid file name"}), 400

    try:
        video_path = _grid_output_dir() / run_dir / filename
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    if not video_path.is_file():
        return jsonify({"error": "File not found"}), 404

    return send_file(video_path, mimetype="video/mp4")


@app.route("/api/animation-video")
def api_animation_video():
    filename = request.args.get("file", "").strip()
    if not filename:
        return jsonify({"error": "Missing file name"}), 400

    if Path(filename).name != filename:
        return jsonify({"error": "Invalid file name"}), 400

    try:
        video_path = _animation_output_dir() / filename
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    if not video_path.is_file():
        return jsonify({"error": "File not found"}), 404

    mimetype_map = {
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".webm": "video/webm",
        ".gif": "image/gif",
    }
    return send_file(video_path, mimetype=mimetype_map.get(video_path.suffix.lower(), "video/mp4"))


@app.route("/api/sdxl-image")
def api_sdxl_image():
    filename = request.args.get("file", "").strip()
    if not filename:
        return jsonify({"error": "Missing file name"}), 400

    if Path(filename).name != filename:
        return jsonify({"error": "Invalid file name"}), 400

    try:
        image_path = _sdxl_output_dir() / filename
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    if not image_path.is_file():
        return jsonify({"error": "File not found"}), 404

    mimetype_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    return send_file(image_path, mimetype=mimetype_map.get(image_path.suffix.lower(), "image/png"))




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
