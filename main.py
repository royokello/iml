import argparse
import os
from pathlib import Path
from typing import Any, Dict, List


from flask import Flask, jsonify, render_template, request, send_file
from routes.fp16 import register as register_fp16

from routes.inspect import register as register_inspect
from routes.sdxl_generate import register as register_sdxl_generate

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
    candidate = Path(_strip_outer_quotes(relative_path)).expanduser()
    if not candidate.is_absolute():
        raise ValueError("Please provide an absolute path to the safetensors file.")
    return candidate.resolve()


def _resolve_dir(dir_path: str) -> Path:
    candidate = Path(_strip_outer_quotes(dir_path)).expanduser()
    if not candidate.is_absolute():
        raise ValueError("Please provide an absolute output directory.")
    return candidate.resolve()


def _strip_outer_quotes(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        return cleaned[1:-1].strip()
    return cleaned


def _sdxl_output_dir() -> Path:
    return _ensure_root_configured() / "sdxl" / "outputs"


register_fp16(app, _resolve_path, _resolve_dir)
register_inspect(app, _resolve_path)
register_sdxl_generate(app, _ensure_root_configured, _sdxl_output_dir)
# register_similar is called inside main() after ROOT_DIR is set


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/inspect")
def inspect():
    return render_template("inspect.html")


@app.route("/viewer")
def viewer():
    return render_template("viewer.html")


@app.route("/api/scan", methods=["POST"])
def api_scan():
    global SCAN_RESULT
    data = request.get_json()
    path = _strip_outer_quotes(data.get("path", "")).strip()
    
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
