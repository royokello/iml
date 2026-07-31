from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request, send_file

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)

SCAN_RESULT: List[Dict[str, Any]] = []  # Stores [{"folder_path": str, "images": [str]}]


def _strip_outer_quotes(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        return cleaned[1:-1].strip()
    return cleaned


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

VIDEO_MIMETYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scan", methods=["POST"])
def api_scan():
    global SCAN_RESULT
    data = request.get_json(silent=True) or {}
    path = _strip_outer_quotes(data.get("path", "")).strip()

    if not path:
        return jsonify({"error": "Path is required"}), 400

    path_obj = Path(path)
    if not path_obj.exists() or not path_obj.is_dir():
        return jsonify({"error": "Invalid directory path"}), 400

    SCAN_RESULT = []
    for root, dirs, files in os.walk(path_obj):
        dirs.sort()
        media_files = sorted(
            os.path.join(root, f)
            for f in files
            if Path(f).suffix.lower() in MEDIA_EXTENSIONS
        )
        if media_files:
            SCAN_RESULT.append({"folder_path": root, "images": media_files})

    total_images = sum(len(folder["images"]) for folder in SCAN_RESULT)
    folders_meta = [
        {"path": folder["folder_path"], "count": len(folder["images"])}
        for folder in SCAN_RESULT
    ]

    return jsonify({
        "folders": folders_meta,
        "total_images": total_images,
        "total_folders": len(SCAN_RESULT),
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
    ext = Path(media_path).suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return send_file(media_path, mimetype=VIDEO_MIMETYPES.get(ext, "video/mp4"))
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
    global_index = sum(len(SCAN_RESULT[j]["images"]) for j in range(f)) + i + 1
    total_images = sum(len(folder["images"]) for folder in SCAN_RESULT)

    return jsonify({
        "filename": os.path.basename(image_path),
        "folder": folder["folder_path"],
        "global_index": global_index,
        "total_images": total_images,
        "folder_index": f,
        "image_index": i,
        "folder_count": len(folder["images"]),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Media Gallery — Web viewer for images and videos")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5053, help="Port to listen on (default: 5053)")
    parser.add_argument("--debug", action="store_true", default=True, help="Enable Flask debug mode")
    args = parser.parse_args()

    print(f"Serving Media Gallery on http://{args.host}:{args.port}")
    for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
        methods = ",".join(sorted(rule.methods - {"HEAD", "OPTIONS"}))
        print(f"  {methods:6s} {rule.rule}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
