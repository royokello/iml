from __future__ import annotations

import argparse
import io
import mimetypes
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request, send_file

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)

SCAN_RESULT: List[Dict[str, Any]] = []  # Stores [{"folder_path": str, "images": [str]}]
_ZIP_CACHE: Dict[str, Dict[str, bytes]] = {}  # zip path -> {entry: bytes}
_ZIP_INDEX: Dict[str, int] = {}  # zip path -> folder index in SCAN_RESULT
CACHE_NEXT_LIMIT = 2  # whole zips held in memory ahead of the current album
CACHE_BACK_LIMIT = 1  # whole zips held in memory behind the current album


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

ZIP_SUFFIX = ".zip"


def _split_zip_ref(ref: str) -> tuple[str, str] | None:
    """Split a 'zip!entry' reference into (zip_path, entry). Returns None if not a zip ref."""
    if "!" not in ref:
        return None
    zip_path, entry = ref.rsplit("!", 1)
    if not zip_path.lower().endswith(ZIP_SUFFIX):
        return None
    if Path(entry).suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    return zip_path, entry


def _load_zip(zip_path: str) -> None:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            images = {
                info.filename: zf.read(info.filename)
                for info in zf.infolist()
                if not info.is_dir() and Path(info.filename).suffix.lower() in IMAGE_EXTENSIONS
            }
    except (zipfile.BadZipFile, OSError):
        return
    if images:
        _ZIP_CACHE[zip_path] = images


def _ensure_window(f: int) -> None:
    """Load the current zip plus the N zips ahead and behind it; evict anything outside the window."""
    folder = SCAN_RESULT[f]
    ref = folder["images"][0]
    split = _split_zip_ref(ref)
    if split is None:
        return
    current_zip, _ = split
    if current_zip not in _ZIP_CACHE:
        _load_zip(current_zip)

    load_ahead = CACHE_NEXT_LIMIT
    load_behind = CACHE_BACK_LIMIT
    n = len(SCAN_RESULT)
    for i in range(f + 1, n):
        if load_ahead == 0:
            break
        zip_path = SCAN_RESULT[i]["folder_path"]
        if zip_path in _ZIP_CACHE or _ZIP_INDEX.get(zip_path) is None:
            continue
        _load_zip(zip_path)
        load_ahead -= 1
    for i in range(f - 1, -1, -1):
        if load_behind == 0:
            break
        zip_path = SCAN_RESULT[i]["folder_path"]
        if zip_path in _ZIP_CACHE or _ZIP_INDEX.get(zip_path) is None:
            continue
        _load_zip(zip_path)
        load_behind -= 1

    cached = [p for p in _ZIP_CACHE if p in _ZIP_INDEX]
    behind = sorted((p for p in cached if _ZIP_INDEX[p] < f), key=lambda p: _ZIP_INDEX[p], reverse=True)
    ahead = sorted((p for p in cached if _ZIP_INDEX[p] > f), key=lambda p: _ZIP_INDEX[p])
    for p in behind[CACHE_BACK_LIMIT:]:
        del _ZIP_CACHE[p]
    for p in ahead[CACHE_NEXT_LIMIT:]:
        del _ZIP_CACHE[p]


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
    _ZIP_CACHE.clear()
    _ZIP_INDEX.clear()
    zip_files = []
    for root, dirs, files in os.walk(path_obj):
        dirs.sort()
        media_files = sorted(
            os.path.join(root, f)
            for f in files
            if Path(f).suffix.lower() in MEDIA_EXTENSIONS
        )
        if media_files:
            SCAN_RESULT.append({"folder_path": root, "images": media_files})
        zip_files.extend(os.path.join(root, f) for f in files if f.lower().endswith(ZIP_SUFFIX))

    for zip_path in sorted(zip_files):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                entries = sorted(
                    info.filename
                    for info in zf.infolist()
                    if not info.is_dir() and Path(info.filename).suffix.lower() in IMAGE_EXTENSIONS
                )
        except (zipfile.BadZipFile, OSError):
            continue
        if not entries:
            continue
        SCAN_RESULT.append({
            "folder_path": zip_path,
            "images": [f"{zip_path}!{entry}" for entry in entries],
        })

    _ZIP_INDEX.update(
        {folder["folder_path"]: i for i, folder in enumerate(SCAN_RESULT)}
    )

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
    split = _split_zip_ref(media_path)
    if split is not None:
        zip_path, entry = split
        _ensure_window(f)
        data = _ZIP_CACHE.get(zip_path, {}).get(entry)
        if data is None:
            return jsonify({"error": "Image unavailable from zip"}), 404
        return send_file(
            io.BytesIO(data),
            mimetype=mimetypes.guess_type(entry)[0] or "application/octet-stream",
        )
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

    split = _split_zip_ref(image_path)
    is_zip = split is not None
    filename = os.path.basename(split[1]) if split else os.path.basename(image_path)
    display_folder = split[0] if split else folder["folder_path"]

    return jsonify({
        "filename": filename,
        "folder": display_folder,
        "global_index": global_index,
        "total_images": total_images,
        "folder_index": f,
        "image_index": i,
        "folder_count": len(folder["images"]),
        "is_zip": is_zip,
    })


def main() -> None:
    global CACHE_NEXT_LIMIT, CACHE_BACK_LIMIT
    parser = argparse.ArgumentParser(description="Media Gallery — Web viewer for images and videos")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5053, help="Port to listen on (default: 5053)")
    parser.add_argument("--debug", action="store_true", default=True, help="Enable Flask debug mode")
    parser.add_argument("--cache-next", type=int, default=2,
                        help="Whole zips held in memory ahead of the current album (default: 2)")
    parser.add_argument("--cache-back", type=int, default=1,
                        help="Whole zips held in memory behind the current album (default: 1)")
    args = parser.parse_args()
    CACHE_NEXT_LIMIT = max(0, args.cache_next)
    CACHE_BACK_LIMIT = max(0, args.cache_back)

    print(f"Serving Media Gallery on http://{args.host}:{args.port}")
    for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
        methods = ",".join(sorted(rule.methods - {"HEAD", "OPTIONS"}))
        print(f"  {methods:6s} {rule.rule}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
