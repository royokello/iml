from __future__ import annotations

import argparse
import io
import mimetypes
import os
import queue
import threading
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from flask import Flask, jsonify, render_template, request, send_file

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)

SCAN_RESULT: List[Dict[str, Any]] = []  # Stores [{"folder_path": str, "images": [str]}]
_IMG_CACHE: "OrderedDict[Tuple[str, str], bytes]" = OrderedDict()  # (zip, entry) -> decoded bytes
_ZIP_INDEX: Dict[str, int] = {}  # zip path -> folder index in SCAN_RESULT
_LOADING: set = set()  # (zip, entry) pairs currently being decoded (dedup guard)
_PREFETCH_QUEUE: "queue.Queue[Tuple[str, str]]" = queue.Queue()
_WORKER_STARTED = False
_WORKER_LOCK = threading.Lock()

CACHE_NEXT_LIMIT = 2  # albums ahead whose first images stay cached
CACHE_BACK_LIMIT = 1  # albums behind whose first images stay cached
PREFETCH_NEXT = 8  # images to preload ahead of the current position
PREFETCH_BACK = 4  # images to preload behind the current position
IMG_CACHE_CAP = 1024  # hard cap on cached images (LRU eviction)


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


def _read_zip_entry(zip_path: str, entry: str) -> bytes | None:
    """Decode a single entry from a zip. Returns None if unavailable."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            return zf.read(entry)
    except (zipfile.BadZipFile, KeyError, OSError):
        return None


def _cache_get(zip_path: str, entry: str) -> bytes | None:
    key = (zip_path, entry)
    if key in _IMG_CACHE:
        _IMG_CACHE.move_to_end(key)
        return _IMG_CACHE[key]
    return None


def _cache_put(zip_path: str, entry: str, data: bytes) -> None:
    _IMG_CACHE[(zip_path, entry)] = data
    _IMG_CACHE.move_to_end((zip_path, entry))
    while len(_IMG_CACHE) > IMG_CACHE_CAP:
        _IMG_CACHE.popitem(last=False)


def _decode(zip_path: str, entry: str) -> None:
    """Decode one entry into the cache (worker path, dedup-guarded)."""
    key = (zip_path, entry)
    if key in _IMG_CACHE or key in _LOADING:
        return
    with _WORKER_LOCK:
        if key in _IMG_CACHE or key in _LOADING:
            return
        _LOADING.add(key)
    try:
        data = _read_zip_entry(zip_path, entry)
        if data is not None:
            _cache_put(zip_path, entry, data)
    finally:
        _LOADING.discard(key)


def _neighbor_zips(f: int) -> tuple[list[str], list[str]]:
    """Zip albums in the window ahead and behind folder index f (zip-count budget)."""
    ahead: List[str] = []
    behind: List[str] = []
    n = len(SCAN_RESULT)
    for idx in range(f + 1, n):
        if len(ahead) >= CACHE_NEXT_LIMIT:
            break
        z = SCAN_RESULT[idx]["folder_path"]
        if z in _ZIP_INDEX:
            ahead.append(z)
    for idx in range(f - 1, -1, -1):
        if len(behind) >= CACHE_BACK_LIMIT:
            break
        z = SCAN_RESULT[idx]["folder_path"]
        if z in _ZIP_INDEX:
            behind.append(z)
    return ahead, behind


def _evict_outside_window(f: int) -> None:
    keep = {SCAN_RESULT[f]["folder_path"]}
    for group in _neighbor_zips(f):
        keep.update(group)
    for key in list(_IMG_CACHE):
        if key[0] not in keep:
            del _IMG_CACHE[key]


def _enqueue(zip_path: str, entry: str) -> None:
    if (zip_path, entry) in _IMG_CACHE or (zip_path, entry) in _LOADING:
        return
    if _PREFETCH_QUEUE.qsize() < IMG_CACHE_CAP:
        _PREFETCH_QUEUE.put((zip_path, entry))


def _enqueue_window(f: int, i: int) -> None:
    images = SCAN_RESULT[f]["images"]
    n = len(images)
    indices = list(range(i + 1, min(n, i + PREFETCH_NEXT + 1))) + \
        list(range(max(0, i - PREFETCH_BACK), i))
    for idx in indices:
        split = _split_zip_ref(images[idx])
        if split is not None:
            _enqueue(*split)
    ahead, behind = _neighbor_zips(f)
    for z in ahead + behind:
        for ref in SCAN_RESULT[_ZIP_INDEX[z]]["images"][:PREFETCH_NEXT]:
            split = _split_zip_ref(ref)
            if split is not None:
                _enqueue(*split)


def _start_worker() -> None:
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return
        _WORKER_STARTED = True

        def loop() -> None:
            while True:
                zip_path, entry = _PREFETCH_QUEUE.get()
                _decode(zip_path, entry)

        threading.Thread(target=loop, daemon=True).start()


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
    _IMG_CACHE.clear()
    _ZIP_INDEX.clear()
    _LOADING.clear()
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
        _ZIP_INDEX[zip_path] = len(SCAN_RESULT) - 1

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
        _start_worker()
        data = _cache_get(zip_path, entry)
        if data is None:
            data = _read_zip_entry(zip_path, entry)
            if data is not None:
                _cache_put(zip_path, entry, data)
        if data is None:
            return jsonify({"error": "Image unavailable from zip"}), 404
        _evict_outside_window(f)
        _enqueue_window(f, i)
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
    global CACHE_NEXT_LIMIT, CACHE_BACK_LIMIT, PREFETCH_NEXT, PREFETCH_BACK
    parser = argparse.ArgumentParser(description="Media Gallery — Web viewer for images and videos")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5053, help="Port to listen on (default: 5053)")
    parser.add_argument("--debug", action="store_true", default=True, help="Enable Flask debug mode")
    parser.add_argument("--cache-next", type=int, default=2,
                        help="Albums ahead whose first images stay cached (default: 2)")
    parser.add_argument("--cache-back", type=int, default=1,
                        help="Albums behind whose first images stay cached (default: 1)")
    parser.add_argument("--prefetch-next", type=int, default=8,
                        help="Images to preload ahead of the current image (default: 8)")
    parser.add_argument("--prefetch-back", type=int, default=4,
                        help="Images to preload behind the current image (default: 4)")
    args = parser.parse_args()
    CACHE_NEXT_LIMIT = max(0, args.cache_next)
    CACHE_BACK_LIMIT = max(0, args.cache_back)
    PREFETCH_NEXT = max(0, args.prefetch_next)
    PREFETCH_BACK = max(0, args.prefetch_back)

    print(f"Serving Media Gallery on http://{args.host}:{args.port}")
    for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
        methods = ",".join(sorted(rule.methods - {"HEAD", "OPTIONS"}))
        print(f"  {methods:6s} {rule.rule}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
