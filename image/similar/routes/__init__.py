from __future__ import annotations

import threading
import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask
from image.dedup.grouping import IMAGE_EXTS, scan_images

from . import page, scan as scan_routes, filter as filter_routes, thumb as thumb_routes, export as export_routes

_root_dir: Path | None = None
_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()


def _next_sid() -> str:
    return uuid.uuid4().hex[:12]


def _scan_thread(session_id: str, root_path: Path, session_dir: Path, recursive: bool, hash_size: int, mode: str) -> None:
    """Background thread: count files, scan images, compute quality."""
    try:
        session_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: count image files for progress reporting
        total_files = 0
        walker = root_path.rglob("*") if recursive else root_path.glob("*")
        for path in walker:
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                total_files += 1

        with _sessions_lock:
            if session_id in _sessions:
                _sessions[session_id]["progress"]["total"] = total_files

        if total_files == 0:
            with _sessions_lock:
                if session_id in _sessions:
                    _sessions[session_id]["progress"]["done"] = True
                    _sessions[session_id]["error"] = "No images found."
            return

        # Phase 2: run scan_images to get dedup hash items
        with _sessions_lock:
            if session_id in _sessions:
                _sessions[session_id]["progress"]["phase"] = "hashing"
        def _progress(n):
            with _sessions_lock:
                if session_id in _sessions:
                    _sessions[session_id]["progress"]["current"] = n
        items, pixel_cache = scan_images(root_path, hash_size=hash_size, mode=mode, progress_cb=_progress)

        with _sessions_lock:
            if session_id in _sessions:
                _sessions[session_id]["items"] = items
                _sessions[session_id]["pixel_cache"] = pixel_cache
                _sessions[session_id]["progress"]["current"] = total_files
                _sessions[session_id]["progress"]["total"] = total_files
                _sessions[session_id]["progress"]["done"] = True

    except Exception as exc:
        with _sessions_lock:
            if session_id in _sessions:
                _sessions[session_id]["progress"]["done"] = True
                _sessions[session_id]["error"] = str(exc)


def init_app(app: Flask, root_dir: Path) -> None:
    global _root_dir
    _root_dir = root_dir
    page.init_app(app)
    scan_routes.init_app(app, _sessions, _sessions_lock, _next_sid, _scan_thread, _root_dir)
    filter_routes.init_app(app, _sessions, _sessions_lock)
    thumb_routes.init_app(app, _sessions, _sessions_lock)
    export_routes.init_app(app, _sessions, _sessions_lock)
