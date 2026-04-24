from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_from_directory

from .core import build_preview_set, initial_global_count, parse_time_ranges, save_native_frames


BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    static_url_path="/static",
    static_folder=str(BASE_DIR / "static"),
    template_folder=str(BASE_DIR / "templates"),
)

ROOT_DIR: Path | None = None

STATE: dict[str, Any] = {
    "video_path": None,
    "output_dir": None,
    "preview_dir": None,
    "items": [],
    "buffer": 3,
    "filename_width": 6,
    "preview_resolution": 384,
    "interval_seconds": 1.0,
    "time_ranges_text": "",
    "video": None,
}


def strip_outer_quotes(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        return cleaned[1:-1].strip()
    return cleaned


def cleanup_preview_dir() -> None:
    STATE["preview_dir"] = None


def reset_state() -> None:
    cleanup_preview_dir()
    STATE.update(
        {
            "video_path": None,
            "output_dir": None,
            "items": [],
            "buffer": 3,
            "filename_width": 6,
            "preview_resolution": 384,
            "interval_seconds": 1.0,
            "time_ranges_text": "",
            "video": None,
        }
    )


def resolve_existing_file(raw_path: str, label: str) -> Path:
    cleaned = strip_outer_quotes(str(raw_path or ""))
    if not cleaned:
        raise ValueError(f"{label} is required.")
    path = Path(cleaned).expanduser()
    if not path.is_file():
        raise ValueError(f"{label} must point to an existing file.")
    return path.resolve()


def resolve_output_dir(raw_path: str) -> Path:
    cleaned = strip_outer_quotes(str(raw_path or ""))
    if not cleaned:
        raise ValueError("Output directory is required.")
    path = Path(cleaned).expanduser()
    resolved = path.resolve()
    if resolved.exists() and not resolved.is_dir():
        raise ValueError("Output directory points to a file.")
    return resolved


def ensure_root_configured() -> Path:
    if ROOT_DIR is None:
        raise RuntimeError("FFmpeg root directory is not configured.")
    return ROOT_DIR


def get_ffmpeg_bin() -> Path:
    root_dir = ensure_root_configured()
    ffmpeg_exe = root_dir / "ffmpeg" / "bin" / "ffmpeg.exe"
    if not ffmpeg_exe.is_file():
        raise ValueError(f"FFmpeg not found at {ffmpeg_exe}")
    return ffmpeg_exe


def parse_positive_int(raw_value: Any, label: str) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if value <= 0:
        raise ValueError(f"{label} must be greater than 0.")
    return value


def parse_non_negative_int(raw_value: Any, label: str) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if value < 0:
        raise ValueError(f"{label} must be >= 0.")
    return value


def parse_positive_float(raw_value: Any, label: str) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc
    if value <= 0:
        raise ValueError(f"{label} must be greater than 0.")
    return value


def state_payload() -> dict[str, Any]:
    preview_dir = STATE.get("preview_dir")
    preview_ready = bool(preview_dir and Path(preview_dir).is_dir())
    return {
        "video_path": STATE["video_path"],
        "output_dir": STATE["output_dir"],
        "buffer": STATE["buffer"],
        "filename_width": STATE["filename_width"],
        "preview_resolution": STATE["preview_resolution"],
        "interval_seconds": STATE["interval_seconds"],
        "time_ranges_text": STATE["time_ranges_text"],
        "video": STATE["video"],
        "items": [
            {
                "mark_index": item["mark_index"],
                "timestamp": item["timestamp"],
                "time_seconds": item["time_seconds"],
                "range_label": item.get("range_label"),
                "preview_name": item["preview_name"],
                "preview_url": f"/preview/{item['preview_name']}" if preview_ready else None,
            }
            for item in STATE["items"]
        ],
    }


def json_error(message: str, status: int = 400):
    return jsonify({"error": message}), status


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/preview/<path:filename>")
def preview_file(filename: str):
    preview_dir = STATE.get("preview_dir")
    if not preview_dir:
        return json_error("No preview set is loaded.", 404)
    return send_from_directory(preview_dir, filename)


@app.route("/api/load", methods=["POST"])
def api_load():
    payload = request.get_json(silent=True) or {}

    try:
        video_path = resolve_existing_file(payload.get("video_path", ""), "Video path")
        output_dir = resolve_output_dir(payload.get("output_dir", ""))
        preview_dir = output_dir / "preview"
        preview_resolution = parse_positive_int(payload.get("preview_resolution", 384), "Preview resolution")
        interval_seconds = parse_positive_float(payload.get("interval_seconds", 1.0), "Interval seconds")
        buffer = parse_non_negative_int(payload.get("buffer", 3), "Buffer")
        filename_width = parse_positive_int(payload.get("filename_width", 6), "Filename width")
        time_ranges_text = strip_outer_quotes(str(payload.get("time_ranges", "")))
        time_ranges = parse_time_ranges(time_ranges_text)
        ffmpeg_exe = get_ffmpeg_bin()
    except ValueError as exc:
        return json_error(str(exc))
    except RuntimeError as exc:
        return json_error(str(exc), 500)

    reset_state()

    try:
        preview_set = build_preview_set(
            video_path=video_path,
            preview_dir=preview_dir,
            interval_seconds=interval_seconds,
            resolution=preview_resolution,
            filename_width=filename_width,
            ffmpeg_exe=ffmpeg_exe,
            time_ranges=time_ranges,
        )
    except Exception as exc:
        return json_error(f"Preview extraction failed: {exc}", 500)

    STATE.update(
        {
            "video_path": str(video_path),
            "output_dir": str(output_dir),
            "preview_dir": preview_dir,
            "items": preview_set["items"],
            "buffer": buffer,
            "filename_width": filename_width,
            "preview_resolution": preview_resolution,
            "interval_seconds": interval_seconds,
            "time_ranges_text": time_ranges_text,
            "video": preview_set["video"],
        }
    )
    return jsonify(state_payload())


@app.route("/api/save", methods=["POST"])
def api_save():
    if not STATE.get("video_path") or not STATE.get("items"):
        return json_error("Load a video before saving.")

    payload = request.get_json(silent=True) or {}
    raw_frames = payload.get("selected_frames", [])
    if not isinstance(raw_frames, list):
        return json_error("selected_frames must be a list.")

    try:
        selected_frames = [int(value) for value in raw_frames]
        output_dir = resolve_output_dir(payload.get("output_dir", STATE["output_dir"] or ""))
        buffer = parse_non_negative_int(payload.get("buffer", STATE["buffer"]), "Buffer")
        filename_width = parse_positive_int(payload.get("filename_width", STATE["filename_width"]), "Filename width")
        ffmpeg_exe = get_ffmpeg_bin()
    except ValueError as exc:
        return json_error(str(exc))
    except RuntimeError as exc:
        return json_error(str(exc), 500)

    items_by_frame = {int(item["mark_index"]): item for item in STATE["items"]}
    missing_frames = [frame_index for frame_index in selected_frames if frame_index not in items_by_frame]
    if missing_frames:
        return json_error("One or more selected frame indexes are invalid.")

    selections = []
    seen_frames: set[int] = set()
    for frame_index in selected_frames:
        if frame_index in seen_frames:
            continue
        seen_frames.add(frame_index)
        selections.append(items_by_frame[frame_index])
    if not selections:
        return json_error("Select at least one preview before saving.")

    output_dir.mkdir(parents=True, exist_ok=True)
    start_count = initial_global_count(output_dir)

    try:
        results = save_native_frames(
            ffmpeg_exe=ffmpeg_exe,
            video_path=STATE["video_path"],
            output_dir=output_dir,
            selections=selections,
            buffer=buffer,
            filename_width=filename_width,
            start_count=start_count,
        )
    except Exception as exc:
        return json_error(f"Save failed: {exc}", 500)

    STATE["output_dir"] = str(output_dir)
    STATE["buffer"] = buffer
    STATE["filename_width"] = filename_width

    return jsonify(
        {
            "saved_count": len(results),
            "output_dir": str(output_dir),
            "results": results,
        }
    )


def main(root: str, host: str = "127.0.0.1", port: int = 5052) -> None:
    global ROOT_DIR
    ROOT_DIR = Path(root).resolve()
    print(f"Serving extract selector on http://{host}:{port}")
    app.run(host=host, port=port, debug=True, use_reloader=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the frame selection app for video.extract.")
    parser.add_argument("--root", required=True, help="Root directory containing ffmpeg/bin/ffmpeg.exe")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5052)
    args = parser.parse_args()
    main(args.root, args.host, args.port)
