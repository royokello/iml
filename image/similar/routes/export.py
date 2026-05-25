from __future__ import annotations

import shutil
from pathlib import Path
from flask import jsonify, request


def init_app(app, sessions, sessions_lock) -> None:
    @app.route("/api/export", methods=["POST"])
    def export_images():
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Request body must be JSON."}), 400

        sid = data.get("sid")
        paths = data.get("paths", [])
        output_dir = data.get("output_dir")

        if not sid:
            return jsonify({"error": "Missing 'sid'."}), 400
        if not paths or not isinstance(paths, list):
            return jsonify({"error": "Missing or invalid 'paths' (must be a non-empty list)."}), 400
        if not output_dir:
            return jsonify({"error": "Missing 'output_dir'."}), 400

        with sessions_lock:
            session = sessions.get(sid)
        if session is None:
            return jsonify({"error": f"Session not found: {sid}"}), 400

        session_root = Path(session["root"]).resolve()
        output_path = Path(output_dir).resolve()

        # Create output directory
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return jsonify({"error": f"Cannot create output directory: {exc}"}), 400

        copied = 0
        captions = 0
        errors = []
        used_filenames: dict[str, int] = {}

        for path_str in paths:
            try:
                src = Path(path_str).resolve()
                # Security: must be within scanned root
                src.relative_to(session_root)
                if not src.is_file():
                    errors.append(f"File not found: {path_str}")
                    continue

                # Handle duplicate filenames
                stem = src.name
                if stem in used_filenames:
                    used_filenames[stem] += 1
                    base = src.stem
                    ext = src.suffix
                    dest_name = f"{base}_{used_filenames[stem]}{ext}"
                else:
                    used_filenames[stem] = 0
                    dest_name = stem

                dst = output_path / dest_name
                shutil.copy2(str(src), str(dst))
                copied += 1

                # Companion caption
                src_caption = src.with_suffix(".txt")
                if src_caption.is_file():
                    caption_name = Path(dest_name).with_suffix(".txt").name
                    dst_caption = output_path / caption_name
                    shutil.copy2(str(src_caption), str(dst_caption))
                    captions += 1

            except ValueError:
                errors.append(f"Path outside scanned root: {path_str}")
            except Exception as exc:
                errors.append(f"Failed to copy {path_str}: {exc}")

        return jsonify({
            "copied": copied,
            "captions": captions,
            "output": str(output_path),
            "errors": errors,
        })
