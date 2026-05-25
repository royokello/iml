from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from flask import jsonify, request, send_file
from PIL import Image, ImageFilter


def init_app(app, sessions, sessions_lock) -> None:
    @app.route("/api/thumb")
    def similar_thumb():
        sid = request.args.get("sid")
        path_str = request.args.get("path")
        if not sid or not path_str:
            return jsonify({"error": "Missing 'sid' or 'path' query parameter."}), 400

        with sessions_lock:
            session = sessions.get(sid)
        if session is None:
            return jsonify({"error": f"Session not found: {sid}"}), 400

        source_path = Path(path_str).resolve()
        session_root = Path(session["root"]).resolve()
        # Verify path is within the scanned root
        try:
            source_path.relative_to(session_root)
        except ValueError:
            return jsonify({"error": "Path is outside the scanned directory."}), 400

        if not source_path.is_file():
            return jsonify({"error": "File not found."}), 404

        session_dir = Path(session["session_dir"])
        thumb_dir = session_dir / "thumbs"
        thumb_dir.mkdir(parents=True, exist_ok=True)

        cache_key = sha256(str(source_path).encode()).hexdigest()[:32]
        thumb_path = thumb_dir / f"{cache_key}.webp"

        # Serve from cache if fresh
        if thumb_path.is_file():
            source_mtime = source_path.stat().st_mtime
            thumb_mtime = thumb_path.stat().st_mtime
            if thumb_mtime >= source_mtime:
                return send_file(str(thumb_path), mimetype="image/webp")

        # Generate composite thumbnail: color | greyscale + edge detect
        try:
            with Image.open(source_path) as img:
                w, h = img.size
                # Crop to square: landscape = center, portrait = top
                side = min(w, h)
                left = (w - side) // 2
                top = 0 if h > w else (h - side) // 2
                square = img.crop((left, top, left + side, top + side)).resize(
                    (256, 256), Image.Resampling.LANCZOS
                )

                # Greyscale (top right, square)
                grey = square.convert("L").resize((128, 128), Image.Resampling.LANCZOS)

                # Edge detect (bottom right, square)
                edges = square.filter(ImageFilter.CONTOUR).resize(
                    (128, 128), Image.Resampling.LANCZOS
                )

                # Composite canvas 384×256
                canvas = Image.new("RGB", (384, 256))
                canvas.paste(square, (0, 0))
                canvas.paste(grey.convert("RGB"), (256, 0))
                canvas.paste(edges.convert("RGB"), (256, 128))
                canvas.save(str(thumb_path), "WEBP", quality=80)
        except Exception as exc:
            return jsonify({"error": f"Thumbnail generation failed: {exc}"}), 500

        return send_file(str(thumb_path), mimetype="image/webp")
