from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from flask import jsonify, request


def init_app(app, sessions, sessions_lock, next_sid, scan_thread, root_dir) -> None:
    @app.route("/api/scan", methods=["POST"])
    def scan_start():
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Request body must be JSON."}), 400

        path_str = data.get("path")
        if not path_str:
            return jsonify({"error": "Missing 'path' in request body."}), 400
        if path_str.startswith('"') and path_str.endswith('"'):
            path_str = path_str[1:-1]

        resolved_path = Path(path_str).resolve()
        if not resolved_path.exists():
            return jsonify({"error": f"Path does not exist: {path_str}"}), 400
        if not resolved_path.is_dir():
            return jsonify({"error": f"Path is not a directory: {path_str}"}), 400

        # Create timestamped session directory
        assert root_dir is not None, "init_app() must be called before routes are used"
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        session_dir = root_dir / "image" / "similar" / timestamp

        recursive = data.get("recursive", True)
        hash_size = data.get("hash_size", 16)
        mode = data.get("mode", "grey")
        session_id = next_sid()

        session_data = {
            "progress": {"current": 0, "total": 0, "done": False},
            "root": str(resolved_path),
            "session_dir": str(session_dir),
            "recursive": recursive,
            "items": None,
            "error": None,
        }

        with sessions_lock:
            sessions[session_id] = session_data

        # Start background scan thread
        t = threading.Thread(
            target=scan_thread,
            args=(session_id, resolved_path, session_dir, recursive, hash_size, mode),
            daemon=True,
        )
        t.start()

        return jsonify({"scanning": True, "sid": session_id})

    @app.route("/api/scan-status")
    def scan_status():
        sid = request.args.get("sid")
        if not sid:
            return jsonify({"error": "Missing 'sid' query parameter."}), 400

        with sessions_lock:
            session = sessions.get(sid)

        if session is None:
            return jsonify({"error": f"Session not found: {sid}"}), 400

        progress = session["progress"]

        if progress["done"]:
            error = session.get("error")
            if error:
                return jsonify(
                    {
                        "scanning": False,
                        "done": True,
                        "sid": sid,
                        "total": progress["total"],
                        "error": error,
                    }
                )
            return jsonify(
                {
                    "scanning": False,
                    "done": True,
                    "sid": sid,
                    "total": progress["total"],
                }
            )

        return jsonify(
            {
                "scanning": True,
                "done": False,
                "progress": progress["current"],
                "total": progress["total"],
                "phase": progress.get("phase", "hashing"),
                "sid": sid,
            }
        )
