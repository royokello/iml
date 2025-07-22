#!/usr/bin/env python3
"""label.py – lightweight Flask app for captioning loop folders.

Creates a small web UI that shows a GIF preview of each loop and lets the user
save comma‑separated tags to `caption.txt` inside every `loop_XXX` directory.

API summary
───────────
/                     → static/index.html
/frames/<path>        → raw file under the loops root (gif or png)
/api/loop?idx=N       → {idx, gif, tags}
/api/nav?dir=prev|next|random → same payload for new idx
/api/save POST        → {idx:int, tags:str}  → writes caption.txt
/api/stats            → {tag: count, …}
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static", static_url_path="/static")
state = {"idx": 0}
app.config["LOOPS_ROOT"] = None
loops: list[Path] = []

# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# -----------------------------------------------------------------------------

def loops_in(root: Path):
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("loop_"))

# ───────────────────────────────────────────────────────────────────────────────
# Static + asset routes
# -----------------------------------------------------------------------------

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/frames/<path:sub>")
def frames(sub: str):
    """Serve any file under the loops root (png or gif)."""
    return send_from_directory(app.config["LOOPS_ROOT"], sub)

# ───────────────────────────────────────────────────────────────────────────────
# API routes
# -----------------------------------------------------------------------------

@app.route("/api/loop")
def api_loop():
    idx = int(request.args.get("idx", state["idx"]))
    idx = max(0, min(idx, len(loops) - 1))
    state["idx"] = idx
    loop_dir = loops[idx]

    payload = {
        "idx": idx,
        "gif": f"{loop_dir.name}.gif",  # relative path served via /frames/
        "tags": "",
    }
    caption_file = loop_dir / "caption.txt"
    if caption_file.exists():
        payload["tags"] = caption_file.read_text().strip()
    return jsonify(payload)


@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.get_json(force=True)
    idx = int(data["idx"])
    tags = data["tags"].strip()
    loop_dir = loops[idx]
    (loop_dir / "caption.txt").write_text(tags)
    return "OK"


@app.route("/api/nav")
def api_nav():
    direction = request.args.get("dir", "next")
    if direction == "random":
        state["idx"] = random.randrange(len(loops))
    elif direction == "prev":
        state["idx"] = (state["idx"] - 1) % len(loops)
    else:  # next
        state["idx"] = (state["idx"] + 1) % len(loops)
    return api_loop()


@app.route("/api/stats")
def api_stats():
    counts: Counter[str] = Counter()
    for loop_dir in loops:
        txt = loop_dir / "caption.txt"
        if txt.exists():
            counts.update(t.strip() for t in txt.read_text().split(",") if t.strip())
    return jsonify(dict(counts))

# ───────────────────────────────────────────────────────────────────────────────
# Main entry
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start caption UI for loop folders")
    parser.add_argument("--input", required=True, help="Folder containing loop_XXX subdirectories")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    root = Path(args.input).resolve()
    if not root.is_dir():
        raise SystemExit(f"Input dir not found: {root}")

    app.config["LOOPS_ROOT"] = str(root)
    loops = loops_in(root)
    if not loops:
        raise SystemExit("No loop_* subfolders found.")

    print(f"[label] {len(loops)} loops loaded from {root}")
    app.run(port=args.port, debug=False)
