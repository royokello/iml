from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import quote as url_quote

from flask import Flask, jsonify, render_template, request, send_from_directory

from .grouping import ImageItem, build_groups


def create_app(image_root: Path, groups: list[list[ImageItem]]) -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates", static_url_path="/static")
    app.config["IMAGE_ROOT"] = str(image_root)
    app.config["GROUPS"] = groups

    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.route("/img/<path:subpath>")
    def img(subpath: str):
        return send_from_directory(app.config["IMAGE_ROOT"], subpath)

    @app.route("/api/group")
    def api_group():
        group_list: list[list[ImageItem]] = app.config["GROUPS"]
        if not group_list:
            return jsonify({"idx": 0, "total": 0, "count": 0, "images": []})
        raw = request.args.get("idx", "0")
        try:
            idx = int(raw)
        except ValueError:
            idx = 0
        idx = max(0, min(idx, len(group_list) - 1))
        group = group_list[idx]
        payload = {
            "idx": idx,
            "total": len(group_list),
            "count": len(group),
            "images": [
                {
                    "url": f"/img/{url_quote(item[1])}",
                    "label": item[1],
                }
                for item in group
            ],
        }
        return jsonify(payload)

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dedup and browse near-duplicate images.")
    p.add_argument("-i", "--input", required=True, type=Path, help="Directory to scan for images.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)

    p.add_argument("--dhash-size", type=int, default=8)
    p.add_argument("--dhash-threshold", type=int, default=8)

    p.add_argument("--ssim-threshold", type=float, default=0.95)
    p.add_argument("--ssim-width", type=int, default=512)
    p.add_argument("--ssim-window", type=int, default=11)
    p.add_argument("--ssim-gaussian", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--min-group-size", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.input.expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: {root} is not a directory.")

    groups = build_groups(
        root,
        dhash_size=args.dhash_size,
        dhash_threshold=args.dhash_threshold,
        ssim_threshold=args.ssim_threshold,
        ssim_width=args.ssim_width,
        ssim_window=args.ssim_window,
        ssim_gaussian=args.ssim_gaussian,
        min_group_size=args.min_group_size,
    )

    app = create_app(root, groups)
    print(f"[dedup] serving {len(groups)} groups from {root}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
