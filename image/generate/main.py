from __future__ import annotations

import argparse
import os
from pathlib import Path

from flask import Flask

from . import routes

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="IML Image Generator — Web GUI")
    parser.add_argument("--root", type=str, required=True, help="Root directory for models and outputs")
    parser.add_argument("--port", type=int, default=5053, help="Port to listen on (default: 5053)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--debug", action="store_true", default=True, help="Enable Flask debug mode")
    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    routes.init_app(app, root_dir)

    print(f"Starting IML Image Generator ... root: {root_dir}")
    for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
        methods = ",".join(sorted(rule.methods - {"HEAD", "OPTIONS"}))
        print(f"  {methods:6s} {rule.rule}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
