from __future__ import annotations

from flask import render_template


def init_app(app) -> None:
    @app.route("/")
    @app.route("/similar")
    def similar_page():
        return render_template("index.html")
