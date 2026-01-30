from __future__ import annotations

from typing import Any, Dict

from flask import render_template, request

from video.animate import create_image_animation


def register(app, strip_outer_quotes, animation_output_dir) -> None:
    @app.route("/animate-grid", methods=["GET", "POST"])
    def animate_grid_page():
        context: Dict[str, Any] = {}
        if request.method == "POST":
            image_path = strip_outer_quotes(request.form.get("image_path", "")).strip()
            length_raw = request.form.get("length", "4").strip()
            context["last_input"] = {
                "image_path": image_path,
                "length": length_raw,
            }

            try:
                length_secs = float(length_raw)
                if length_secs <= 0:
                    raise ValueError("Length must be greater than 0.")
            except ValueError as exc:
                context["error"] = f"Invalid length: {exc}"
                return render_template("animate.html", **context)

            if not image_path:
                context["error"] = "Please provide an absolute path to the grid image."
            else:
                try:
                    output_dir = animation_output_dir()
                    video_result = create_image_animation(
                        image_path=image_path,
                        output_root=output_dir,
                        length_seconds=length_secs,
                    )
                    context["result"] = {
                        "run_path": str(output_dir),
                        "length": length_secs,
                        "video": video_result,
                        "frame_count": video_result["frame_count"],
                        "cell_count": video_result["cells"],
                    }
                except Exception as exc:
                    context["error"] = f"Failed to create animation: {exc}"

        return render_template("animate.html", **context)
