from __future__ import annotations

from typing import Any, Dict

from flask import render_template, request

from video.grid import create_video_grids


def register_grid_generate(app, strip_outer_quotes, parse_pair, grid_output_dir) -> None:
    @app.route("/grids", methods=["GET", "POST"])
    def grids_page():
        context: Dict[str, Any] = {}
        if request.method == "POST":
            video_path = strip_outer_quotes(request.form.get("video", "")).strip()
            grid_format = request.form.get("grid_format", "2x2").strip()
            cell_ratio = request.form.get("cell_ratio", "1x1").strip()
            alignment = request.form.get("alignment", "center").strip().lower()
            selection_mode = request.form.get("selection_mode", "interval").strip().lower()
            frame_interval_raw = request.form.get("frame_interval", "1").strip()
            cell_height_raw = request.form.get("cell_height", "384").strip()

            context["last_input"] = {
                "video": video_path,
                "grid_format": grid_format,
                "cell_ratio": cell_ratio,
                "alignment": alignment,
                "selection_mode": selection_mode,
                "frame_interval": frame_interval_raw,
                "cell_height": cell_height_raw,
            }

            if not video_path:
                context["error"] = "Please provide a video path."
            else:
                try:
                    rows, cols = parse_pair(grid_format, "Grid format")
                    ratio_w, ratio_h = parse_pair(cell_ratio, "Cell ratio")
                    frame_interval = float(frame_interval_raw) if selection_mode == "interval" else None
                    cell_height = int(cell_height_raw)

                    result = create_video_grids(
                        video_path=video_path,
                        output_dir=grid_output_dir(),
                        grid_rows=rows,
                        grid_cols=cols,
                        cell_ratio=(ratio_w, ratio_h),
                        cell_height=cell_height,
                        frame_interval_sec=frame_interval,
                        alignment=alignment,
                        selection_mode=selection_mode,
                    )
                    context["result"] = result
                except Exception as exc:
                    context["error"] = f"Grid generation failed: {exc}"

        return render_template("grids.html", **context)
