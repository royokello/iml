from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

from flask import render_template, request


def register(
    app,
    ensure_root_configured: Callable[[], Path],
    sdxl_output_dir: Callable[[], Path],
) -> None:
    @app.route("/sdxl", methods=["GET", "POST"])
    def sdxl_page():
        fallback_defaults = {
            "prompt": "A propaganda poster depicting a cat dressed as french emperor napoleon holding a piece of cheese.",
            "negative": "",
            "seed": 19930625,
            "height": 512,
            "width": 512,
            "steps": 20,
            "cfg": 5.0,
        }

        sdxl_inference = None
        defaults = dict(fallback_defaults)
        load_error = None

        try:
            from sdxl import inference as sdxl_inference

            defaults = {
                "prompt": sdxl_inference.DEFAULT_PROMPT,
                "negative": sdxl_inference.DEFAULT_NEGATIVE,
                "seed": sdxl_inference.DEFAULT_SEED,
                "height": sdxl_inference.DEFAULT_HEIGHT,
                "width": sdxl_inference.DEFAULT_WIDTH,
                "steps": sdxl_inference.DEFAULT_STEPS,
                "cfg": sdxl_inference.DEFAULT_CFG,
            }
        except Exception as exc:
            load_error = f"Unable to load SDXL inference dependencies: {exc}"

        model_base = ensure_root_configured() / "sdxl" / "base"
        context: Dict[str, Any] = {"defaults": defaults, "model_base": str(model_base)}

        if request.method == "POST":
            prompt = request.form.get("prompt", "").strip()
            negative = request.form.get("negative", "").strip()
            seed_raw = request.form.get("seed", "").strip()
            width_raw = request.form.get("width", "").strip()
            height_raw = request.form.get("height", "").strip()
            steps_raw = request.form.get("steps", "").strip()
            cfg_raw = request.form.get("cfg", "").strip()

            context["last_input"] = {
                "prompt": prompt,
                "negative": negative,
                "seed": seed_raw,
                "width": width_raw,
                "height": height_raw,
                "steps": steps_raw,
                "cfg": cfg_raw,
            }

            if load_error:
                context["error"] = load_error
                return render_template("sdxl.html", **context)

            if not model_base.is_dir():
                context["error"] = f"SDXL base directory not found: {model_base}"
                return render_template("sdxl.html", **context)

            try:
                seed = int(seed_raw) if seed_raw else defaults["seed"]
                width = int(width_raw) if width_raw else defaults["width"]
                height = int(height_raw) if height_raw else defaults["height"]
                steps = int(steps_raw) if steps_raw else defaults["steps"]
                cfg = float(cfg_raw) if cfg_raw else defaults["cfg"]
            except ValueError as exc:
                context["error"] = f"Invalid numeric value: {exc}"
                return render_template("sdxl.html", **context)

            try:
                output_dir = sdxl_output_dir()
                output_paths = sdxl_inference.generate_images(
                    output_dir=str(output_dir),
                    base_dir=str(model_base),
                    prompt=prompt or defaults["prompt"],
                    negative=negative,
                    seed=seed,
                    height=height,
                    width=width,
                    steps=steps,
                    cfg=cfg,
                )
            except Exception as exc:
                context["error"] = f"SDXL generation failed: {exc}"
            else:
                context["result"] = {
                    "output_dir": str(output_dir),
                    "images": [{"filename": Path(path).name, "path": path} for path in output_paths],
                }
        elif load_error:
            context["error"] = load_error

        return render_template("sdxl.html", **context)
