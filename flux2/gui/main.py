#!/usr/bin/env python
"""
Flux2 GUI — Flask backend with API endpoints.
"""

from __future__ import annotations

import argparse
import sys
import uuid
import threading
from io import StringIO
from pathlib import Path

from flask import Flask, jsonify, request, send_file, render_template

app = Flask(__name__, template_folder=".")

# ── In-memory job store ────────────────────────────────────────────────
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _store(job_id: str, **fields: object) -> None:
    with _jobs_lock:
        _jobs[job_id] = {**_jobs.get(job_id, {}), **fields}


# ── Background generator worker ────────────────────────────────────────
def _generate_worker(job_id: str, params: dict) -> None:
    from flux2.gen import generate_image
    import torch

    _store(job_id, status="running", log="")

    log_buf = StringIO()
    old_stdout = sys.stdout

    class _Tee:
        def write(self, text: str) -> None:
            log_buf.write(text)
            old_stdout.write(text)

        def flush(self) -> None:
            log_buf.flush()
            old_stdout.flush()

    try:
        sys.stdout = _Tee()  # type: ignore[assignment]

        root = app.config["ROOT"]

        # Resolve LoRA paths  ────────────────────────────────────────
        loras: dict[str, float] | None = None
        raw_loras = params.get("loras", [])
        if raw_loras:
            loras = {}
            lora_dir = root / f"flux2/{params['version']}" / "loras"
            for entry in raw_loras:
                loras[str(lora_dir / entry["name"])] = float(
                    entry.get("weight", 1.0)
                )

        # Reference images  ─────────────────────────────────────────
        ref_images = params.get("ref_images", [])

        print(f"  root={root}")
        print(f"  version={params['version']}")
        print(f"  prompt={params.get('prompt')!r}")
        print(f"  negative_prompt={params.get('negative_prompt')!r}")
        print(f"  width={params['width']}  height={params['height']}  seed={params['seed']}")
        print(f"  base={params.get('base', False)}")
        print(f"  loras={loras}")
        print(f"  ref_images={ref_images}")
        print(f"  ref_res={params.get('ref_res', 512)}")
        print(f"  text_quant={params.get('text_quant')}")
        print(f"  denoiser_quant={params.get('denoiser_quant')}")
        generate_image(
            root=str(root),
            version=params["version"],
            image=",".join(ref_images) if ref_images else None,
            prompt=params["prompt"],
            negative_prompt=params.get("negative_prompt"),
            width=params["width"],
            height=params["height"],
            seed=params["seed"],
            base=params.get("base", False),
            ref_size=params.get("ref_res", 512),
            text_quant_method=params.get("text_quant", "sym-high")
            or None,
            denoiser_quant_method=params.get("denoiser_quant", "sym-med")
            or None,
            loras=loras,
        )

        # Find the latest output image  ────────────────────────────
        output_dir = root / f"flux2/{params['version']}" / "outputs"
        images = sorted(
            output_dir.glob("*.png"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        output_rel = (
            str(images[0].relative_to(root)) if images else None
        )

        _store(job_id, status="done", output=output_rel, log=log_buf.getvalue())

    except Exception as exc:
        _store(job_id, status="error", error=str(exc), log=log_buf.getvalue())
    finally:
        sys.stdout = old_stdout
        torch.cuda.empty_cache()


# ── Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("page.html")


@app.route("/api/models")
def list_models():
    """Return available versions (4b / 9b) after checking root."""
    root: Path = app.config["ROOT"]
    versions = []
    for v in ("4b", "9b"):
        model_dir = root / f"flux2/{v}" / "model"
        if model_dir.is_dir():
            versions.append(v)
    return jsonify({"versions": versions})


@app.route("/api/loras")
def list_loras():
    root: Path = app.config["ROOT"]
    version = request.args.get("version", "4b")
    lora_dir = root / f"flux2/{version}" / "loras"
    if not lora_dir.is_dir():
        return jsonify({"loras": []})
    files = sorted(f.name for f in lora_dir.glob("*.safetensors"))
    return jsonify({"loras": files})


@app.route("/api/text-encoder-methods")
def list_text_encoder_methods():
    root: Path = app.config["ROOT"]
    version = request.args.get("version", "4b")
    te_dir = root / f"flux2/{version}" / "model" / "text_encoder"
    methods = []
    if te_dir.is_dir():
        for fpath in te_dir.glob("*_quant.safetensors"):
            name = fpath.stem  # e.g. "sym_high_quant"
            if name.endswith("_quant"):
                name = name[:-6]
            methods.append(name.replace("_", "-"))
    return jsonify({"methods": sorted(methods)})


@app.route("/api/denoiser-methods")
def list_denoiser_methods():
    root: Path = app.config["ROOT"]
    version = request.args.get("version", "4b")
    variant = request.args.get("variant", "base")
    methods = []
    variant_dir = root / f"flux2/{version}" / "model" / "transformer" / variant
    if variant_dir.is_dir():
        for fpath in variant_dir.glob("*_quant.safetensors"):
            name = fpath.stem
            if name.endswith("_quant"):
                name = name[:-6]
            methods.append(name.replace("_", "-"))
    return jsonify({"methods": sorted(methods)})


@app.route("/api/history")
def get_history():
    root: Path = app.config["ROOT"]
    version = request.args.get("version", "4b")
    output_dir = root / f"flux2/{version}" / "outputs"
    if not output_dir.is_dir():
        return jsonify({"images": []})
    images = sorted(
        output_dir.glob("*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:4]
    return jsonify({
        "images": [
            {"path": str(img.relative_to(root)), "name": img.name}
            for img in images
        ]
    })


@app.route("/api/image")
def serve_image():
    root: Path = app.config["ROOT"]
    path = request.args.get("path", "")
    full_path = root / path
    if not full_path.is_file():
        return jsonify({"error": "not found"}), 404
    return send_file(str(full_path))


@app.route("/api/generate", methods=["POST"])
def start_generation():
    data = request.get_json(force=True)
    data["root"] = str(app.config["ROOT"])
    job_id = str(uuid.uuid4())[:8]

    _store(job_id, status="queued")

    t = threading.Thread(
        target=_generate_worker,
        args=(job_id, data),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"status": "unknown"}), 404
    return jsonify(job)


# ── Entry point ────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flux2 GUI")
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing flux2/4b/ and/or flux2/9b/",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app.config["ROOT"] = Path(args.root).expanduser().resolve()
    app.run(host="0.0.0.0", port=5000, debug=True)
