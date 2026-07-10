from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from flask import jsonify, render_template, request, send_file
from PIL import Image

from . import config
from .inference import generate_images
from .jobs import JobStore
from .quant_scan import scan_denoiser, scan_text_encoder


def _entry_from_metadata(meta: dict, outputs: list[str]) -> dict[str, Any]:
    iml = meta.get("iml_generate", {})
    return {
        "id": iml.get("job_id", ""),
        "job_id": iml.get("job_id", ""),
        "timestamp": iml.get("timestamp", ""),
        "model": iml.get("model", ""),
        "mode": iml.get("mode", "text"),
        "flux_version": iml.get("flux_version"),
        "flux_base": iml.get("flux_base", False),
        "prompt": iml.get("prompt", ""),
        "settings": iml.get("settings", {}),
        "reference_images": iml.get("reference_images", []),
        "outputs": outputs,
    }


def _read_png_metadata(png_path: Path) -> dict | None:
    try:
        with Image.open(png_path) as img:
            raw = img.text.get("iml_generate")
        if not raw:
            return None
        return json.loads(raw)
    except Exception:
        return None


def _job_history_entries(root: Path, limit: int = 50) -> list[dict[str, Any]]:
    base = root / config.OUTPUT_SUBDIR
    if not base.is_dir():
        return []
    pngs = sorted(base.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    groups: dict[str, dict[str, Any]] = {}
    for png in pngs:
        if len(groups) >= limit:
            break
        meta = _read_png_metadata(png)
        if meta is None:
            continue
        job_id = meta.get("iml_generate", {}).get("job_id", "")
        if not job_id or job_id in groups:
            continue
        groups[job_id] = {"meta": meta, "pngs": [str(png.as_posix())]}
    entries = []
    # Maintain reverse-mtime order from the sorted png loop
    for job_id, g in groups.items():
        entries.append(_entry_from_metadata(g["meta"], g["pngs"]))
    return entries


def _job_history_entry(root: Path, job_id: str) -> dict[str, Any] | None:
    base = root / config.OUTPUT_SUBDIR
    if not base.is_dir():
        return None
    found = []
    meta = None
    for png in sorted(base.glob("*.png")):
        m = _read_png_metadata(png)
        if m is None:
            continue
        if m.get("iml_generate", {}).get("job_id") == job_id:
            found.append(png)
            meta = m
    if not found or meta is None:
        return None
    outputs = [str(p.as_posix()) for p in found]
    return _entry_from_metadata(meta, outputs)


def _parse_model(value: str) -> tuple[str, str | None]:
    value = value.strip().lower()
    if value.startswith("flux"):
        parts = value.split("-", 1)
        return "flux", parts[1] if len(parts) > 1 else config.DEFAULT_FLUX_VERSION
    return "ideogram", None


def init_app(app, root_dir: Path) -> None:
    jobs = JobStore()

    ref_temp_dir = root_dir / config.OUTPUT_SUBDIR / "_refs"
    ref_temp_dir.mkdir(parents=True, exist_ok=True)

    @app.route("/")
    def index():
        return render_template("generate.html")

    @app.route("/api/config")
    def api_config():
        static = Path(app.static_folder)

        def _load_json(p: Path) -> dict:
            if p.is_file():
                return json.loads(p.read_text(encoding="utf-8"))
            return {}

        shared_styles = _load_json(static / "config" / "styles.json")
        flux_schema = _load_json(static / "config" / "flux_schema.json")
        ideogram_schema = _load_json(static / "config" / "ideogram_schema.json")

        for field in flux_schema.get("fields", []):
            if field.get("key") == "style" and not field.get("suggestions"):
                field["suggestions"] = shared_styles

        for field in ideogram_schema.get("fields", []):
            if field.get("key") == "style_description.art_style" and not field.get("suggestions"):
                field["suggestions"] = shared_styles

        return jsonify({
            "flux_schema": flux_schema,
            "ideogram_schema": ideogram_schema,
        })

    @app.route("/api/quant-methods")
    def api_quant_methods():
        model = request.args.get("model", "flux-4b")
        version = request.args.get("version", None)
        variant = request.args.get("variant", config.DEFAULT_FLUX_VARIANT)

        model_type, parsed_version = _parse_model(model)
        ver = version or parsed_version

        return jsonify({
            "text_encoder": scan_text_encoder(root_dir, model_type, ver),
            "denoiser": scan_denoiser(root_dir, model_type, ver, variant),
        })

    @app.route("/api/upload-ref", methods=["POST"])
    def api_upload_ref():
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400

        ext = os.path.splitext(f.filename)[1].lower()
        allowed = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        if ext not in allowed:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400

        name = f"{uuid.uuid4().hex[:12]}_{f.filename}"
        dest = ref_temp_dir / name
        f.save(str(dest))
        return jsonify({"path": str(dest)})

    @app.route("/api/generate", methods=["POST"])
    def api_generate():
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload"}), 400

        raw_model = data.get("model", "flux-4b")
        model_type, flux_ver = _parse_model(raw_model)
        base = bool(data.get("base", False))
        raw_prompt = data.get("prompt", "")
        mode = data.get("mode", "text")

        if mode == "json" and isinstance(raw_prompt, dict):
            prompt = json.dumps(raw_prompt)
        elif isinstance(raw_prompt, dict):
            prompt = json.dumps(raw_prompt)
        else:
            prompt = str(raw_prompt)

        width = int(data.get("width", 1024))
        height = int(data.get("height", 1024))
        steps = int(data.get("steps", 50))
        guidance_scale = float(data.get("guidance_scale", 4.0 if model_type == "flux" else 7.0))
        seed_raw = data.get("seed")
        seed = int(seed_raw) if seed_raw is not None else None
        text_quant = data.get("text_quant_method")
        denoiser_quant = data.get("denoiser_quant_method")
        ref_size = int(data.get("reference_size", config.DEFAULT_REF_SIZE))
        refs = data.get("reference_images") or []
        offloading = bool(data.get("offloading", False))

        job_id = jobs.new_job_id()
        output_dir = root_dir / config.OUTPUT_SUBDIR

        def run_job():
            return generate_images(
                root_dir,
                model=model_type,
                mode=mode,
                version=flux_ver,
                base=base,
                prompt=prompt,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                text_quant_method=text_quant,
                denoiser_quant_method=denoiser_quant,
                reference_images=refs,
                reference_size=ref_size,
                output_dir=output_dir,
                job_id=job_id,
                offloading=offloading,
            )

        jobs.create_entry(job_id)
        jobs.submit(job_id, run_job)

        return jsonify({"job_id": job_id})

    @app.route("/api/generate/status/<job_id>")
    def api_generate_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            return jsonify({"status": "unknown"}), 404

        result: dict[str, Any] = {
            "id": job["id"],
            "status": job["status"],
            "error": job["error"],
        }

        if job["status"] == "done" and job["result"]:
            r = job["result"]
            paths = r.get("paths", [])
            meta = r.get("metadata", {})

            result["paths"] = paths
            result["metadata"] = meta
            result["entry_id"] = job_id

        return jsonify(result)

    @app.route("/api/generate/image/<filename>")
    def api_generate_image(filename: str):
        if "/" in filename or "\\" in filename:
            return jsonify({"error": "Invalid filename"}), 400
        img_path = root_dir / config.OUTPUT_SUBDIR / filename
        if not img_path.is_file():
            return jsonify({"error": "File not found"}), 404
        return send_file(str(img_path))

    @app.route("/api/history")
    def api_history():
        entries = _job_history_entries(root_dir, limit=50)
        return jsonify({"entries": entries})

    @app.route("/api/history/<entry_id>/regen")
    def api_history_regen(entry_id: str):
        entry = _job_history_entry(root_dir, entry_id)
        if entry is None:
            return jsonify({"error": "Entry not found"}), 404
        return jsonify(entry)
