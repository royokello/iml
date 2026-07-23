from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image, PngImagePlugin

from . import config


def _resize_image(path: Path, max_side: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w <= max_side and h <= max_side:
        return img
    if w >= h:
        new_w = max_side
        new_h = round(h * max_side / w)
    else:
        new_h = max_side
        new_w = round(w * max_side / h)
    return img.resize((new_w, new_h), Image.LANCZOS)


def _build_metadata(
    job_id: str,
    model: str,
    mode: str,
    version: str | None,
    base: bool,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int | None,
    text_quant_method: str,
    denoiser_quant_method: str,
    reference_images: list[str] | None,
    reference_size: int,
    loras: list[dict] | None = None,
    offloading: bool = False,
    negative_prompt: str | None = None,
    anima_variant: str | None = None,
) -> str:
    data = {
        "iml_generate": {
            "version": "1.0",
            "job_id": job_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "model": model,
            "mode": mode,
            "flux_version": version if model != "anima" else None,
            "flux_base": base if model != "anima" else None,
            "prompt": prompt,
            "settings": {
                "width": width,
                "height": height,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "text_quant_method": text_quant_method,
                "denoiser_quant_method": denoiser_quant_method,
                "reference_size": reference_size,
                "offloading": offloading,
                "negative_prompt": negative_prompt,
                "anima_variant": anima_variant if model == "anima" else None,
            },
            "reference_images": reference_images or [],
            "loras": loras or [],
        }
    }
    if model == "anima":
        data["iml_generate"]["anima_variant"] = anima_variant
    return json.dumps(data)


def _save_with_metadata(
    images: list[Image.Image],
    output_dir: Path,
    pnginfo: PngImagePlugin.PngInfo,
) -> list[str]:
    saved: list[str] = []
    ts = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    for i, img in enumerate(images):
        suffix = f"_{i}" if i else ""
        fname = f"{ts}{suffix}.png"
        out_path = output_dir / fname
        img.save(out_path, pnginfo=pnginfo)
        saved.append(str(out_path.as_posix()))
    return saved


def generate_images(
    root: str | Path,
    *,
    model: str,
    mode: str,
    version: str | None = None,
    base: bool = False,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int | None,
    text_quant_method: str,
    denoiser_quant_method: str,
    reference_images: list[str] | None,
    reference_size: int,
    loras: list[dict] | None = None,
    output_dir: Path,
    job_id: str,
    offloading: bool = False,
    negative_prompt: str | None = None,
) -> dict:
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    anima_variant = version if model == "anima" else None

    metadata_str = _build_metadata(
        job_id, model, mode, version, base, prompt, width, height, steps,
        guidance_scale, seed, text_quant_method,
        denoiser_quant_method, reference_images, reference_size,
        loras=loras,
        offloading=offloading,
        negative_prompt=negative_prompt,
        anima_variant=anima_variant,
    )
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("iml_generate", metadata_str)

    if "flux" in model:
        from flux2.gen import generate_image as flux_generate

        flux_ver = version or config.DEFAULT_FLUX_VERSION
        loras_dict: dict[str, float] | None = None
        if loras:
            loras_dict = {l["path"]: l.get("weight", 1.0) for l in loras}

        images = flux_generate(
            root,
            version=flux_ver,
            base=base,
            image=",".join(reference_images) if reference_images else None,
            prompt=prompt if isinstance(prompt, str) else json.dumps(prompt),
            width=width,
            height=height,
            num_inference_steps=steps,
            seed=seed,
            guidance_scale=guidance_scale,
            ref_size=reference_size,
            text_quant_method=text_quant_method,
            denoiser_quant_method=denoiser_quant_method,
            loras=loras_dict,
            offloading=offloading,
        )

        if not images:
            raise RuntimeError("Flux generation produced no output")

        saved_paths = _save_with_metadata(images, output_dir, pnginfo)

    elif model == "ideogram":
        from ideogram.gen import generate_image as ideogram_generate

        images = ideogram_generate(
            root,
            prompt=prompt if isinstance(prompt, str) else json.dumps(prompt),
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            text_quant_method=text_quant_method,
            denoiser_quant_method=denoiser_quant_method,
            offloading=offloading,
        )

        if not images:
            raise RuntimeError("Ideogram generation produced no output")

        saved_paths = _save_with_metadata(images, output_dir, pnginfo)

    elif model == "anima":
        from anima.gen import generate_image as anima_generate

        images = anima_generate(
            root,
            prompt=prompt,
            negative_prompt=negative_prompt,
            variant=anima_variant or "base",
            steps=steps,
            cfg=guidance_scale,
            seed=seed,
            width=width,
            height=height,
            text_quant_method=text_quant_method or None,
            denoiser_quant_method=denoiser_quant_method or None,
        )

        if not images:
            raise RuntimeError("Anima generation produced no output")

        saved_paths = _save_with_metadata(images, output_dir, pnginfo)

    else:
        raise ValueError(f"Unknown model: {model}")

    return {
        "paths": saved_paths,
        "metadata": json.loads(metadata_str),
    }



