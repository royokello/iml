from __future__ import annotations

import argparse
import base64
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

IMAGE_EXTS: set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

DEFAULT_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"

WOMAN_PROMPT = """Create a concise natural-language training caption for a woman character LoRA from the provided image.

Rules:
- Start with the trigger phrase: `[trigger] woman`
- Describe only visible, identity-relevant details: face shape, hair, eyes, brows, lips, skin tone, expression, pose, clothing, accessories, framing, and setting.
- Keep style descriptions minimal.
- Do not write "same woman," "the character," or "reference image."
- Do not use comma-separated tag spam or DeepBooru style.
- Do not infer ethnicity, celebrity identity, age, personality, or backstory.
- Do not include subjective beauty judgments.
- Keep the caption to one sentence, ideally 25-45 words.
- Output only the caption, no explanation.

Caption format:
`[trigger] woman with [hair], [face/eyes/lips], [expression/pose], wearing [clothing/accessories], [framing/setting].`
"""

STYLE_PROMPT = """Create a concise natural-language training caption for a style LoRA from the provided image.

Rules:
- Start with the trigger phrase: `[trigger] style`
- Describe the visual style, not the subject's identity.
- Focus on visible style-relevant features: medium, rendering approach, linework or brushwork, shading, texture, color palette, contrast, lighting treatment, composition, layout, typography treatment if present, and any clearly visible print/film/process characteristics.
- Mention the subject matter only briefly and only if it helps define the style.
- Keep style descriptions specific but concise. Do not over-describe the literal content of the image.
- Do not write "same image," "the style of this image," "reference image," or anything that depends on previous images.
- Do not use comma-separated tag spam or DeepBooru style.
- Do not infer artist identity, brand, movement, era, or technique unless it is clearly and visibly dominant.
- Keep the caption to one sentence, ideally 25-45 words.
- Output only the caption, no explanation.

Caption format:
`[trigger] style, [medium/rendering], [linework/brushwork/shading/texture], [color palette/lighting/contrast], [composition/layout treatment], [brief subject matter if useful].`
"""

MAN_PROPMT = """Create a concise natural-language training caption for a man character LoRA from the provided image.

Rules:
- Start with the trigger phrase: `[trigger] man`
- Describe only visible, identity-relevant details: face shape, jawline, chin, facial hair, hairline, hairstyle, eyes, brows, nose, lips, skin tone, expression, pose, clothing, accessories, framing, and setting.
- Keep style descriptions minimal. Do not over-emphasize film grain, lighting style, camera stock, era, color grading, or image quality unless they are essential and visibly dominant.
- Do not write "same man," "the character," "reference image," or anything that depends on previous images.
- Do not use comma-separated tag spam or DeepBooru style.
- Do not infer ethnicity, celebrity identity, age, personality, or backstory.
- Do not include subjective attractiveness judgments.
- Keep the caption to one sentence, ideally 25-45 words.
- Output only the caption, no explanation.

Caption format:
`[trigger] man with [hair/facial hair], [face/jaw/eyes/brows/nose/lips], [expression/pose], wearing [clothing/accessories], [framing/setting].`
"""

OBJECT_PROMPT = """Create a concise natural-language training caption for an object LoRA from the provided image.

Rules:
- Start with the trigger phrase: `[trigger] object`
- Describe only visible, identity-relevant details of the object.
- Focus on: object type, shape, proportions, material, color, surface texture, distinguishing features, markings or labels, condition, orientation, and whether it is shown alone or with visible supporting context.
- Include parts or components only if they are clearly visible and important to the object's identity.
- Mention the background or setting only briefly, and only if it helps clarify presentation.
- Keep style descriptions minimal. Do not over-emphasize lighting, film grain, camera style, era, or color grading unless they are essential and visibly dominant.
- Do not write "the object," "reference image," or anything that depends on previous images.
- Do not use comma-separated tag spam or DeepBooru style.
- Do not infer brand, purpose, history, or material unless clearly visible.
- Keep the caption to one sentence, ideally 20-40 words.
- Output only the caption, no explanation.

Caption format:
`[trigger] object, a [object type] with [shape/proportions], [material/color/texture], [distinctive features/parts/markings], [orientation or presentation], [brief setting if useful].`
"""

PROMPTS = {
    "man": MAN_PROPMT,
    "object": OBJECT_PROMPT,
    "woman": WOMAN_PROMPT,
    "style": STYLE_PROMPT,
}


def iter_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def image_to_data_url(image_path: Path, max_res: int) -> str:
    from PIL import Image, ImageOps

    with Image.open(image_path) as pil_img:
        img = ImageOps.exif_transpose(pil_img)
        width, height = img.size
        longest = max(width, height)

        if longest > max_res:
            scale = max_res / longest
            new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        if img.mode in {"RGBA", "LA"} or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            background.alpha_composite(rgba)
            img = background.convert("RGB")
        else:
            img = img.convert("RGB")

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=95, optimize=True)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def caption_image(
    image_path: Path,
    trigger: str,
    *,
    caption_class: str,
    max_res: int,
    max_tokens: int,
    server_url: str = DEFAULT_SERVER_URL,
    model: str | None = None,
    print_response: bool = False,
) -> tuple[str, dict[str, Any]]:
    import requests

    payload: dict[str, object] = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(image_path, max_res),
                        },
                    },
                    {
                        "type": "text",
                        "text": PROMPTS[caption_class].replace("[trigger]", trigger),
                    },
                ],
            }
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    }
    if model:
        payload["model"] = model

    response = requests.post(server_url, json=payload, timeout=300)
    response.raise_for_status()

    if print_response:
        print("[caption] first response:")
        print(response.text)

    data = response.json()
    caption = data["choices"][0]["message"]["content"].strip()
    if not caption:
        raise ValueError("empty caption response")
    usage = data.get("usage")
    return caption, usage if isinstance(usage, dict) else {}


def format_usage(usage: dict[str, Any]) -> str:
    if not usage:
        return "tokens unknown"

    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    parts = []
    if prompt is not None:
        parts.append(f"prompt={prompt}")
    if completion is not None:
        parts.append(f"completion={completion}")
    if total is not None:
        parts.append(f"total={total}")
    return "tokens " + ", ".join(parts) if parts else "tokens unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Caption images with a local OpenAI-compatible vision model.")
    p.add_argument("-i", "--input", required=True, type=Path, help="Directory of images.")
    p.add_argument("--trigger", required=True, help="Trigger phrase to start each caption with.")
    p.add_argument(
        "--class",
        dest="caption_class",
        choices=sorted(PROMPTS),
        required=True,
        help="Caption prompt class.",
    )
    p.add_argument(
        "--max-res",
        type=int,
        default=768,
        help="Scale images down before captioning so the longest side is at most this size.",
    )
    p.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help=f"OpenAI-compatible chat completions endpoint. Default: {DEFAULT_SERVER_URL}",
    )
    p.add_argument("--max-tokens", type=int, default=512, help="Maximum response tokens.")
    p.add_argument("--print-response", action="store_true", help="Print the first raw server response.")
    p.add_argument("--model", help="Optional model name to include in the chat completions request.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = args.input.expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: {root} is not a directory.")
    if not args.trigger.strip():
        sys.exit("ERROR: --trigger must not be empty.")
    if args.max_res <= 0:
        sys.exit("ERROR: --max-res must be > 0.")
    if args.max_tokens <= 0:
        sys.exit("ERROR: --max-tokens must be > 0.")

    files = iter_images(root)
    if not files:
        print(f"[caption] no images found under {root}")
        return

    total = len(files)
    printed_first_response = False
    for i, img_path in enumerate(files, start=1):
        out_path = img_path.with_suffix(".txt")
        try:
            if out_path.exists() and out_path.read_text(encoding="utf-8", errors="ignore").strip():
                print(f"{i}/{total}: {img_path} ... skipped")
                continue
            if out_path.exists():
                print(f"{i}/{total}: {img_path} ... empty caption, regenerating")

            print_response = args.print_response and not printed_first_response
            if print_response:
                printed_first_response = True
            caption, usage = caption_image(
                img_path,
                args.trigger.strip(),
                caption_class=args.caption_class,
                max_res=args.max_res,
                max_tokens=args.max_tokens,
                server_url=args.server_url,
                model=args.model,
                print_response=print_response,
            )
            out_path.write_text(caption, encoding="utf-8")
            print(f"{i}/{total}: {img_path} ... done ({format_usage(usage)})")
        except Exception as exc:
            print(f"{i}/{total}: {img_path} ... error ({exc})")


if __name__ == "__main__":
    main()
