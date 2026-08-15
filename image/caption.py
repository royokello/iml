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


def iter_images(root: Path, recursive: bool = False) -> list[Path]:
    pattern = root.rglob if recursive else root.glob
    return sorted(p for p in pattern("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


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
    prompt_text: str,
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
                        "text": prompt_text.replace("[trigger]", trigger),
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
    p.add_argument(
        "-r", "--recursive",
        action="store_true",
        default=False,
        help="Recurse into subdirectories. Default: scan only top-level images.",
    )
    p.add_argument("--trigger", required=True, help="Trigger phrase to start each caption with.")
    p.add_argument(
        "--prompt",
        "--prompt-file",
        dest="prompt_file",
        type=Path,
        required=True,
        help="Path to a text file containing the system prompt. The placeholder [trigger] is replaced with the --trigger value.",
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
    p.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Re-caption images that already have a caption file, overwriting the existing caption.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = args.input.expanduser().resolve()
    if root.is_file():
        if root.suffix.lower() not in IMAGE_EXTS:
            sys.exit(f"ERROR: {root} is not an image file.")
        files = [root]
    else:
        if not root.is_dir():
            sys.exit(f"ERROR: {root} is not a directory or image file.")
        files = iter_images(root, recursive=args.recursive)
    if not args.trigger.strip():
        sys.exit("ERROR: --trigger must not be empty.")
    if args.max_res <= 0:
        sys.exit("ERROR: --max-res must be > 0.")
    if args.max_tokens <= 0:
        sys.exit("ERROR: --max-tokens must be > 0.")

    prompt_file = args.prompt_file.expanduser().resolve()
    if not prompt_file.is_file():
        sys.exit(f"ERROR: --prompt file not found: {prompt_file}")
    prompt_text = prompt_file.read_text(encoding="utf-8", errors="ignore").strip()
    if not prompt_text:
        sys.exit(f"ERROR: --prompt file is empty: {prompt_file}")

    if not files:
        print(f"[caption] no images found under {root}")
        return

    total = len(files)
    printed_first_response = False
    for i, img_path in enumerate(files, start=1):
        out_path = img_path.with_suffix(".txt")
        try:
            if not args.reset and out_path.exists() and out_path.read_text(encoding="utf-8", errors="ignore").strip():
                print(f"{i}/{total}: {img_path} ... skipped")
                continue
            if out_path.exists():
                print(f"{i}/{total}: {img_path} ... {'reset' if args.reset else 'empty caption'}, regenerating")

            print_response = args.print_response and not printed_first_response
            if print_response:
                printed_first_response = True
            caption, usage = caption_image(
                img_path,
                args.trigger.strip(),
                prompt_text=prompt_text,
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
