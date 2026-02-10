from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

IMAGE_EXTS: set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

IMAGE_SIZE = 448


def load_tags(tags_csv: Path) -> tuple[list[str], list[str]]:
    with tags_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    rows = rows[1:]
    names = [r[1] for r in rows]
    cats = [r[2] for r in rows]
    return names, cats


def preprocess(pil_img: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(pil_img).convert("RGB")
    img = np.array(img)
    img = img[:, :, ::-1]  # RGB -> BGR

    size = max(img.shape[0], img.shape[1])
    pad_x = size - img.shape[1]
    pad_y = size - img.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2

    img = np.pad(
        img,
        ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)),
        mode="constant",
        constant_values=255,
    )

    interp = cv2.INTER_AREA if size > IMAGE_SIZE else cv2.INTER_LANCZOS4
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)

    return img.astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_session_cuda_only(onnx_path: Path) -> ort.InferenceSession:
    # CUDA only (no CPU fallback). Will error if CUDAExecutionProvider isn't available.
    return ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider"])


def run_model(sess: ort.InferenceSession, input_name: str, x: np.ndarray) -> np.ndarray:
    y = sess.run(None, {input_name: x})[0]
    y = np.asarray(y)
    if y.ndim == 2:
        y = y[0]
    return y if (y.min() >= 0.0 and y.max() <= 1.0) else sigmoid(y)


def infer_channels_first(input_shape: list[int | None] | tuple[int | None, ...]) -> bool:
    if len(input_shape) != 4:
        return False
    _, d1, d2, d3 = input_shape
    if d1 == 3:
        return True
    if d3 == 3:
        return False
    if d1 is None and d2 == IMAGE_SIZE and d3 == IMAGE_SIZE:
        return True
    if d3 is None and d1 == IMAGE_SIZE and d2 == IMAGE_SIZE:
        return False
    return False


def prepare_input(x_hwc: np.ndarray, channels_first: bool) -> np.ndarray:
    if channels_first:
        x = np.transpose(x_hwc, (2, 0, 1))  # HWC -> CHW
    else:
        x = x_hwc
    return x[None, ...]


def tag_image(
    image_path: Path,
    sess: ort.InferenceSession,
    input_name: str,
    names: list[str],
    cats: list[str],
    *,
    thresh: float,
    general_thresh: float | None,
    character_thresh: float | None,
    channels_first: bool,
) -> list[tuple[str, float]]:
    if general_thresh is None:
        general_thresh = thresh
    if character_thresh is None:
        character_thresh = thresh

    with Image.open(image_path) as im:
        x = prepare_input(preprocess(im), channels_first)

    probs = run_model(sess, input_name, x)

    out: list[tuple[str, float]] = []
    limit = min(len(probs), len(names), len(cats))

    for i in range(4, limit):  # skip rating labels
        p = float(probs[i])
        cat = cats[i]
        name = names[i]
        if cat == "0" and p >= general_thresh:
            out.append((name, p))
        elif cat == "4" and p >= character_thresh:
            out.append((name, p))

    out.sort(key=lambda t: t[1], reverse=True)
    return out


def iter_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def resolve_model_paths(model_dir: Path) -> tuple[Path, Path]:
    p = model_dir.expanduser().resolve()
    if not p.is_dir():
        sys.exit(f"ERROR: model dir not found: {p}")

    onnx_path = p / "model.onnx"
    tags_path = p / "selected_tags.csv"

    if not onnx_path.is_file():
        sys.exit(f"ERROR: model.onnx not found: {onnx_path}")
    if not tags_path.is_file():
        sys.exit(f"ERROR: selected_tags.csv not found: {tags_path}")

    return onnx_path, tags_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tag images and write captions beside them (ONNXRuntime CUDA).")
    p.add_argument("-i", "--input", required=True, type=Path, help="Directory of images.")
    p.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Directory containing model.onnx + selected_tags.csv.",
    )
    p.add_argument("--prefix", type=str, default="", help="String to prepend to each caption.")
    p.add_argument("--thresh", type=float, default=0.35)
    p.add_argument("--general-thresh", type=float)
    p.add_argument("--character-thresh", type=float)
    p.add_argument("--max-tags", type=int, default=50, help="Limit tags written per image (0 for none).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = args.input.expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: {root} is not a directory.")
    if args.max_tags is not None and args.max_tags < 0:
        sys.exit("ERROR: --max-tags must be >= 0.")

    onnx_path, tags_path = resolve_model_paths(args.model)
    names, cats = load_tags(tags_path)

    try:
        sess = build_session_cuda_only(onnx_path)
    except Exception as exc:
        sys.exit(f"ERROR: failed to create CUDA session (CUDAExecutionProvider): {exc}")

    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    channels_first = infer_channels_first(input_shape)

    files = iter_images(root)
    if not files:
        print(f"[tagger] no images found under {root}")
        return

    total = len(files)
    for i, img_path in enumerate(files, start=1):
        try:
            out_path = img_path.with_suffix(".txt")
            if out_path.exists():
                print(f"{i}/{total}: {img_path} ... skipped")
                continue
            tags = tag_image(
                img_path,
                sess,
                input_name,
                names,
                cats,
                thresh=args.thresh,
                general_thresh=args.general_thresh,
                character_thresh=args.character_thresh,
                channels_first=channels_first,
            )
            if args.max_tags is not None:
                tags = tags[: args.max_tags]
            caption = f"{args.prefix}{', '.join([t for t, _ in tags])}"
            out_path.write_text(caption, encoding="utf-8")
            print(f"{i}/{total}: {img_path} ... done")
        except Exception as exc:
            print(f"{i}/{total}: {img_path} ... error ({exc})")


if __name__ == "__main__":
    main()
