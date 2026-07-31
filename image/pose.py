from __future__ import annotations

import argparse
import sys
from pathlib import Path

IMAGE_EXTS: set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

# COCO 17 keypoint names (indices 0-16)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# COCO skeleton connections as (i, j) pairs
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # face (nose -> eyes -> ears)
    (5, 6),                                 # shoulders
    (5, 7), (7, 9),                         # left arm
    (6, 8), (8, 10),                        # right arm
    (5, 11), (6, 12),                       # shoulders -> hips
    (11, 12),                               # hips
    (11, 13), (13, 15),                     # left leg
    (12, 14), (14, 16),                     # right leg
]

# Indices of face keypoints that --no-face excludes
FACE_INDICES = {0, 1, 2, 3, 4}


def iter_images(root: Path, recursive: bool = False) -> list[Path]:
    pattern = root.rglob if recursive else root.glob
    return sorted(
        p for p in pattern("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and not p.name.lower().endswith(".pose.png")
    )


def draw_pose(
    h: int,
    w: int,
    keypoints: "np.ndarray",
    min_conf: float = 0.5,
    no_face: bool = False,
) -> "np.ndarray":
    import cv2
    import numpy as np
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    valid = keypoints[:, 2] > min_conf
    if no_face:
        valid[list(FACE_INDICES)] = False

    for i, j in SKELETON:
        if no_face and (i in FACE_INDICES or j in FACE_INDICES):
            continue
        if valid[i] and valid[j]:
            pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
            pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
            cv2.line(canvas, pt1, pt2, (255, 255, 255), 2)

    for i in range(len(keypoints)):
        if valid[i]:
            pt = (int(keypoints[i, 0]), int(keypoints[i, 1]))
            cv2.circle(canvas, pt, 4, (255, 255, 255), -1)

    return canvas


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect poses in images and render skeleton images in per-image ref directories (pose.png)."
    )
    p.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="Directory of images to process",
    )
    p.add_argument(
        "--recursive", "-r",
        action="store_true",
        default=False,
        help="Recurse into subdirectories.",
    )
    p.add_argument(
        "--model",
        required=True,
        help="Model path or bare name (auto-downloaded by ultralytics).",
    )
    p.add_argument(
        "--res",
        type=int,
        default=640,
        help="Inference resolution (imgsz). Higher values improve small-body-part detection. Default: 640",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Keypoint confidence threshold. Default: 0.5",
    )
    p.add_argument(
        "--no-face",
        action="store_true",
        default=False,
        help="Exclude facial keypoints (nose, eyes, ears) from the rendered skeleton.",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Re-process images that already have a .pose.png file.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input.expanduser().resolve()
    if not input_dir.is_dir():
        sys.exit(f"ERROR: --input is not a directory: {input_dir}")

    files = iter_images(input_dir, recursive=args.recursive)
    if not files:
        print(f"[pose] no images found under {input_dir}")
        return

    model_spec = args.model
    p = Path(model_spec)
    if not p.is_file() and not p.is_absolute():
        model_spec = args.model

    print(f"[pose] loading model ({model_spec}) ...")
    import cv2
    import numpy as np
    from PIL import Image
    from ultralytics import YOLO
    model = YOLO(model_spec)

    total = len(files)
    for i, img_path in enumerate(files, start=1):
        out_path = img_path.with_suffix(".pose.png")
        try:
            if not args.reset and out_path.exists():
                print(f"{i}/{total}: {img_path.name} ... skipped")
                continue
            if out_path.exists():
                print(f"{i}/{total}: {img_path.name} ... reset, reprocessing")

            results = model(img_path, conf=args.conf, imgsz=args.res, device="cuda")[0]

            pil_img = Image.open(img_path)
            h, w = pil_img.height, pil_img.width

            canvas = np.zeros((h, w, 3), dtype=np.uint8)

            if results.keypoints is not None and results.keypoints.data.numel():
                for person_kpts in results.keypoints.data:
                    kpts_np = person_kpts.cpu().numpy()
                    person_canvas = draw_pose(h, w, kpts_np, min_conf=args.conf, no_face=args.no_face)
                    canvas = np.maximum(canvas, person_canvas)

            cv2.imwrite(str(out_path), canvas)
            person_count = len(results.keypoints.data) if results.keypoints is not None and results.keypoints.data.numel() else 0
            print(f"{i}/{total}: {img_path.name} -> {out_path.name} ({person_count} pose(s))")
        except Exception as exc:
            print(f"{i}/{total}: {img_path.name} ... error ({exc})")


if __name__ == "__main__":
    main()
