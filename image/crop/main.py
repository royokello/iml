import argparse, csv, os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from utils.stages import find_latest_stage
import numpy as np

CLASS_RATIOS = {0: (1, 1), 1: (3, 4), 2: (4, 3), 3: (1, 2), 4: (2, 1)}  # w:h


def read_yolo_class_mapping(project: str, stage: int) -> tuple[dict[int, int], dict[int, int]]:
    labels_path = os.path.join(project, f"stage_{stage}_crop_labels.csv")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(labels_path)

    original_classes = set()
    with open(labels_path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            try:
                original_classes.add(int(row[1]))
            except (IndexError, TypeError, ValueError):
                continue

    if not original_classes:
        raise RuntimeError(f"No classes found in {labels_path}")

    present_classes = sorted(original_classes)
    yolo_to_original = {idx: cls_id for idx, cls_id in enumerate(present_classes)}
    original_to_yolo = {cls_id: idx for idx, cls_id in yolo_to_original.items()}
    return yolo_to_original, original_to_yolo


def yolo_ratio_map(yolo_to_original: dict[int, int]) -> dict[int, tuple[int, int]]:
    return {
        yolo_id: CLASS_RATIOS.get(original_cls_id, (1, 1))
        for yolo_id, original_cls_id in yolo_to_original.items()
    }


def map_requested_classes(classes, original_to_yolo: dict[int, int]) -> list[int] | None:
    if classes is None:
        return None

    mapped = []
    for cls_id in classes:
        if cls_id in original_to_yolo:
            mapped.append(original_to_yolo[cls_id])

    return sorted(set(mapped))

def list_images(d: str):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")
    return sorted([f for f in os.listdir(d) if f.lower().endswith(exts)])

def expand_to_ratio(xyxy, ratio_w, ratio_h, W, H):
    # Enforce target aspect ratio exactly, even at image borders.
    x1, y1, x2, y2 = map(float, xyxy)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)

    target_ar = ratio_w / ratio_h  # width / height

    # First, adjust current box to target AR (keep center)
    if bw / bh > target_ar:
        bh = bw / target_ar
    else:
        bw = bh * target_ar

    # Compute the max size that still fits inside the image, keeping AR
    hx_max = min(cx, W - cx)  # max half-width allowed by borders
    hy_max = min(cy, H - cy)  # max half-height allowed

    # Scale factor to fit within bounds (<= 1)
    scale = min((hx_max * 2) / bw, (hy_max * 2) / bh, 1.0)
    bw *= scale
    bh *= scale

    # Convert to integers while preserving AR exactly
    # Derive height first, then recompute width from AR to avoid drift.
    height = max(1, int(round(bh)))
    width  = max(1, int(round(height * target_ar)))

    # If rounding pushed us out of bounds, shrink one step
    if width > int(2 * hx_max):
        width = int(2 * hx_max)
        height = max(1, int(round(width / target_ar)))
    if height > int(2 * hy_max):
        height = int(2 * hy_max)
        width  = max(1, int(round(height * target_ar)))

    # Final box around center
    nx1 = int(round(cx - width / 2))
    ny1 = int(round(cy - height / 2))
    nx2 = nx1 + width
    ny2 = ny1 + height

    # Last safety clamp (shouldn’t change AR anymore)
    nx1 = max(0, min(nx1, W - width))
    ny1 = max(0, min(ny1, H - height))
    nx2 = nx1 + width
    ny2 = ny1 + height

    return nx1, ny1, nx2, ny2

def resize_long_side(img: Image.Image, target: int) -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        return img

    if w >= h:
        new_w = target
        new_h = max(1, int(round(target * (h / w))))
    else:
        new_h = target
        new_w = max(1, int(round(target * (w / h))))

    if new_w == w and new_h == h:
        return img
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def parse_resolution(value: str) -> int | None:
    lowered = value.strip().lower()
    if lowered in {"none", "null"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--resolution must be a positive integer or 'none'")
    return parsed


def parse_filename_width(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--filename-width must be a positive integer")
    return parsed


def perform_cropping(
    project: str,
    stage: int | None,
    resolution: int | None,
    conf: float,
    iou: float,
    max_det: int,
    classes,
    batch: int,
    filename_width: int,
):
    if stage is None:
        stage = find_latest_stage(project)
        print(f"Using latest stage: {stage}")

    src_dir = os.path.join(project, f"stage_{stage}")
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(src_dir)

    weights = os.path.join(project, f"stage_{stage}_crop_model.pt")
    if not os.path.isfile(weights):
        raise FileNotFoundError(weights)

    out_dir = os.path.join(project, f"stage_{stage+1}")
    os.makedirs(out_dir, exist_ok=True)

    model = YOLO(weights)
    yolo_to_original, original_to_yolo = read_yolo_class_mapping(project, stage)
    ratio_by_yolo_id = yolo_ratio_map(yolo_to_original)
    mapped_classes = map_requested_classes(classes, original_to_yolo)

    if classes is not None:
        print(f"Mapped requested classes {classes} to YOLO classes {mapped_classes}")
    print(
        "YOLO crop ratios: "
        + ", ".join(
            f"{yolo_id}->class_{yolo_to_original[yolo_id]}={rw}:{rh}"
            for yolo_id, (rw, rh) in sorted(ratio_by_yolo_id.items())
        )
    )
    
    if not list_images(src_dir):
        raise RuntimeError("No images found.")

    results = model.predict(
        source=src_dir,
        conf=conf,
        iou=iou,
        max_det=max_det,
        classes=mapped_classes,
        device="cuda",
        stream=True,
        save=False,
        verbose=True,
        batch=batch,
        workers=0,
    )

    counter = 1
    for res in results:
        imW, imH = res.orig_shape[1], res.orig_shape[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            continue

        im_np = res.orig_img
        if im_np is None:
            continue
        im = Image.fromarray(im_np[..., ::-1])
        
        best_by_class = {}
        for j in range(len(boxes)):
            cls_id = int(boxes.cls[j].item())
            conf = float(boxes.conf[j].item())
            if mapped_classes is not None and cls_id not in mapped_classes:
                continue
            # keep only highest confidence per class
            if cls_id not in best_by_class or conf > best_by_class[cls_id][0]:
                best_by_class[cls_id] = (conf, boxes.xyxy[j].tolist())

        # now crop once per class
        for cls_id, (_, (x1, y1, x2, y2)) in best_by_class.items():
            rw, rh = ratio_by_yolo_id.get(cls_id, (1,1))
            ex1, ey1, ex2, ey2 = expand_to_ratio((x1,y1,x2,y2), rw, rh, imW, imH)
            if ex2 <= ex1 or ey2 <= ey1:
                continue
            crop = im.crop((ex1, ey1, ex2, ey2))
            if resolution is not None:
                crop = resize_long_side(crop, resolution)
            save_path = os.path.join(out_dir, f"{counter:0{filename_width}d}.png")
            crop.save(save_path)
            counter += 1


    print(f"Saved {counter} crops → {out_dir}")

def main():
    ap = argparse.ArgumentParser("YOLO-based cropper")
    ap.add_argument("--project", required=True)
    ap.add_argument("--stage", type=int)
    ap.add_argument(
        "--resolution",
        type=parse_resolution,
        default=None,
        help="Long-side target size for output crops. Use an integer, or omit/'none' to keep native crop size.",
    )
    ap.add_argument("--conf", type=float, default=0.25) # confidence threshold: only keep detections above this score
    ap.add_argument("--iou", type=float, default=0.7) # IoU threshold for non-max suppression (higher = fewer boxes kept)
    ap.add_argument("--max_det", type=int, default=50) # maximum number of detections per image to return
    ap.add_argument("--classes", type=int, nargs="*")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument(
        "--filename-width",
        type=parse_filename_width,
        default=6,
        help="Zero-pad output filenames to this width.",
    )
    args = ap.parse_args()

    perform_cropping(
        project=args.project,
        stage=args.stage,
        resolution=args.resolution,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        classes=args.classes,
        batch=args.batch,
        filename_width=args.filename_width,
    )

if __name__ == "__main__":
    main()
