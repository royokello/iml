import argparse
import csv
import os
import random
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple

from utils.stages import find_latest_stage  # re-use your helper

# A label is (img_name, class_id, cx, cy, hx, hy)
Label = Tuple[str, int, float, float, float, float]


def read_labels_csv(csv_path: str) -> List[Label]:
    labels: List[Label] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # Expect: ["img", "class", "cx", "cy", "hx", "hy"]
        for row in reader:
            try:
                img, cls_s, cx_s, cy_s, hx_s, hy_s = row
                labels.append((img, int(cls_s), float(cx_s), float(cy_s), float(hx_s), float(hy_s)))
            except Exception:
                # Skip malformed rows
                continue
    return labels


def make_dirs(base_out: str):
    for split in ("train", "val"):
        os.makedirs(os.path.join(base_out, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(base_out, split, "labels"), exist_ok=True)


def write_yolo_label_file(path: str, labels: List[Label]):
    # YOLO expects: class cx cy w h, with w,h normalized full sizes
    with open(path, "w", newline="") as f:
        for (_, cls_id, cx, cy, hx, hy) in labels:
            w = 2.0 * hx
            h = 2.0 * hy
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def balance_and_split(
    all_labels: List[Label],
    val_split: float,
    seed: int = 42,
) -> Tuple[Dict[str, List[Label]], Dict[str, List[Label]]]:
    """
    Returns (train_labels_by_img, val_labels_by_img)
    - Balance per class to the size of the least-labeled class.
    - Split that balanced set into train/val by val_split per class.
    - Ensure no image leakage: if an image ends up in both splits, keep the side with more labels.
    """
    rnd = random.Random(seed)

    # group labels by class
    by_class: Dict[int, List[Label]] = defaultdict(list)
    for lab in all_labels:
        by_class[lab[1]].append(lab)

    if not by_class:
        return {}, {}

    # find cap = min class count
    cap = min(len(v) for v in by_class.values())

    # sample per class to cap
    capped_by_class: Dict[int, List[Label]] = {}
    for c, lst in by_class.items():
        lst_copy = lst[:]
        rnd.shuffle(lst_copy)
        capped_by_class[c] = lst_copy[:cap]

    # split per class into train/val
    train_sel: List[Label] = []
    val_sel: List[Label] = []
    for c, lst in capped_by_class.items():
        n_val = int(round(val_split * len(lst)))
        rnd.shuffle(lst)
        val_sel.extend(lst[:n_val])
        train_sel.extend(lst[n_val:])

    # collect by image
    train_by_img: Dict[str, List[Label]] = defaultdict(list)
    val_by_img: Dict[str, List[Label]] = defaultdict(list)
    for lab in train_sel:
        train_by_img[lab[0]].append(lab)
    for lab in val_sel:
        val_by_img[lab[0]].append(lab)

    # resolve collisions: images appearing in both splits
    collisions = set(train_by_img.keys()) & set(val_by_img.keys())
    for img in collisions:
        if len(train_by_img[img]) >= len(val_by_img[img]):
            # keep train; drop val labels for this image
            val_by_img.pop(img, None)
        else:
            # keep val; drop train labels for this image
            train_by_img.pop(img, None)

    return train_by_img, val_by_img


def copy_and_write(
    split_labels_by_img: Dict[str, List[Label]],
    images_root: str,
    out_root: str,
    split_name: str,
):
    img_out = os.path.join(out_root, split_name, "images")
    lbl_out = os.path.join(out_root, split_name, "labels")

    for img_name, lbls in split_labels_by_img.items():
        # copy image
        src = os.path.join(images_root, img_name)
        dst = os.path.join(img_out, img_name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

        # write label file (same basename, .txt)
        base, _ = os.path.splitext(img_name)
        lbl_path = os.path.join(lbl_out, base + ".txt")
        write_yolo_label_file(lbl_path, lbls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Path to project root")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation fraction (0..1)")
    parser.add_argument("--stage", type=int, default=None, help="Stage number (optional)")
    parser.add_argument("--seed", type=int, default=19930625, help="Random seed")
    args = parser.parse_args()

    if not (0.0 <= args.val_split <= 1.0):
        raise SystemExit("--val_split must be between 0 and 1")

    # Locate stage and paths
    stage = args.stage if args.stage is not None else find_latest_stage(args.project)
    stage_dir = os.path.join(args.project, f"stage_{stage}")
    if not os.path.isdir(stage_dir):
        raise SystemExit(f"Stage directory not found: {stage_dir}")

    csv_path = os.path.join(args.project, f"stage_{stage}_crop_labels.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(f"Labels CSV not found: {csv_path}")

    # Read labels
    all_labels = read_labels_csv(csv_path)
    if not all_labels:
        raise SystemExit("No labels found in CSV.")

    # Balance & split
    train_by_img, val_by_img = balance_and_split(all_labels, args.val_split, seed=args.seed)

    # Prepare output structure under <project>/yolo
    out_root = os.path.join(args.project, f"stage_{stage}_yolo")
    make_dirs(out_root)

    # Copy and write
    copy_and_write(train_by_img, stage_dir, out_root, "train")
    copy_and_write(val_by_img, stage_dir, out_root, "val")

    # Small summary
    n_train_imgs = len(train_by_img)
    n_val_imgs = len(val_by_img)
    n_train_labels = sum(len(v) for v in train_by_img.values())
    n_val_labels = sum(len(v) for v in val_by_img.values())
    print(f"Done. Train: {n_train_imgs} images / {n_train_labels} labels; "
          f"Val: {n_val_imgs} images / {n_val_labels} labels")
    print(f"Output: {out_root}/train and {out_root}/val")


if __name__ == "__main__":
    main()
