import argparse
import csv
import os
import random
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple

from utils.stages import find_latest_stage

Label = Tuple[str, int, float, float, float, float]


def read_labels_csv(csv_path: str) -> List[Label]:
    labels: List[Label] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            try:
                img, cls_s, cx_s, cy_s, hx_s, hy_s = row
                labels.append((img, int(cls_s), float(cx_s), float(cy_s), float(hx_s), float(hy_s)))
            except Exception:
                continue
    return labels


def make_dirs(base_out: str):
    for split in ("train", "val"):
        os.makedirs(os.path.join(base_out, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(base_out, split, "labels"), exist_ok=True)


def write_yolo_label_file(path: str, labels: List[Label]):
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

    - Balance per class down to the least-labeled class (cap).
    - Split each class into train/val with an exact val fraction.
    - Allows the same image to be present in both splits (labels are split).
    """
    rnd = random.Random(seed)

    # group labels by class
    by_class: Dict[int, List[Label]] = defaultdict(list)
    for lab in all_labels:
        by_class[lab[1]].append(lab)
    if not by_class:
        return {}, {}

    cap = min(len(v) for v in by_class.values())

    # cap per class
    capped_by_class: Dict[int, List[Label]] = {}
    for c, lst in by_class.items():
        lst_copy = lst[:]
        rnd.shuffle(lst_copy)
        capped_by_class[c] = lst_copy[:cap]

    classes = sorted(capped_by_class.keys())
    per_class_val_counts: Dict[int, int] = {}
    raw_targets = [val_split * cap for _ in classes]
    floors = [int(x) for x in map(lambda x: int(x), [int(val_split * cap) for _ in classes])]
    fracs = [(i, (raw_targets[i] - int(raw_targets[i]))) for i in range(len(classes))]
    remaining = int(round(sum(raw_targets))) - sum(floors)
    
    for i, c in enumerate(classes):
        per_class_val_counts[c] = floors[i]
        
    fracs.sort(key=lambda t: t[1], reverse=True)
    for i in range(remaining):
        per_class_val_counts[classes[fracs[i][0]]] += 1

    train_by_img: Dict[str, List[Label]] = defaultdict(list)
    val_by_img: Dict[str, List[Label]] = defaultdict(list)

    for c in classes:
        lst = capped_by_class[c][:]
        rnd.shuffle(lst)
        n_val = per_class_val_counts[c]
        val_part = lst[:n_val]
        train_part = lst[n_val:]

        for lab in val_part:
            val_by_img[lab[0]].append(lab)
        for lab in train_part:
            train_by_img[lab[0]].append(lab)

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

    yaml_path = os.path.join(args.project, f"stage_{stage}_yolo.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"# YOLO dataset config for stage {stage}\n")
        f.write(f"path: {out_root}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n\n")
        f.write("nc: 4\n")
        f.write("names:\n")
        f.write("  0: face\n")
        f.write("  1: portrait\n")
        f.write("  2: landscape\n")
        f.write("  3: full_body\n")
    print(f"Wrote dataset YAML: {yaml_path}")

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
