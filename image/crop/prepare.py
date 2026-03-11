import argparse
import csv
import os
import random
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple

from utils.stages import find_latest_stage

Label = Tuple[str, int, float, float, float, float]
CLASS_NAMES: Dict[int, str] = {
    0: "face",
    1: "portrait",
    2: "landscape",
    3: "full_body",
}


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


def write_yolo_label_file(path: str, labels: List[Label], class_id_map: Dict[int, int]):
    with open(path, "w", newline="") as f:
        for (_, cls_id, cx, cy, hx, hy) in labels:
            mapped_cls = class_id_map.get(cls_id, cls_id)
            w = 2.0 * hx
            h = 2.0 * hy
            f.write(f"{mapped_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def balance_and_split(
    all_labels: List[Label],
    val_split: float,
    balance: bool = False,
    seed: int = 42,
) -> Tuple[Dict[str, List[Label]], Dict[str, List[Label]]]:
    """
    Returns (train_labels_by_img, val_labels_by_img)

    - Optionally balance per class down to the least-labeled class (cap).
    - Split each class into train/val.
    - Allows the same image to be present in both splits (labels are split).
    """
    rnd = random.Random(seed)

    # group labels by class
    by_class: Dict[int, List[Label]] = defaultdict(list)
    for lab in all_labels:
        by_class[lab[1]].append(lab)
    if not by_class:
        return {}, {}

    if balance:
        cap = min(len(v) for v in by_class.values())
        labels_by_class: Dict[int, List[Label]] = {}
        for c, lst in by_class.items():
            lst_copy = lst[:]
            rnd.shuffle(lst_copy)
            labels_by_class[c] = lst_copy[:cap]
    else:
        labels_by_class = {}
        for c, lst in by_class.items():
            lst_copy = lst[:]
            rnd.shuffle(lst_copy)
            labels_by_class[c] = lst_copy

    classes = sorted(labels_by_class.keys())

    train_by_img: Dict[str, List[Label]] = defaultdict(list)
    val_by_img: Dict[str, List[Label]] = defaultdict(list)

    for c in classes:
        lst = labels_by_class[c][:]
        n_val = int(round(val_split * len(lst)))
        if 0.0 < val_split < 1.0 and len(lst) > 1:
            n_val = max(1, min(len(lst) - 1, n_val))
        else:
            n_val = max(0, min(len(lst), n_val))
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
    class_id_map: Dict[int, int],
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
        write_yolo_label_file(lbl_path, lbls, class_id_map)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Path to project root")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation fraction (0..1)")
    parser.add_argument("--stage", type=int, default=None, help="Stage number (optional)")
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Downsample each class to the smallest class count before splitting",
    )
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

    present_classes = sorted({lab[1] for lab in all_labels})
    class_id_map = {cls_id: idx for idx, cls_id in enumerate(present_classes)}
    class_names = [CLASS_NAMES.get(cls_id, f"class_{cls_id}") for cls_id in present_classes]
    class_counts = {cls_id: 0 for cls_id in present_classes}
    for _, cls_id, *_ in all_labels:
        class_counts[cls_id] += 1

    print(f"Loaded {len(all_labels)} labels across {len(present_classes)} classes from {csv_path}")
    for cls_id in present_classes:
        print(f"  class {cls_id} ({CLASS_NAMES.get(cls_id, f'class_{cls_id}')}): {class_counts[cls_id]}")
    if args.balance:
        print(f"Balancing enabled: capping each class to {min(class_counts.values())} labels before splitting.")

    # Balance & split
    train_by_img, val_by_img = balance_and_split(
        all_labels,
        args.val_split,
        balance=args.balance,
        seed=args.seed,
    )

    # Prepare output structure under <project>/yolo
    out_root = os.path.join(args.project, f"stage_{stage}_yolo")
    make_dirs(out_root)

    # Copy and write
    copy_and_write(train_by_img, stage_dir, out_root, "train", class_id_map)
    copy_and_write(val_by_img, stage_dir, out_root, "val", class_id_map)

    yaml_path = os.path.join(args.project, f"stage_{stage}_yolo.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"# YOLO dataset config for stage {stage}\n")
        f.write(f"path: {out_root}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names:\n")
        for idx, name in enumerate(class_names):
            f.write(f"  {idx}: {name}\n")
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
