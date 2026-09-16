import csv
import os
import random
import shutil
from collections import defaultdict
from typing import Dict, List, Tuple

from image.crop.ratios import normalize_ratio, pair_sort_key

Label = Tuple[str, int, str, float, float, float, float]
Pair = Tuple[int, str]
CLASS_NAMES: Dict[int, str] = {
    0: "face",
    1: "head_shot",
    2: "half_body",
    3: "cowboy_shot",
    4: "three_quarter",
    5: "full_body",
}


def read_labels_csv(csv_path: str) -> List[Label]:
    labels: List[Label] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            try:
                img, cls_s, ratio_s, cx_s, cy_s, hx_s, hy_s = row
                labels.append((
                    img,
                    int(cls_s),
                    normalize_ratio(ratio_s),
                    float(cx_s),
                    float(cy_s),
                    float(hx_s),
                    float(hy_s),
                ))
            except Exception:
                continue
    return labels


def make_dirs(base_out: str):
    for split in ("train", "val"):
        os.makedirs(os.path.join(base_out, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(base_out, split, "labels"), exist_ok=True)


def write_yolo_label_file(path: str, labels: List[Label], pair_id_map: Dict[Pair, int]):
    with open(path, "w", newline="") as f:
        for (_, cls_id, ratio, cx, cy, hx, hy) in labels:
            mapped_cls = pair_id_map[(cls_id, ratio)]
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

    by_class: Dict[Pair, List[Label]] = defaultdict(list)
    for lab in all_labels:
        by_class[(lab[1], lab[2])].append(lab)
    if not by_class:
        return {}, {}

    if balance:
        cap = min(len(v) for v in by_class.values())
        labels_by_class: Dict[Pair, List[Label]] = {}
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

    classes = sorted(labels_by_class.keys(), key=pair_sort_key)

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
    pair_id_map: Dict[Pair, int],
):
    img_out = os.path.join(out_root, split_name, "images")
    lbl_out = os.path.join(out_root, split_name, "labels")

    for img_name, lbls in split_labels_by_img.items():
        src = os.path.join(images_root, img_name)
        dst = os.path.join(img_out, img_name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

        base, _ = os.path.splitext(img_name)
        lbl_path = os.path.join(lbl_out, base + ".txt")
        write_yolo_label_file(lbl_path, lbls, pair_id_map)


def run(
    project: str,
    stage: int,
    val_split: float = 0.2,
    balance: bool = False,
    seed: int = 19930625,
):
    if not (0.0 <= val_split <= 1.0):
        raise SystemExit("--val_split must be between 0 and 1")

    stage_dir = os.path.join(project, f"stage_{stage}")
    if not os.path.isdir(stage_dir):
        raise SystemExit(f"Stage directory not found: {stage_dir}")

    csv_path = os.path.join(project, f"stage_{stage}_crop_labels.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(f"Labels CSV not found: {csv_path}")

    all_labels = read_labels_csv(csv_path)
    if not all_labels:
        raise SystemExit("No labels found in CSV.")

    present_pairs = sorted({(lab[1], lab[2]) for lab in all_labels}, key=pair_sort_key)
    pair_id_map = {pair: idx for idx, pair in enumerate(present_pairs)}
    class_names = [
        f"{CLASS_NAMES.get(cls_id, f'class_{cls_id}')}_{ratio}"
        for cls_id, ratio in present_pairs
    ]
    pair_counts = {pair: 0 for pair in present_pairs}
    for _, cls_id, ratio, *_ in all_labels:
        pair_counts[(cls_id, ratio)] += 1

    print(f"Loaded {len(all_labels)} labels across {len(present_pairs)} class/ratio combinations from {csv_path}")
    for (cls_id, ratio), count in pair_counts.items():
        print(f"  {CLASS_NAMES.get(cls_id, f'class_{cls_id}')}_{ratio}: {count}")
    if balance:
        print(f"Balancing enabled: capping each combination to {min(pair_counts.values())} labels before splitting.")

    train_by_img, val_by_img = balance_and_split(
        all_labels,
        val_split,
        balance=balance,
        seed=seed,
    )

    out_root = os.path.join(project, f"stage_{stage}_yolo")
    make_dirs(out_root)

    copy_and_write(train_by_img, stage_dir, out_root, "train", pair_id_map)
    copy_and_write(val_by_img, stage_dir, out_root, "val", pair_id_map)

    yaml_path = os.path.join(project, f"stage_{stage}_yolo.yaml")
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

    n_train_imgs = len(train_by_img)
    n_val_imgs = len(val_by_img)
    n_train_labels = sum(len(v) for v in train_by_img.values())
    n_val_labels = sum(len(v) for v in val_by_img.values())
    print(
        f"Done. Train: {n_train_imgs} images / {n_train_labels} labels; "
        f"Val: {n_val_imgs} images / {n_val_labels} labels"
    )
    print(f"Output: {out_root}/train and {out_root}/val")
