import os
import csv
import random
import torch
from PIL import Image
from torchvision import transforms
from image.rank.model import IMLRankModel
from utils.stages import find_latest_stage


def elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def elo_update(ra: float, rb: float, sa: float, k: float = 32.0) -> tuple[float, float]:
    ea = elo_expected(ra, rb)
    eb = 1.0 - ea
    ra_new = ra + k * (sa - ea)
    rb_new = rb + k * ((1.0 - sa) - eb)
    return ra_new, rb_new


def list_stage_images(stage_dir: str) -> list[str]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = []
    for f in os.listdir(stage_dir):
        if os.path.isfile(os.path.join(stage_dir, f)):
            _, ext = os.path.splitext(f)
            if ext.lower() in exts:
                files.append(f)
    return files


def score_images(model: IMLRankModel, image_dir: str, images: list[str], batch_size: int, device: str) -> dict[str, float]:
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    scores = {}
    sizes = {}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_files = images[i:i + batch_size]
            tensors = []
            names = []
            for name in batch_files:
                img_path = os.path.join(image_dir, name)
                try:
                    image = Image.open(img_path).convert("RGB")
                    sizes[name] = image.size
                    tensors.append(transform(image))
                    names.append(name)
                except Exception as e:
                    print(f"Error processing image {name}: {e}")
            if not tensors:
                continue
            batch_tensor = torch.stack(tensors, dim=0).to(device)
            batch_scores = model(batch_tensor).cpu().tolist()
            for name, score in zip(names, batch_scores):
                scores[name] = float(score)
    return scores, sizes


def rank_stage(project_dir: str, stage: int | None = None, comparisons: int = 8, batch_size: int = 32) -> str | None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        if stage is None:
            stage = find_latest_stage(project_dir)
            print(f"Using latest stage: {stage}")
        stage_dir_path = os.path.join(project_dir, f"stage_{stage}")
        if not os.path.exists(stage_dir_path):
            raise ValueError(f"Stage directory not found: {stage_dir_path}")
    except ValueError as e:
        print(f"Error: {e}")
        return None

    model_path = os.path.join(project_dir, f"stage_{stage}_rank_model.pth")
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}. Please ensure the model is trained.")
        return None

    images = list_stage_images(stage_dir_path)
    if len(images) < 2:
        print(f"Need at least 2 images to rank. Found {len(images)}.")
        return None

    print(f"Loading model from {model_path}")
    print(f"Using device: {device}")
    model = IMLRankModel().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"Scoring {len(images)} images...")
    scores, sizes = score_images(model, stage_dir_path, images, batch_size, device)
    if len(scores) < 2:
        print("Not enough images could be scored to compute rankings.")
        return None

    ratings = {name: 1000.0 for name in scores.keys()}
    k_factor = 32.0
    comparisons = max(1, int(comparisons))

    image_list = list(scores.keys())
    random.shuffle(image_list)
    for name in image_list:
        res = sizes.get(name)
        opponents = [n for n in image_list if n != name and sizes.get(n) == res]
        if not opponents:
            continue
        sample_count = min(comparisons, len(opponents))
        for opp in random.sample(opponents, sample_count):
            score_a = scores[name]
            score_b = scores[opp]
            if score_a == score_b:
                sa = 0.5
            else:
                sa = 1.0 if score_a > score_b else 0.0
            ra, rb = ratings[name], ratings[opp]
            ra_new, rb_new = elo_update(ra, rb, sa, k_factor)
            ratings[name] = ra_new
            ratings[opp] = rb_new

    out_path = os.path.join(project_dir, f"stage_{stage}_rankings.csv")
    sorted_rows = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["image", "elo"])
        for name, rating in sorted_rows:
            w.writerow([name, f"{rating:.4f}"])

    print(f"Wrote rankings to {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rank images in a stage using Elo ratings.")
    parser.add_argument("--project", required=True, type=str)
    parser.add_argument("--stage", type=int, default=None)
    parser.add_argument("--comparisons", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    rank_stage(args.project, args.stage, args.comparisons, args.batch_size)
