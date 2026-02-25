import argparse
import os
import random

from utils.stages import find_latest_stage

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_DEFAULT_BATCH_SIZE = 64


def list_stage_images(stage_dir: str) -> list[str]:
    files: list[str] = []
    for name in os.listdir(stage_dir):
        path = os.path.join(stage_dir, name)
        if not os.path.isfile(path):
            continue
        _, ext = os.path.splitext(name)
        if ext.lower() in _IMAGE_EXTENSIONS:
            files.append(name)
    files.sort()
    return files


def _clear_sample_dir_entries(sample_dir: str) -> tuple[int, int]:
    os.makedirs(sample_dir, exist_ok=True)

    removed = 0
    skipped = 0
    for entry in os.scandir(sample_dir):
        try:
            if entry.is_symlink():
                try:
                    os.unlink(entry.path)
                except IsADirectoryError:
                    os.rmdir(entry.path)
                removed += 1
                continue

            if entry.is_file(follow_symlinks=False):
                # Hard links are regular files, so clear file entries as well.
                os.unlink(entry.path)
                removed += 1
                continue

            skipped += 1
            print(f"Skipping non-file entry in sample directory: {entry.path}")
        except Exception as e:
            skipped += 1
            print(f"Error removing existing sample entry {entry.path}: {e}")

    return removed, skipped


def create_cull_sample(
    project_dir: str,
    stage: int | None,
    count: int,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> str | None:
    if count <= 0:
        print("Error: --count must be greater than 0.")
        return None
    if batch_size <= 0:
        print("Error: internal batch size must be greater than 0.")
        return None

    try:
        if stage is None:
            stage = find_latest_stage(project_dir)
            print(f"Using latest stage: {stage}")
        stage_dir = os.path.join(project_dir, f"stage_{stage}")
        if not os.path.isdir(stage_dir):
            raise ValueError(f"Stage directory not found: {stage_dir}")
    except ValueError as e:
        print(f"Error: {e}")
        return None

    image_files = list_stage_images(stage_dir)
    total_images = len(image_files)
    if total_images == 0:
        print(f"No images found in {stage_dir}.")
        return None
    if count > total_images:
        print(
            f"Error: requested {count} samples, but only {total_images} images are available in stage_{stage}."
        )
        return None

    model_path = os.path.join(project_dir, f"stage_{stage}_cull_model.pth")
    if not os.path.isfile(model_path):
        print(f"No model found at {model_path}. Please ensure the cull model is trained.")
        return None

    try:
        import torch
        from PIL import Image
        from torchvision import transforms

        from image.cull.model import IMLCullModel
    except Exception as e:
        print(f"Error importing cull model dependencies: {e}")
        return None

    sample_dir = os.path.join(project_dir, f"stage_{stage}_cull_samples")
    removed_entries, skipped_entries = _clear_sample_dir_entries(sample_dir)
    if removed_entries:
        print(f"Removed {removed_entries} existing file/link entries from {sample_dir}")
    if skipped_entries:
        print(f"Skipped {skipped_entries} non-file entries in {sample_dir}")

    selected = random.sample(image_files, count)
    selected.sort()
    print(f"Selected {count} random images from stage_{stage} for cull-model preview.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_path}")
    print(f"Using device: {device}")

    try:
        model = IMLCullModel().to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    linked_count = 0
    kept_count = 0
    culled_count = 0
    batch_count = (len(selected) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(batch_count):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(selected))
            current_batch = selected[start_idx:end_idx]
            print(f"Processing sample batch {batch_idx+1}/{batch_count} ({len(current_batch)} images)...")

            tensors = []
            names = []
            for img_name in current_batch:
                img_path = os.path.join(stage_dir, img_name)
                try:
                    with Image.open(img_path) as image:
                        rgb_image = image.convert("RGB")
                    tensors.append(transform(rgb_image))
                    names.append(img_name)
                except Exception as e:
                    print(f"Error processing image {img_name}: {e}")

            if not tensors:
                continue

            batch_tensor = torch.stack(tensors, dim=0).to(device)
            logits = model(batch_tensor)
            preds = logits.argmax(dim=1).cpu()  # 0=keep, 1=cull

            for img_name, pred in zip(names, preds):
                if int(pred.item()) == 0:
                    kept_count += 1
                    src = os.path.join(stage_dir, img_name)
                    dst = os.path.join(sample_dir, img_name)
                    try:
                        if os.path.lexists(dst):
                            os.unlink(dst)
                        os.link(src, dst)
                        linked_count += 1
                    except Exception as e:
                        print(f"Error linking keep sample {img_name}: {e}")
                else:
                    culled_count += 1

    processed_count = kept_count + culled_count
    print(f"Processed {processed_count}/{count} sampled images.")
    if processed_count < count:
        print(f"Warning: {count - processed_count} sampled images could not be processed due to errors.")
    print(
        f"Cull preview complete. Predicted keep={kept_count}, cull={culled_count}. "
        f"Linked {linked_count} predicted keeps to {sample_dir}."
    )
    return sample_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run the cull model on a random sample and hard-link predicted keeps for review."
    )
    parser.add_argument("--project", required=True, type=str, help="Path to project root")
    parser.add_argument("--stage", type=int, default=None, help="Stage number (optional)")
    parser.add_argument(
        "--count",
        required=True,
        type=int,
        help="Number of random stage images to evaluate with the cull model",
    )
    args = parser.parse_args()

    create_cull_sample(args.project, args.stage, args.count)


if __name__ == "__main__":
    main()
