import os
import shutil
import torch
from PIL import Image
from torchvision import transforms

from image.cull.model import IMLCullModel
from utils.stages import find_latest_stage

def perform_culling(project_dir, stage=None, batch_size=64):
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
        return 0, 0

    src_dir = stage_dir_path
    next_stage = stage + 1
    dest_dir = os.path.join(project_dir, f"stage_{next_stage}")
    model_path = os.path.join(project_dir, f"stage_{stage}_cull_model.pth")

    if os.path.exists(dest_dir):
        print(f"Removing existing stage_{next_stage} directory...")
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)

    if not os.path.exists(model_path):
        print(f"No model found at {model_path}. Please ensure the model is trained.")
        return 0, 0

    print(f"Loading model from {model_path}")
    print(f"Using device: {device}")
    try:
        model = IMLCullModel(pretrained=False).to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 0, 0

    image_files = [f for f in os.listdir(src_dir) if not os.path.isdir(os.path.join(src_dir, f))]
    total_images = len(image_files)
    print(f"Processing {total_images} images in batches of {batch_size}...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    culled_count = 0
    kept_count = 0
    batch_count = (total_images + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(batch_count):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_images)
            current_batch = image_files[start_idx:end_idx]

            print(f"Processing batch {batch_idx+1}/{batch_count} ({len(current_batch)} images)...")

            tensors = []
            names = []
            for img_name in current_batch:
                img_path = os.path.join(src_dir, img_name)
                try:
                    image = Image.open(img_path).convert("RGB")
                    tensors.append(transform(image))
                    names.append(img_name)
                except Exception as e:
                    print(f"Error processing image {img_name}: {e}")

            if not tensors:
                continue

            batch_tensor = torch.stack(tensors, dim=0).to(device)
            logits = model(batch_tensor)          # (B,)
            probs = torch.sigmoid(logits)         # Cull probability if label 1 = cull
            preds = (probs >= 0.5).long().cpu()   # 0=keep, 1=cull

            for img_name, pred in zip(names, preds):
                src_path = os.path.join(src_dir, img_name)
                if int(pred.item()) == 0:
                    shutil.copy(src_path, os.path.join(dest_dir, img_name))
                    kept_count += 1
                    print(f"Keeping {img_name} → stage_{next_stage}")
                else:
                    culled_count += 1
                    print(f"Culling {img_name}")

    processed_count = kept_count + culled_count
    print(f"Processed {processed_count}/{total_images} images.")
    if processed_count < total_images:
        print(f"Warning: {total_images - processed_count} images could not be processed due to errors.")
    print(f"Culling complete. Kept {kept_count}, culled {culled_count}. Saved to stage_{next_stage}.")

    return culled_count, kept_count

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Use a trained model to predict which images to cull')
    parser.add_argument('--project', required=True, type=str)
    parser.add_argument('--stage', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()

    perform_culling(args.project, args.stage, args.batch_size)
