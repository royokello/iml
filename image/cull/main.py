import argparse
import os
import shutil

import torch
from PIL import Image
from torchvision import transforms

from image.cull.model import IMLCullModel
from utils.stages import find_latest_stage


def _parse_numeric_stem(filename):
    stem, _ = os.path.splitext(filename)
    try:
        return int(stem)
    except ValueError:
        return None


def _get_ordered_image_files(src_dir, resume=None):
    numeric_files = []
    non_numeric_files = []

    for name in os.listdir(src_dir):
        path = os.path.join(src_dir, name)
        if os.path.isdir(path):
            continue

        numeric_stem = _parse_numeric_stem(name)
        if numeric_stem is None:
            non_numeric_files.append(name)
            continue

        numeric_files.append((numeric_stem, name))

    numeric_files.sort(key=lambda item: (item[0], item[1]))
    non_numeric_files.sort()

    numeric_total = len(numeric_files)
    non_numeric_total = len(non_numeric_files)
    skipped_below_start = 0
    skipped_non_numeric = 0

    if resume is not None:
        filtered_numeric = [item for item in numeric_files if item[0] >= resume]
        skipped_below_start = numeric_total - len(filtered_numeric)
        skipped_non_numeric = non_numeric_total
        ordered_files = [name for _, name in filtered_numeric]
    else:
        ordered_files = [name for _, name in numeric_files] + non_numeric_files

    return (
        ordered_files,
        numeric_total,
        non_numeric_total,
        skipped_below_start,
        skipped_non_numeric,
    )


def _keep_image_file(src_path, dest_path, mode):
    if mode == "copy":
        shutil.copy(src_path, dest_path)
        return
    if mode == "link":
        os.link(src_path, dest_path)
        return
    if mode == "inplace":
        shutil.move(src_path, dest_path)
        return
    raise ValueError(f"Unsupported cull output mode: {mode}")


def perform_culling(project_dir, stage=None, batch_size=64, resume=None, mode="copy"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        if stage is None:
            latest_stage = find_latest_stage(project_dir)
            if resume is not None:
                if latest_stage < 2:
                    raise ValueError(
                        "Resume mode requires at least stage_2 to exist (source is inferred as latest - 1)."
                    )
                stage = latest_stage - 1
                print(f"Resume mode: inferred source stage {stage} from latest stage {latest_stage}.")
            else:
                stage = latest_stage
                print(f"Using latest stage: {stage}")
        stage_dir_path = os.path.join(project_dir, f"stage_{stage}")
        if not os.path.exists(stage_dir_path):
            raise ValueError(f"Stage directory not found: {stage_dir_path}")
    except ValueError as e:
        print(f"Error: {e}")
        return 0, 0

    if batch_size <= 0:
        print("Error: --batch-size must be greater than 0.")
        return 0, 0
    if resume is not None and resume < 0:
        print("Error: --resume must be greater than or equal to 0.")
        return 0, 0
    if mode not in {"copy", "inplace", "link"}:
        print("Error: output mode must be one of copy, inplace, or link.")
        return 0, 0

    src_dir = stage_dir_path
    next_stage = stage + 1
    dest_dir = os.path.join(project_dir, f"stage_{next_stage}")
    model_path = os.path.join(project_dir, f"stage_{stage}_cull_model.pth")
    resume_mode = resume is not None

    if os.path.exists(dest_dir):
        if resume_mode:
            print(f"Resume mode: preserving existing stage_{next_stage} directory.")
        else:
            print(f"Removing existing stage_{next_stage} directory...")
            shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"No model found at {model_path}. Please ensure the model is trained.")
        return 0, 0

    print(f"Loading model from {model_path}")
    print(f"Using device: {device}")
    if mode == "copy":
        print(f"Output mode: copy kept images to stage_{next_stage} (default).")
    elif mode == "link":
        print(f"Output mode: hard-link kept images into stage_{next_stage}.")
    else:
        print(f"Output mode: inplace (move kept images to stage_{next_stage} and delete culled images).")
    try:
        model = IMLCullModel().to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 0, 0

    (
        image_files,
        numeric_total,
        non_numeric_total,
        skipped_below_start,
        skipped_non_numeric,
    ) = _get_ordered_image_files(src_dir, resume=resume)
    total_images = len(image_files)
    total_source_files = numeric_total + non_numeric_total

    print(
        f"Discovered {total_source_files} files in stage_{stage} "
        f"({numeric_total} numeric, {non_numeric_total} non-numeric)."
    )
    if resume_mode:
        print(f"Resume mode: processing numeric filenames with stem >= {resume} (inclusive).")
        if skipped_below_start:
            print(f"Skipping {skipped_below_start} numeric files below start.")
        if skipped_non_numeric:
            print("Skipping non-numeric filenames in resume mode (no numeric stem to compare).")
    elif non_numeric_total:
        print("Warning: non-numeric filenames will be processed after numeric-sorted files.")

    print(f"Processing {total_images} images in batches of {batch_size}...")
    if total_images == 0:
        print("No images matched the requested criteria.")

    transform = transforms.Compose([
        transforms.Resize((384, 384)),
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
                    with Image.open(img_path) as image:
                        rgb_image = image.convert("RGB")
                    tensors.append(transform(rgb_image))
                    names.append(img_name)
                except Exception as e:
                    print(f"Error processing image {img_name}: {e}")

            if not tensors:
                continue

            batch_tensor = torch.stack(tensors, dim=0).to(device)
            logits = model(batch_tensor)          # (B, 2) keep/cull logits
            preds = logits.argmax(dim=1).cpu()    # 0=keep, 1=cull

            for img_name, pred in zip(names, preds):
                src_path = os.path.join(src_dir, img_name)
                dest_path = os.path.join(dest_dir, img_name)
                try:
                    if int(pred.item()) == 0:
                        if resume_mode and os.path.exists(dest_path):
                            if mode == "inplace" and os.path.exists(src_path):
                                os.remove(src_path)
                                print(
                                    f"Keeping {img_name} (already exists in stage_{next_stage}; removed source)"
                                )
                            else:
                                print(f"Keeping {img_name} (already exists in stage_{next_stage})")
                        else:
                            _keep_image_file(src_path, dest_path, mode)
                            if mode == "link":
                                print(f"Keeping {img_name} -> stage_{next_stage} (hard link)")
                            elif mode == "inplace":
                                print(f"Keeping {img_name} -> stage_{next_stage} (moved)")
                            else:
                                print(f"Keeping {img_name} -> stage_{next_stage}")
                        kept_count += 1
                    else:
                        removed_dest = False
                        if (resume_mode or mode == "inplace") and os.path.exists(dest_path):
                            os.remove(dest_path)
                            removed_dest = True

                        if mode == "inplace":
                            if os.path.exists(src_path):
                                os.remove(src_path)
                            if removed_dest:
                                print(
                                    f"Culling {img_name} (deleted source and removed existing file from stage_{next_stage})"
                                )
                            else:
                                print(f"Culling {img_name} (deleted source)")
                        else:
                            if removed_dest:
                                print(f"Culling {img_name} (removed existing copy from stage_{next_stage})")
                            else:
                                print(f"Culling {img_name}")
                        culled_count += 1
                except Exception as e:
                    print(f"Error saving cull result for {img_name}: {e}")

    processed_count = kept_count + culled_count
    print(f"Processed {processed_count}/{total_images} images.")
    if processed_count < total_images:
        print(f"Warning: {total_images - processed_count} images could not be processed due to errors.")
    print(f"Culling complete. Kept {kept_count}, culled {culled_count}. Saved to stage_{next_stage}.")

    return culled_count, kept_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use a trained model to predict which images to cull')
    parser.add_argument('--project', required=True, type=str)
    parser.add_argument('--stage', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument(
        '--resume',
        type=int,
        default=None,
        help='Resume from numeric filename stem (inclusive), e.g. 14487 matches 014487.png',
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['copy', 'inplace', 'link'],
        default='copy',
        help='Output handling for kept images: copy (default), inplace (move keep/delete cull), or link (hard link).',
    )
    args = parser.parse_args()

    perform_culling(args.project, args.stage, args.batch_size, args.resume, mode=args.mode)
