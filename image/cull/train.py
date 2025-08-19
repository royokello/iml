from datetime import datetime
import os
import csv
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from typing import Tuple, Optional
from image.cull.dataset import IMLCullDataset
from image.cull.model import IMLCullModel
from utils.stages import find_latest_stage

def set_backbone_trainable(model: torch.nn.Module, train_backbone: bool):
    for name, p in model.backbone.named_parameters():
        if name.startswith("fc."):
            p.requires_grad = True
        else:
            p.requires_grad = train_backbone

    if not train_backbone:
        model.backbone.eval()
    else:
        model.backbone.train()


def train_epoch(model, dataloader, criterion, optimizer, dataset_size, device) -> float:
    model.train()
    running_loss = 0.0
    for pixel_values, labels, *_ in dataloader:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(pixel_values)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * pixel_values.size(0)
    return running_loss / dataset_size

def validate_epoch(model, dataloader, criterion, dataset_size, device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for pixel_values, labels, *_ in dataloader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            running_loss += loss.item() * pixel_values.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)
    return running_loss / dataset_size, (correct / total if total > 0 else 0.0)

def run_stage(
    training_stage: int,
    model: torch.nn.Module,
    num_epochs: int,
    learning_rate: float,
    patience: int,
    device: str,
    training_plateau: float,
    epoch_log_file: str,
    model_file: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    train_size: int,
    val_size: int,
    best_val_acc: float,
    train_backbone: bool,
) -> Tuple[torch.nn.Module, float]:
    set_backbone_trainable(model, train_backbone)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    epochs_no_improve = 0
    history_val_acc = []

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, train_size, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, val_size, device)
        improved = val_acc > best_val_acc
        
        epoch_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        print(
            f"[{epoch_timestamp}] training stage {training_stage}, "
            f"epoch {epoch+1}, train loss = {train_loss:.6f}, "
            f"val loss = {val_loss:.6f}, val_acc = {val_acc:.6f} "
            f"{"SAVED" if improved else ""}"
        )

        with open(epoch_log_file, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch_timestamp,
                training_stage,
                epoch + 1,
                train_loss,
                val_loss,
                val_acc,
                "yes" if improved else "no"
            ])

        if improved:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_file)
        else:
            epochs_no_improve += 1

        history_val_acc.append(val_acc)
        if len(history_val_acc) >= patience:
            window = history_val_acc[-patience:]
            plateau = (max(window) - min(window)) <= training_plateau
            if epochs_no_improve >= patience or plateau:
                break

    return (model, best_val_acc)


def train_model(
    project_dir: str,
    batch_size: int = 64,
    validation_split: float = 0.25,
    device: str = "cpu",
    stage: Optional[int] = None,
    training_plateau: float = 1e-3,
    stage1_epochs: int = 64,
    stage1_lr: float = 1e-3,
    stage1_patience: int = 8,
    stage2_epochs: int = 128,
    stage2_lr: float = 1e-4,
    stage2_patience: int = 16,
    model_path: Optional[str] = None,
) -> Optional[torch.nn.Module]:
    if stage is None:
        try:
            stage = find_latest_stage(project_dir)
            print(f"Using latest stage: {stage}")
        except ValueError as e:
            print(f"Error: No stage specified and {str(e)}")
            return None

    model_file = os.path.join(project_dir, f"stage_{stage}_cull_model.pth")
    epoch_log_file = os.path.join(project_dir, f"stage_{stage}_epoch_log.csv")

    if os.path.exists(model_file):
        try:
            os.remove(model_file)
        except Exception:
            pass

    with open(epoch_log_file, 'w', newline='') as f:
        csv.writer(f).writerow(['datetime', 'training_stage', 'epoch', 'train_loss', 'val_loss', 'val_accuracy', 'saved'])

    print("Initializing ResNet-18 (ImageNet) with single-logit head.")
    model = IMLCullModel(pretrained=True).to(device)

    if model_path and os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

    stage_dir = f"stage_{stage}"
    image_dir_path = os.path.join(project_dir, stage_dir)
    label_file_path = os.path.join(project_dir, f"stage_{stage}_cull_labels.csv")
    if not os.path.exists(image_dir_path):
        raise ValueError(f"Stage directory not found: {image_dir_path}")
    if not os.path.exists(label_file_path):
        raise ValueError(f"Label file not found: {label_file_path}")

    full_dataset = IMLCullDataset(image_dir_path, label_file_path)
    if len(full_dataset) == 0:
        raise ValueError(f"No labeled images found in {image_dir_path}")

    total_size = len(full_dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    gen = torch.Generator().manual_seed(19930625)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=gen)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_acc = 0.0

    print(f"Training stage 1: lr={stage1_lr}, epochs={stage1_epochs}, patience={stage1_patience}")
    model, best_val_acc = run_stage(
        training_stage=1,
        model=model, num_epochs=stage1_epochs, learning_rate=stage1_lr,
        patience=stage1_patience, device=device, training_plateau=training_plateau,
        epoch_log_file=epoch_log_file, model_file=model_file,
        train_loader=train_loader, val_loader=val_loader,
        train_size=train_size, val_size=val_size, best_val_acc=best_val_acc,
        train_backbone=False,
    )

    print(f"Training stage 2: lr={stage2_lr}, epochs={stage2_epochs}, patience={stage2_patience}")
    model, best_val_acc = run_stage(
        training_stage=2,
        model=model, num_epochs=stage2_epochs, learning_rate=stage2_lr,
        patience=stage2_patience, device=device, training_plateau=training_plateau,
        epoch_log_file=epoch_log_file, model_file=model_file,
        train_loader=train_loader, val_loader=val_loader,
        train_size=train_size, val_size=val_size, best_val_acc=best_val_acc,
        train_backbone=True,
    )

    print("Training complete.")
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, type=str)
    parser.add_argument('--stage', type=int)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--validation-split', type=float, default=0.2)
    parser.add_argument('--training-plateau', type=float, default=1e-3)
    parser.add_argument('--stage1-epochs', type=int, default=64)
    parser.add_argument('--stage1-lr', type=float, default=1e-3)
    parser.add_argument('--stage1-patience', type=int, default=8)
    parser.add_argument('--stage2-epochs', type=int, default=128)
    parser.add_argument('--stage2-lr', type=float, default=1e-4)
    parser.add_argument('--stage2-patience', type=int, default=16)
    parser.add_argument("--model", type=str, default=None, help="Path to pretrained .pth model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_model(
        args.project,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        device=device,
        stage=args.stage,
        training_plateau=args.training_plateau,
        stage1_epochs=args.stage1_epochs,
        stage1_lr=args.stage1_lr,
        stage1_patience=args.stage1_patience,
        stage2_epochs=args.stage2_epochs,
        stage2_lr=args.stage2_lr,
        stage2_patience=args.stage2_patience,
        model_path=args.model,
    )

