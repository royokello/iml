from datetime import datetime
import os
import csv
import random
import torch
from torch.utils.data import DataLoader, Subset
from torch import nn, optim
from typing import Dict, List, Tuple, Optional
from image.cull.dataset import IMLCullDataset
from image.cull.model import IMLCullModel
from utils.stages import find_latest_stage
from sklearn.metrics import roc_auc_score


def train_epoch(model, dataloader, criterion, optimizer, dataset_size, device) -> float:
    model.train()
    running_loss = 0.0
    for pixel_values, labels, *_ in dataloader:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()
        logits = model(pixel_values) # (B, 2)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * pixel_values.size(0)
    return running_loss / dataset_size

def classification_metrics(probs, labels) -> Tuple[float, float]:
    preds = [1 if p >= 0.5 else 0 for p in probs]
    recalls: List[float] = []
    f1s: List[float] = []

    for cls in (0, 1):
        tp = sum(1 for p, y in zip(preds, labels) if p == cls and y == cls)
        fp = sum(1 for p, y in zip(preds, labels) if p == cls and y != cls)
        fn = sum(1 for p, y in zip(preds, labels) if p != cls and y == cls)
        support = tp + fn
        if support > 0:
            recalls.append(tp / support)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1s.append((2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0)

    balanced_accuracy = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return balanced_accuracy, macro_f1


def validate_epoch(model, dataloader, criterion, dataset_size, device) -> Tuple[float, float, float, float, float]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for pixel_values, labels, *_ in dataloader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device).long()

            logits = model(pixel_values)
            loss = criterion(logits, labels)
            running_loss += loss.item() * pixel_values.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size
    accuracy = correct / total if total > 0 else 0.0
    auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    balanced_accuracy, macro_f1 = classification_metrics(all_probs, all_labels)

    return avg_loss, accuracy, auroc, balanced_accuracy, macro_f1


_TOL = 1e-6


def _is_better(candidate: Dict, best: Dict) -> bool:
    if candidate["bal_acc"] > best["bal_acc"] + _TOL:
        return True
    if abs(candidate["bal_acc"] - best["bal_acc"]) <= _TOL:
        if candidate["auroc"] > best["auroc"] + _TOL:
            return True
        if abs(candidate["auroc"] - best["auroc"]) <= _TOL and candidate["acc"] > best["acc"] + _TOL:
            return True
    return False


def run_stage(
    training_stage: int,
    model: IMLCullModel,
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
    best_state: Dict,
) -> Tuple[IMLCullModel, Dict]:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    epochs_no_improve = 0
    history_bal_acc = []

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, train_size, device)
        val_loss, val_acc, val_auroc, val_bal_acc, val_f1 = validate_epoch(
            model, val_loader, criterion, val_size, device
        )

        candidate = {
            "bal_acc": val_bal_acc,
            "auroc": val_auroc,
            "acc": val_acc,
            "epoch": epoch + 1,
            "stage": training_stage,
        }
        improved = _is_better(candidate, best_state)

        saved_str = "SAVED" if improved else ""

        epoch_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        print(
            f"[{epoch_timestamp}] training stage {training_stage}, "
            f"epoch {epoch+1}, train loss = {train_loss:.6f}, "
            f"val loss = {val_loss:.6f}, val_acc = {val_acc:.6f}, "
            f"val_bal_acc = {val_bal_acc:.6f}, val_auroc = {val_auroc:.6f}, "
            f"val_f1 = {val_f1:.6f} {saved_str}"
        )

        with open(epoch_log_file, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch_timestamp,
                training_stage,
                epoch + 1,
                train_loss,
                val_loss,
                val_acc,
                val_bal_acc,
                val_auroc,
                val_f1,
                "yes" if improved else "no",
            ])

        if improved:
            best_state = candidate
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_file)
        else:
            epochs_no_improve += 1

        history_bal_acc.append(val_bal_acc)
        patience_2x = patience * 2
        if len(history_bal_acc) >= patience_2x:
            window = history_bal_acc[-patience_2x:]
            plateau = (max(window) - min(window)) <= training_plateau
            if epochs_no_improve >= patience_2x or plateau:
                break

        if epochs_no_improve > patience:
            break

        scheduler.step()

    return model, best_state


def stratified_split_indices(labels: List[int], val_split: float, seed: int) -> Tuple[List[int], List[int]]:
    rnd = random.Random(seed)
    by_class: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    for indices in by_class.values():
        indices = indices[:]
        rnd.shuffle(indices)
        n_val = int(round(val_split * len(indices)))
        if 0.0 < val_split < 1.0 and len(indices) > 1:
            n_val = max(1, min(len(indices) - 1, n_val))
        else:
            n_val = max(0, min(len(indices), n_val))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    rnd.shuffle(train_idx)
    rnd.shuffle(val_idx)
    return train_idx, val_idx


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
        csv.writer(f).writerow([
            'datetime', 'training_stage', 'epoch', 'train_loss', 'val_loss',
            'val_accuracy', 'val_bal_acc', 'val_auroc', 'val_f1', 'saved',
        ])

    print("Initializing model ...")
    model = IMLCullModel().to(device)

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

    sample_labels = [full_dataset.labels[name] for name in full_dataset.images]
    train_indices, val_indices = stratified_split_indices(sample_labels, validation_split, seed=19930625)
    if not val_indices:
        print("Warning: validation split is empty; metrics will be unreliable.")

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    train_size = len(train_indices)
    val_size = len(val_indices)
    print(f"Stratified split: {train_size} train / {val_size} val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_state = {"bal_acc": -1.0, "auroc": -1.0, "acc": -1.0, "epoch": 0, "stage": 0}

    model.freeze_backbone()

    print(f"Training stage 1: lr={stage1_lr}, epochs={stage1_epochs}, patience={stage1_patience}")
    model, best_state = run_stage(
        training_stage=1,
        model=model, num_epochs=stage1_epochs, learning_rate=stage1_lr,
        patience=stage1_patience, device=device, training_plateau=training_plateau,
        epoch_log_file=epoch_log_file, model_file=model_file,
        train_loader=train_loader, val_loader=val_loader,
        train_size=train_size, val_size=val_size, best_state=best_state
    )
    print(f"Best after stage 1: {best_state}")

    model.unfreeze_backbone()

    print(f"Training stage 2: lr={stage2_lr}, epochs={stage2_epochs}, patience={stage2_patience}")
    model, best_state = run_stage(
        training_stage=2,
        model=model, num_epochs=stage2_epochs, learning_rate=stage2_lr,
        patience=stage2_patience, device=device, training_plateau=training_plateau,
        epoch_log_file=epoch_log_file, model_file=model_file,
        train_loader=train_loader, val_loader=val_loader,
        train_size=train_size, val_size=val_size, best_state=best_state
    )
    print(f"Best after stage 2: {best_state}")

    print("Training complete.")
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, type=str)
    parser.add_argument('--stage', type=int)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--validation-split', type=float, default=0.25)
    parser.add_argument('--training-plateau', type=float, default=1e-3)
    parser.add_argument('--stage1-epochs', type=int, default=64)
    parser.add_argument('--stage1-lr', type=float, default=1e-3)
    parser.add_argument('--stage1-patience', type=int, default=3)
    parser.add_argument('--stage2-epochs', type=int, default=128)
    parser.add_argument('--stage2-lr', type=float, default=1e-4)
    parser.add_argument('--stage2-patience', type=int, default=5)
    parser.add_argument("--model", type=str, default=None)
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

