from datetime import datetime
import os
import csv
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch import nn, optim
from typing import Tuple, Optional
from PIL import Image
from torchvision import transforms
from image.rank.model import IMLRankModel
from utils.stages import find_latest_stage
from sklearn.metrics import roc_auc_score


def resolve_image_path(image_dir: str, image_id: str) -> Optional[str]:
    if not image_id:
        return None
    if os.path.isabs(image_id) and os.path.exists(image_id):
        return image_id

    direct_path = os.path.join(image_dir, image_id)
    if os.path.exists(direct_path):
        return direct_path

    name, ext = os.path.splitext(image_id)
    if ext:
        return None

    for e in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        p = os.path.join(image_dir, f"{image_id}{e}")
        if os.path.exists(p):
            return p
    return None


class IMLRankDataset(Dataset):
    def __init__(self, image_dir_path: str, label_file_path: str):
        self.image_dir = image_dir_path
        self.pairs = []

        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        if not os.path.exists(image_dir_path):
            raise ValueError(f"Image directory not found: {image_dir_path}")
        if not os.path.exists(label_file_path):
            raise ValueError(f"Label file not found: {label_file_path}")

        with open(label_file_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "winner" in row and "loser" in row:
                    left_id = row.get("winner", "").strip()
                    right_id = row.get("loser", "").strip()
                    if not left_id or not right_id:
                        continue
                    self.pairs.append((left_id, right_id, 1, row.get("label", "")))
                    continue

                img_1 = row.get("img_1", "").strip()
                img_2 = row.get("img_2", "").strip()
                pref = row.get("preference", "").strip().lower()
                if not img_1 or not img_2:
                    continue
                if pref not in ("left", "right"):
                    continue
                label = 1 if pref == "left" else 0
                self.pairs.append((img_1, img_2, label, row.get("label", "")))

        if len(self.pairs) == 0:
            raise ValueError(f"No ranked pairs found in {label_file_path}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        left_id, right_id, label, reason = self.pairs[idx]
        left_path = resolve_image_path(self.image_dir, left_id)
        right_path = resolve_image_path(self.image_dir, right_id)
        if left_path is None or right_path is None:
            raise ValueError(f"Missing image for pair: {left_id}, {right_id}")

        left_img = Image.open(left_path).convert("RGB")
        right_img = Image.open(right_path).convert("RGB")
        left_tensor = self.transform(left_img)
        right_tensor = self.transform(right_img)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return left_tensor, right_tensor, label_tensor, left_path, right_path, reason


def train_epoch(model, dataloader, criterion, optimizer, dataset_size, device) -> float:
    model.train()
    running_loss = 0.0
    for left, right, labels, *_ in dataloader:
        left = left.to(device)
        right = right.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()
        score_left = model(left)
        score_right = model(right)
        logits = score_left - score_right
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * left.size(0)
    return running_loss / dataset_size


def validate_epoch(model, dataloader, criterion, dataset_size, device) -> Tuple[float, float, float]:
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for left, right, labels, *_ in dataloader:
            left = left.to(device)
            right = right.to(device)
            labels = labels.to(device).float()

            score_left = model(left)
            score_right = model(right)
            logits = score_left - score_right
            loss = criterion(logits, labels)
            running_loss += loss.item() * left.size(0)

            preds = (logits >= 0).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / dataset_size
    accuracy = correct / total if total > 0 else 0.0
    auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5

    return avg_loss, accuracy, auroc


def run_stage(
    training_stage: int,
    model: IMLRankModel,
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
    best_val_auroc: float,
    best_val_acc: float,
) -> Tuple[IMLRankModel, float, float]:
    criterion = nn.BCEWithLogitsLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    epochs_no_improve = 0
    history_val_auroc = []

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, train_size, device)
        val_loss, val_acc, val_auroc = validate_epoch(model, val_loader, criterion, val_size, device)

        if val_auroc > best_val_auroc:
            improved = True
            if abs(val_auroc - best_val_auroc) < 1e-6 and val_acc < best_val_acc:
                improved = False
        else:
            improved = False

        saved_str = "SAVED" if improved else ""

        epoch_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        print(
            f"[{epoch_timestamp}] training stage {training_stage}, "
            f"epoch {epoch+1}, train loss = {train_loss:.6f}, "
            f"val loss = {val_loss:.6f}, val_acc = {val_acc:.6f}, val_auroc = {val_auroc:.6f} {saved_str}"
        )

        with open(epoch_log_file, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch_timestamp,
                training_stage,
                epoch + 1,
                train_loss,
                val_loss,
                val_acc,
                val_auroc,
                "yes" if improved else "no",
            ])

        if improved:
            best_val_auroc = val_auroc
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_file)
        else:
            epochs_no_improve += 1

        history_val_auroc.append(val_auroc)
        patience_2x = patience * 2
        if len(history_val_auroc) >= patience_2x:
            window = history_val_auroc[-patience_2x:]
            plateau = (max(window) - min(window)) <= training_plateau
            if epochs_no_improve >= patience_2x or plateau:
                break

        if epochs_no_improve > patience:
            break

        scheduler.step()

    return (model, best_val_auroc, best_val_acc)


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

    model_file = os.path.join(project_dir, f"stage_{stage}_rank_model.pth")
    epoch_log_file = os.path.join(project_dir, f"stage_{stage}_rank_epoch_log.csv")

    if os.path.exists(model_file):
        try:
            os.remove(model_file)
        except Exception:
            pass

    with open(epoch_log_file, 'w', newline='') as f:
        csv.writer(f).writerow(['datetime', 'training_stage', 'epoch', 'train_loss', 'val_loss', 'val_accuracy', 'val_auroc', 'saved'])

    print("Initializing model ...")
    model = IMLRankModel().to(device)

    if model_path and os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

    stage_dir = f"stage_{stage}"
    image_dir_path = os.path.join(project_dir, stage_dir)
    label_file_path = os.path.join(project_dir, f"stage_{stage}_rank_labels.csv")
    if not os.path.exists(image_dir_path):
        raise ValueError(f"Stage directory not found: {image_dir_path}")
    if not os.path.exists(label_file_path):
        raise ValueError(f"Label file not found: {label_file_path}")

    full_dataset = IMLRankDataset(image_dir_path, label_file_path)
    total_size = len(full_dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    gen = torch.Generator().manual_seed(19930625)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=gen)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_auroc = -1.0
    best_val_acc = -1.0

    model.freeze_backbone()

    print(f"Training stage 1: lr={stage1_lr}, epochs={stage1_epochs}, patience={stage1_patience}")
    model, best_val_auroc, best_val_acc = run_stage(
        training_stage=1,
        model=model, num_epochs=stage1_epochs, learning_rate=stage1_lr,
        patience=stage1_patience, device=device, training_plateau=training_plateau,
        epoch_log_file=epoch_log_file, model_file=model_file,
        train_loader=train_loader, val_loader=val_loader,
        train_size=train_size, val_size=val_size, best_val_auroc=best_val_auroc, best_val_acc=best_val_acc
    )

    model.unfreeze_backbone()

    print(f"Training stage 2: lr={stage2_lr}, epochs={stage2_epochs}, patience={stage2_patience}")
    model, best_val_auroc, best_val_acc = run_stage(
        training_stage=2,
        model=model, num_epochs=stage2_epochs, learning_rate=stage2_lr,
        patience=stage2_patience, device=device, training_plateau=training_plateau,
        epoch_log_file=epoch_log_file, model_file=model_file,
        train_loader=train_loader, val_loader=val_loader,
        train_size=train_size, val_size=val_size, best_val_auroc=best_val_auroc, best_val_acc=best_val_acc
    )

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
    parser.add_argument('--stage2-lr', type=int, default=1e-4)
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
