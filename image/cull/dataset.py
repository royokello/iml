import os
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class IMLCullDataset(Dataset):
    def __init__(self, image_dir_path, label_file_path):
        self.image_dir = image_dir_path
        self.labels = {}

        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        if not os.path.exists(image_dir_path):
            raise ValueError(f"Image directory not found: {image_dir_path}")
        if not os.path.exists(label_file_path):
            raise ValueError(f"Label file not found: {label_file_path}")

        with open(label_file_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    img_path_csv = row["image_path"]
                    lbl = int(row["label"])
                    if lbl not in (0, 1):
                        continue
                    self.labels[img_path_csv] = lbl
                except Exception:
                    continue

        self.images = list(self.labels.keys())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path_csv = self.images[idx]
        img_path = img_path_csv if (os.path.isabs(img_path_csv) and os.path.exists(img_path_csv)) else os.path.join(self.image_dir, img_path_csv)

        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)
        label = torch.tensor(self.labels[self.images[idx]], dtype=torch.int64)

        return pixel_values, label, img_path
