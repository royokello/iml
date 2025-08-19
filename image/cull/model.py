import torch
import torch.nn as nn
from torchvision.models import resnet18

class IMLCullModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)  # single logit head

    def forward(self, x):
        return self.backbone(x).squeeze(1)  # (B,) logits
