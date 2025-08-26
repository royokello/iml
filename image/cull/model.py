import torch
import torch.nn as nn
from transformers import DeiTModel, DeiTConfig

class IMLCullModel(nn.Module):
    def __init__(self):
        super(IMLCullModel, self).__init__()
        self.backbone = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-384")
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        logits = self.classifier(pooled_output)
        return logits  # shape [batch_size, 2]

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
