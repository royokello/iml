import torch
import torch.nn as nn
from transformers import DeiTModel


class IMLRankModel(nn.Module):
    def __init__(self, backbone_name: str = "facebook/deit-base-distilled-patch16-384", dropout: float = 0.0):
        super(IMLRankModel, self).__init__()
        self.backbone = DeiTModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.score_head = nn.Linear(hidden_size, 1)

    def forward(self, pixel_values, return_features: bool = False):
        outputs = self.backbone(pixel_values=pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        scores = self.score_head(pooled_output).squeeze(-1)  # [B]
        if return_features:
            return scores, pooled_output
        return scores

    def forward_pair(self, pixel_values_a, pixel_values_b, return_diff: bool = True):
        score_a = self.forward(pixel_values_a)
        score_b = self.forward(pixel_values_b)
        if return_diff:
            return score_a, score_b, score_a - score_b
        return score_a, score_b

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
