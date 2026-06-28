from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def dice_bce_loss(seg_logits: torch.Tensor, masks: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(seg_logits, masks)
    probs = torch.sigmoid(seg_logits)
    intersection = (probs * masks).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + smooth) / (denom + smooth)).mean()
    return bce + dice


class SegmentationClassificationHead(nn.Module):
    """A compact image+mask classification head for segmentation backbones."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, images: torch.Tensor, seg_logits: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([images, torch.sigmoid(seg_logits)], dim=1))


class SegClsModel(nn.Module):
    """Wrap any binary segmentation model with a classification head."""

    def __init__(self, segmentation_model: nn.Module, num_classes: int = 3) -> None:
        super().__init__()
        self.segmentation_model = segmentation_model
        self.classification_head = SegmentationClassificationHead(num_classes=num_classes)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seg_logits = self.segmentation_model(images)
        if isinstance(seg_logits, (tuple, list)):
            seg_logits = seg_logits[0]
        cls_logits = self.classification_head(images, seg_logits)
        return seg_logits, cls_logits
