from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .frequency import FrequencyPrompt, apply_frequency_prompt


@dataclass
class FlexConfig:
    benign_index: int = 0
    malignant_index: int = 1
    normal_index: int = 2
    lesion_topk_ratio: float = 0.01
    frequency_weights: tuple[float, float, float] = (0.25, 0.50, 0.25)
    rho_seg: float = 0.50
    rho_cls: float = 0.50
    lr: float = 1e-4
    steps: int = 1
    lambda_sub: float = 0.10
    lambda_src: float = 0.10
    lambda_pres: float = 0.50
    lambda_neg: float = 0.50
    lambda_prompt: float = 1e-3
    source_fg_threshold: float = 0.50
    source_bg_threshold: float = 0.10


def binary_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    student = torch.sigmoid(student_logits).clamp(1e-6, 1 - 1e-6)
    teacher = torch.sigmoid(teacher_logits).clamp(1e-6, 1 - 1e-6)
    return teacher * (teacher.log() - student.log()) + (1 - teacher) * (
        (1 - teacher).log() - (1 - student).log()
    )


def lesion_score_from_mask(seg_logits: torch.Tensor, topk_ratio: float = 0.01) -> torch.Tensor:
    probs = torch.sigmoid(seg_logits).flatten(1)
    topk = max(1, int(probs.size(1) * topk_ratio))
    return probs.topk(topk, dim=1).values.mean(1).clamp(0.0, 1.0)


def lesion_first_probabilities(cls_logits: torch.Tensor, seg_logits: torch.Tensor, cfg: FlexConfig) -> torch.Tensor:
    """Convert lesion evidence and benign/malignant logits into three-class probabilities."""

    lesion_score = lesion_score_from_mask(seg_logits, cfg.lesion_topk_ratio)
    subtype_logits = torch.stack(
        [cls_logits[:, cfg.benign_index], cls_logits[:, cfg.malignant_index]],
        dim=1,
    )
    subtype_probs = subtype_logits.softmax(dim=1)
    probs = torch.zeros_like(cls_logits.softmax(dim=1))
    probs[:, cfg.normal_index] = 1.0 - lesion_score
    probs[:, cfg.benign_index] = lesion_score * subtype_probs[:, 0]
    probs[:, cfg.malignant_index] = lesion_score * subtype_probs[:, 1]
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)


def flex_objective(
    adapted_seg_logits: torch.Tensor,
    adapted_cls_logits: torch.Tensor,
    source_seg_logits: torch.Tensor,
    prompt: FrequencyPrompt,
    cfg: FlexConfig,
) -> torch.Tensor:
    """Unsupervised FLeX objective used during test-time prompt adaptation."""

    subtype_logits = torch.stack(
        [adapted_cls_logits[:, cfg.benign_index], adapted_cls_logits[:, cfg.malignant_index]],
        dim=1,
    )
    subtype_probs = subtype_logits.softmax(dim=1)
    loss_sub = -(subtype_probs * subtype_probs.clamp_min(1e-8).log()).sum(1).mean()

    pred_prob = torch.sigmoid(adapted_seg_logits)
    src_prob = torch.sigmoid(source_seg_logits).detach()
    loss_src = binary_kl(adapted_seg_logits, source_seg_logits.detach()).mean()

    src_fg = (src_prob > cfg.source_fg_threshold).float()
    fg_denom = src_fg.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
    loss_pres = (F.relu(src_prob - pred_prob).square() * src_fg).sum(dim=(1, 2, 3), keepdim=True)
    loss_pres = (loss_pres / fg_denom).mean()

    src_bg = (src_prob < cfg.source_bg_threshold).float()
    bg_denom = src_bg.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
    loss_neg = (pred_prob.square() * src_bg).sum(dim=(1, 2, 3), keepdim=True)
    loss_neg = (loss_neg / bg_denom).mean()

    prompt_reg = sum(param.square().mean() for param in prompt.parameters())
    return (
        cfg.lambda_sub * loss_sub
        + cfg.lambda_src * loss_src
        + cfg.lambda_pres * loss_pres
        + cfg.lambda_neg * loss_neg
        + cfg.lambda_prompt * prompt_reg
    )


class FlexAdapter:
    """Model-agnostic FLeX test-time adapter.

    The wrapped model must return `(segmentation_logits, classification_logits)`.
    Backbone weights are frozen; only frequency prompts are updated at test time.
    """

    def __init__(
        self,
        model: nn.Module,
        image_size: int = 224,
        cfg: FlexConfig | None = None,
    ) -> None:
        self.model = model
        self.source_model = copy.deepcopy(model).eval()
        self.prompt = FrequencyPrompt(image_size=image_size)
        self.cfg = cfg or FlexConfig()
        for param in self.model.parameters():
            param.requires_grad_(False)
        for param in self.source_model.parameters():
            param.requires_grad_(False)

    def to(self, device: torch.device | str) -> "FlexAdapter":
        self.model.to(device)
        self.source_model.to(device)
        self.prompt.to(device)
        return self

    def adapt(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        self.model.eval()
        self.source_model.eval()
        self.prompt.train()
        optimizer = torch.optim.Adam(self.prompt.parameters(), lr=self.cfg.lr)
        source_seg_logits, _source_cls_logits = self.source_model(images)

        adapted_seg_logits = adapted_cls_logits = None
        for _ in range(self.cfg.steps):
            optimizer.zero_grad(set_to_none=True)
            prompted = apply_frequency_prompt(images, self.prompt, self.cfg.frequency_weights)
            adapted_seg_logits, adapted_cls_logits = self.model(prompted)
            loss = flex_objective(adapted_seg_logits, adapted_cls_logits, source_seg_logits, self.prompt, self.cfg)
            loss.backward()
            optimizer.step()

        if adapted_seg_logits is None or adapted_cls_logits is None:
            prompted = apply_frequency_prompt(images, self.prompt, self.cfg.frequency_weights)
            adapted_seg_logits, adapted_cls_logits = self.model(prompted)
        return {
            "source_seg_logits": source_seg_logits,
            "adapted_seg_logits": adapted_seg_logits,
            "adapted_cls_logits": adapted_cls_logits,
        }

    def predict(self, images: torch.Tensor, adapt: bool = True) -> dict[str, torch.Tensor]:
        self.model.eval()
        self.source_model.eval()
        with torch.no_grad():
            source_seg_logits, source_cls_logits = self.source_model(images)
        if adapt:
            outputs = self.adapt(images)
            adapted_seg_logits = outputs["adapted_seg_logits"]
            adapted_cls_logits = outputs["adapted_cls_logits"]
        else:
            adapted_seg_logits, adapted_cls_logits = source_seg_logits, source_cls_logits

        with torch.no_grad():
            source_cls_probs = lesion_first_probabilities(source_cls_logits, source_seg_logits, self.cfg)
            adapted_cls_probs = lesion_first_probabilities(adapted_cls_logits, adapted_seg_logits, self.cfg)
            source_seg_probs = torch.sigmoid(source_seg_logits)
            adapted_seg_probs = torch.sigmoid(adapted_seg_logits)
            seg_probs = source_seg_probs + self.cfg.rho_seg * (adapted_seg_probs - source_seg_probs)
            cls_probs = source_cls_probs + self.cfg.rho_cls * (adapted_cls_probs - source_cls_probs)
            cls_probs = cls_probs.clamp_min(1e-8)
            cls_probs = cls_probs / cls_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return {
            "seg_probs": seg_probs,
            "cls_probs": cls_probs,
            "source_seg_probs": source_seg_probs,
            "adapted_seg_probs": adapted_seg_probs,
        }
