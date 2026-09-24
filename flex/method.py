from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .frequency import FrequencyBands, FrequencyPrompt, apply_frequency_prompt

try:
    from scipy import ndimage
except Exception:  # pragma: no cover
    ndimage = None


@dataclass
class FlexConfig:
    benign_index: int = 0
    malignant_index: int = 1
    normal_index: int = 2
    kappa: float = 1.0
    eta_min: float = 0.50
    eta_max: float = 0.95
    delta: float = 0.05
    frequency_weights: tuple[float, float, float] = (0.25, 0.50, 0.25)
    frequency_alpha_low: float = 2.0 / 224.0
    frequency_alpha_mid: float = 8.0 / 224.0
    prompt_amplitude: float = 0.05
    prompt_mode: str = "bands"
    rho_seg: float = 0.50
    rho_cls: float = 0.50
    reliability_fusion: bool = True
    lr: float = 1e-4
    steps: int = 1
    lambda_src: float = 0.10
    lambda_pres: float = 0.50
    lambda_neg: float = 0.50
    beta_area: float = 1.0
    mask_threshold: float = 0.40
    morphology_refinement: bool = True
    min_component_area_ratio: float = 5e-4


def adaptive_threshold(prob: torch.Tensor, cfg: FlexConfig) -> torch.Tensor:
    flat = prob.flatten(1)
    mean = flat.mean(dim=1)
    std = flat.std(dim=1, unbiased=False)
    return (mean + cfg.kappa * std).clamp(cfg.eta_min, cfg.eta_max)


def lesion_support(prob: torch.Tensor, cfg: FlexConfig) -> tuple[torch.Tensor, torch.Tensor]:
    eta = adaptive_threshold(prob, cfg)
    support = prob >= eta.view(-1, 1, 1, 1)
    return support, eta


def lesion_score(prob: torch.Tensor, cfg: FlexConfig) -> torch.Tensor:
    support, _eta = lesion_support(prob, cfg)
    flat_prob = prob.flatten(1)
    flat_support = support.flatten(1)
    support_count = flat_support.sum(dim=1)
    support_sum = (flat_prob * flat_support.float()).sum(dim=1)
    fallback = flat_prob.max(dim=1).values
    return torch.where(support_count > 0, support_sum / support_count.clamp_min(1.0), fallback).clamp(0.0, 1.0)


def lesion_first_probabilities(cls_logits: torch.Tensor, seg_logits: torch.Tensor, cfg: FlexConfig) -> torch.Tensor:
    """Eq. (4): Normal is absence of lesion evidence; Benign/Malignant share lesion mass."""

    prob = torch.sigmoid(seg_logits)
    score = lesion_score(prob, cfg)
    subtype_logits = torch.stack(
        [cls_logits[:, cfg.benign_index], cls_logits[:, cfg.malignant_index]],
        dim=1,
    )
    subtype_probs = subtype_logits.softmax(dim=1)
    probs = torch.zeros_like(cls_logits.softmax(dim=1))
    probs[:, cfg.normal_index] = 1.0 - score
    probs[:, cfg.benign_index] = score * subtype_probs[:, 0]
    probs[:, cfg.malignant_index] = score * subtype_probs[:, 1]
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)


def source_supported_weight(source_seg_logits: torch.Tensor, source_cls_logits: torch.Tensor, cfg: FlexConfig) -> torch.Tensor:
    p0 = torch.sigmoid(source_seg_logits)
    q0 = source_cls_logits.softmax(dim=1)
    return (lesion_score(p0, cfg) * torch.maximum(1.0 - q0[:, cfg.normal_index], q0.new_full(q0[:, 0].shape, cfg.delta))).detach()


def flex_objective(
    adapted_seg_logits: torch.Tensor,
    adapted_cls_logits: torch.Tensor,
    source_seg_logits: torch.Tensor,
    source_cls_logits: torch.Tensor,
    cfg: FlexConfig,
) -> torch.Tensor:
    """Eq. (15)-(20): constrained lesion-aware prompt objective."""

    pt = torch.sigmoid(adapted_seg_logits)
    p0 = torch.sigmoid(source_seg_logits).detach()
    alpha0 = source_supported_weight(source_seg_logits, source_cls_logits, cfg)
    subtype_logits = torch.stack(
        [adapted_cls_logits[:, cfg.benign_index], adapted_cls_logits[:, cfg.malignant_index]],
        dim=1,
    )
    subtype_probs = subtype_logits.softmax(dim=1)
    subtype_entropy = -(subtype_probs * subtype_probs.clamp_min(1e-8).log()).sum(dim=1)
    loss_sub = (alpha0 * subtype_entropy).mean()

    loss_src_map = (pt - p0).abs().flatten(1).mean(dim=1)
    area_t = pt.flatten(1).mean(dim=1)
    area_0 = p0.flatten(1).mean(dim=1)
    loss_src = (loss_src_map + cfg.beta_area * (area_t - area_0).abs()).mean()

    support0, eta0 = lesion_support(p0, cfg)
    background0 = p0 < eta0.view(-1, 1, 1, 1)
    support_count = support0.flatten(1).sum(dim=1).clamp_min(1.0)
    background_count = background0.flatten(1).sum(dim=1).clamp_min(1.0)

    loss_pres = (F.relu(p0 - pt) * support0.float()).flatten(1).sum(dim=1)
    loss_pres = (alpha0 * loss_pres / support_count).mean()
    loss_neg = (F.relu(pt - eta0.view(-1, 1, 1, 1)) * background0.float()).flatten(1).sum(dim=1)
    loss_neg = (loss_neg / background_count).mean()
    return loss_sub + cfg.lambda_src * loss_src + cfg.lambda_pres * loss_pres + cfg.lambda_neg * loss_neg


def refine_mask(mask: np.ndarray, cfg: FlexConfig) -> np.ndarray:
    if not cfg.morphology_refinement:
        return mask.astype(bool)
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return mask
    min_area = max(1, int(mask.size * cfg.min_component_area_ratio))
    if ndimage is None:
        return mask if mask.sum() >= min_area else np.zeros_like(mask, dtype=bool)
    filled = ndimage.binary_fill_holes(mask)
    labels, num = ndimage.label(filled)
    if num == 0:
        return filled
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    keep = int(areas.argmax())
    if areas[keep] < min_area:
        return np.zeros_like(mask, dtype=bool)
    return labels == keep


class FlexAdapter:
    """FLeX adapter matching the paper method.

    The wrapped model must return `(segmentation_logits, classification_logits)`.
    Source model weights are frozen; only low/mid/high prompt parameters are
    updated on unlabeled target images.
    """

    def __init__(
        self,
        model: nn.Module,
        image_size: int = 224,
        cfg: FlexConfig | None = None,
    ) -> None:
        self.model = model
        self.source_model = copy.deepcopy(model).eval()
        self.cfg = cfg or FlexConfig()
        self.prompt = FrequencyPrompt(
            image_size=image_size,
            xi=self.cfg.prompt_amplitude,
            bands=FrequencyBands(
                alpha_low=self.cfg.frequency_alpha_low,
                alpha_mid=self.cfg.frequency_alpha_mid,
            ),
        )
        self.optimizer = torch.optim.Adam(self.prompt.parameters(), lr=self.cfg.lr)
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
        with torch.no_grad():
            source_seg_logits, source_cls_logits = self.source_model(images)

        adapted_seg_logits = adapted_cls_logits = None
        for _ in range(self.cfg.steps):
            self.optimizer.zero_grad(set_to_none=True)
            prompted = apply_frequency_prompt(
                images, self.prompt, self.cfg.frequency_weights, self.cfg.prompt_mode
            )
            adapted_seg_logits, adapted_cls_logits = self.model(prompted)
            loss = flex_objective(
                adapted_seg_logits,
                adapted_cls_logits,
                source_seg_logits,
                source_cls_logits,
                self.cfg,
            )
            loss.backward()
            self.optimizer.step()

        # Predict again after the final update so the returned output uses the
        # prompt fitted to the current batch.
        with torch.no_grad():
            prompted = apply_frequency_prompt(
                images, self.prompt, self.cfg.frequency_weights, self.cfg.prompt_mode
            )
            adapted_seg_logits, adapted_cls_logits = self.model(prompted)
        return {
            "source_seg_logits": source_seg_logits,
            "source_cls_logits": source_cls_logits,
            "adapted_seg_logits": adapted_seg_logits,
            "adapted_cls_logits": adapted_cls_logits,
        }

    def reset_adaptation(self) -> None:
        """Reset prompts and Adam state before an independent TTA transfer."""

        with torch.no_grad():
            for parameter in self.prompt.parameters():
                parameter.zero_()
        self.optimizer = torch.optim.Adam(self.prompt.parameters(), lr=self.cfg.lr)

    def predict(
        self,
        images: torch.Tensor,
        adapt: bool = True,
        fixed_frequency: bool = False,
        source_hierarchy: bool = False,
    ) -> dict[str, torch.Tensor]:
        if adapt and (fixed_frequency or source_hierarchy):
            raise ValueError("Choose only one prediction mode")
        if fixed_frequency and source_hierarchy:
            raise ValueError("Choose only one prediction mode")
        self.model.eval()
        self.source_model.eval()
        with torch.no_grad():
            source_seg_logits, source_cls_logits = self.source_model(images)
        if adapt:
            outputs = self.adapt(images)
            adapted_seg_logits = outputs["adapted_seg_logits"]
            adapted_cls_logits = outputs["adapted_cls_logits"]
        elif fixed_frequency:
            # The zero-prompt control keeps the weighted frequency image but
            # performs no parameter or optimizer update.
            with torch.no_grad():
                weighted_image = apply_frequency_prompt(
                    images, self.prompt, self.cfg.frequency_weights,
                    self.cfg.prompt_mode, zero_prompt=True,
                )
                adapted_seg_logits, adapted_cls_logits = self.model(weighted_image)
        else:
            adapted_seg_logits, adapted_cls_logits = source_seg_logits, source_cls_logits

        if not (adapt or fixed_frequency or source_hierarchy):
            # Paper No Adapt is a frozen source forward pass, without FLeX's
            # lesion hierarchy, fusion, or morphology.
            with torch.no_grad():
                source_prob = torch.sigmoid(source_seg_logits)
                raw_class_prob = source_cls_logits.softmax(dim=1)
            return {
                "seg_probs": source_prob,
                "cls_probs": raw_class_prob,
                "source_seg_probs": source_prob,
                "adapted_seg_probs": source_prob,
                "reliability": torch.zeros(images.shape[0], device=images.device),
            }

        with torch.no_grad():
            p0 = torch.sigmoid(source_seg_logits)
            pt = torch.sigmoid(adapted_seg_logits)
            reliability = source_supported_weight(
                source_seg_logits, source_cls_logits, self.cfg
            ).clamp(0.0, 1.0)
            p0_cls = lesion_first_probabilities(source_cls_logits, source_seg_logits, self.cfg)
            pt_cls = lesion_first_probabilities(adapted_cls_logits, adapted_seg_logits, self.cfg)
            if self.cfg.reliability_fusion:
                seg_gate = (self.cfg.rho_seg * reliability).view(-1, 1, 1, 1)
                cls_gate = (self.cfg.rho_cls * reliability).view(-1, 1)
                p_final = (1.0 - seg_gate) * p0 + seg_gate * pt
                cls_probs = (1.0 - cls_gate) * p0_cls + cls_gate * pt_cls
            else:
                p_final = pt
                cls_probs = pt_cls
            cls_probs = cls_probs / cls_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return {
            "seg_probs": p_final,
            "cls_probs": cls_probs,
            "source_seg_probs": p0,
            "adapted_seg_probs": pt,
            "reliability": reliability,
        }
