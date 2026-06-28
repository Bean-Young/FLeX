from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FrequencyBands:
    low_radius_ratio: float = 0.005
    mid_radius_multiplier: float = 2.0


class FrequencyPrompt(nn.Module):
    """Low/mid/high amplitude prompts for ultrasound frequency adaptation."""

    def __init__(
        self,
        image_size: int = 224,
        bands: FrequencyBands | None = None,
        init_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.bands = bands or FrequencyBands()
        self.low = nn.Parameter(torch.full((1, 1, image_size, image_size), init_scale))
        self.mid = nn.Parameter(torch.full((1, 1, image_size, image_size), init_scale))
        self.high = nn.Parameter(torch.full((1, 1, image_size, image_size), init_scale))

    def masks(self, height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        min_dim = min(height, width)
        low_radius = max(1, int(min_dim * self.bands.low_radius_ratio))
        mid_radius = max(low_radius + 1, int(low_radius * self.bands.mid_radius_multiplier))
        center_h, center_w = height // 2, width // 2
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        dist = torch.sqrt((yy - center_h).float().square() + (xx - center_w).float().square())
        low = dist <= low_radius
        mid = (dist > low_radius) & (dist <= mid_radius)
        high = dist > mid_radius
        return low.float(), mid.float(), high.float()


def _resize_prompt(prompt: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if prompt.shape[-2:] == (height, width):
        return prompt
    return F.interpolate(prompt, size=(height, width), mode="bilinear", align_corners=False)


def apply_frequency_prompt(
    images: torch.Tensor,
    prompt: FrequencyPrompt,
    weights: torch.Tensor | tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """Apply learnable amplitude modulation in the Fourier domain."""

    if images.ndim != 4:
        raise ValueError(f"Expected BCHW images, got shape {tuple(images.shape)}")
    batch, _channels, height, width = images.shape
    if weights is None:
        weights = images.new_tensor((0.25, 0.50, 0.25)).view(1, 3).repeat(batch, 1)
    elif not torch.is_tensor(weights):
        weights = images.new_tensor(weights).view(1, 3).repeat(batch, 1)
    else:
        weights = weights.to(device=images.device, dtype=images.dtype)
        if weights.ndim == 1:
            weights = weights.view(1, 3).repeat(batch, 1)

    masks = prompt.masks(height, width, images.device)
    params = (
        _resize_prompt(prompt.low, height, width),
        _resize_prompt(prompt.mid, height, width),
        _resize_prompt(prompt.high, height, width),
    )

    fft = torch.fft.fft2(images, dim=(-2, -1))
    amplitude = torch.fft.fftshift(torch.abs(fft))
    phase = torch.angle(fft)
    log_multiplier = torch.zeros_like(amplitude)
    for band_idx, (mask, param) in enumerate(zip(masks, params)):
        band_weight = weights[:, band_idx].view(batch, 1, 1, 1)
        log_multiplier = log_multiplier + band_weight * mask.view(1, 1, height, width) * param

    modified_amplitude = torch.fft.ifftshift(amplitude * torch.exp(log_multiplier.clamp(-0.25, 0.25)))
    real = torch.cos(phase) * modified_amplitude
    imag = torch.sin(phase) * modified_amplitude
    return torch.fft.ifft2(torch.complex(real=real, imag=imag), dim=(-2, -1)).real
