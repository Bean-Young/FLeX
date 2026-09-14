from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FrequencyBands:
    # Paper setting for 224 x 224 inputs: r < 2, 2 <= r < 8, r >= 8.
    alpha_low: float = 2.0 / 224.0
    alpha_mid: float = 8.0 / 224.0


class FrequencyPrompt(nn.Module):
    """Frequency-aware lesion prompts from the FLeX paper.

    The image is decomposed into low, mid, and high Fourier bands. Each band is
    mapped back to image space and receives a bounded residual whose Fourier
    support is restricted to the same band.
    """

    def __init__(
        self,
        image_size: int = 224,
        channels: int = 3,
        bands: FrequencyBands | None = None,
        xi: float = 0.05,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.channels = int(channels)
        self.bands = bands or FrequencyBands()
        self.xi = float(xi)
        self.low = nn.Parameter(torch.zeros(1, channels, image_size, image_size))
        self.mid = nn.Parameter(torch.zeros(1, channels, image_size, image_size))
        self.high = nn.Parameter(torch.zeros(1, channels, image_size, image_size))

    def masks(self, height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        min_dim = min(height, width)
        low_radius = max(1, round(min_dim * self.bands.alpha_low))
        mid_radius = max(low_radius + 1, round(min_dim * self.bands.alpha_mid))
        center_h, center_w = height // 2, width // 2
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        radius = torch.sqrt((yy - center_h).float().square() + (xx - center_w).float().square())
        low = radius < low_radius
        mid = (radius >= low_radius) & (radius < mid_radius)
        high = radius >= mid_radius
        return low.float(), mid.float(), high.float()


def _resize(param: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if param.shape[-2:] == (height, width):
        return param
    return F.interpolate(param, size=(height, width), mode="bilinear", align_corners=False)


def _band_limited_residual(
    param: torch.Tensor,
    mask: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Apply the band mask after tanh so the learned residual stays in-band."""

    spatial = torch.tanh(_resize(param, height, width))
    spectrum = torch.fft.fftshift(torch.fft.fft2(spatial, dim=(-2, -1)), dim=(-2, -1))
    masked = spectrum * mask.view(1, 1, height, width)
    return torch.fft.ifft2(
        torch.fft.ifftshift(masked, dim=(-2, -1)), dim=(-2, -1)
    ).real


def decompose_frequency_bands(
    images: torch.Tensor,
    prompt: FrequencyPrompt,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if images.ndim != 4:
        raise ValueError(f"Expected BCHW images, got {tuple(images.shape)}")
    _batch, _channels, height, width = images.shape
    masks = prompt.masks(height, width, images.device)
    fft = torch.fft.fftshift(torch.fft.fft2(images, dim=(-2, -1)), dim=(-2, -1))
    components = []
    for mask in masks:
        band_fft = fft * mask.view(1, 1, height, width)
        band = torch.fft.ifft2(torch.fft.ifftshift(band_fft, dim=(-2, -1)), dim=(-2, -1)).real
        components.append(band)
    return tuple(components)  # type: ignore[return-value]


def apply_frequency_prompt(
    images: torch.Tensor,
    prompt: FrequencyPrompt,
    weights: torch.Tensor | tuple[float, float, float] | None = None,
    mode: str = "bands",
) -> torch.Tensor:
    """Build the lesion-frequency representation in Eq. (11)."""

    batch, _channels, height, width = images.shape
    if mode == "full":
        # Full-band ablation: one prompt acts on the undecomposed image.
        residual = _resize(prompt.low, height, width)
        return images + prompt.xi * torch.tanh(residual)
    if mode != "bands":
        raise ValueError(f"Unknown prompt mode: {mode!r}")
    if weights is None:
        weights = images.new_tensor((0.25, 0.50, 0.25)).view(1, 3).repeat(batch, 1)
    elif not torch.is_tensor(weights):
        weights = images.new_tensor(weights).view(1, 3).repeat(batch, 1)
    else:
        weights = weights.to(device=images.device, dtype=images.dtype)
        if weights.ndim == 1:
            weights = weights.view(1, 3).repeat(batch, 1)

    masks = prompt.masks(height, width, images.device)
    bands = decompose_frequency_bands(images, prompt)
    params = (
        _resize(prompt.low, height, width),
        _resize(prompt.mid, height, width),
        _resize(prompt.high, height, width),
    )
    output = torch.zeros_like(images)
    for band_idx, (band, param, mask) in enumerate(zip(bands, params, masks)):
        residual = _band_limited_residual(param, mask, height, width)
        prompted = band + prompt.xi * residual
        output = output + weights[:, band_idx].view(batch, 1, 1, 1) * prompted
    return output
