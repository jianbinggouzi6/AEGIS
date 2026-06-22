"""Neural network definitions for the staged AEGIS experiments."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )


class DeconvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )


class AEGISModel(nn.Module):
    """Stage-1 2D convolutional autoencoder.

    The model accepts any practical log-Mel size. The decoder is resized back
    to the exact input shape, avoiding brittle assumptions about divisibility
    by the three stride-2 encoder blocks.
    """

    def __init__(
        self,
        stage: int = 1,
        num_classes: int = 1,
        base_channels: int = 16,
        latent_channels: int = 64,
    ) -> None:
        super().__init__()
        if stage != 1:
            raise ValueError("Stage 1 is implemented in this revision; use stage=1")
        if base_channels < 1 or latent_channels < 1:
            raise ValueError("channel counts must be positive")

        self.stage = stage
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            ConvBlock(1, base_channels),
            ConvBlock(base_channels, base_channels * 2),
            ConvBlock(base_channels * 2, latent_channels),
        )
        self.decoder = nn.Sequential(
            DeconvBlock(latent_channels, base_channels * 2),
            DeconvBlock(base_channels * 2, base_channels),
            nn.ConvTranspose2d(base_channels, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"expected [batch, 1, mel, time], got {tuple(x.shape)}")
        target_size = x.shape[-2:]
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        if reconstruction.shape[-2:] != target_size:
            reconstruction = F.interpolate(
                reconstruction,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        return reconstruction, None

