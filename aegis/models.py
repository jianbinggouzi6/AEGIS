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


class FrequencyAxisAttention(nn.Module):
    """Learn one data-dependent gate for every Mel-frequency bin.

    Average and max descriptors are pooled over channels and time while the
    frequency dimension is preserved. A small 1D convolution lets neighboring
    frequency bins influence each gate without fixing the input Mel count.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("attention kernel_size must be a positive odd number")
        self.gate = nn.Conv1d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def weights(self, x: torch.Tensor) -> torch.Tensor:
        average = x.mean(dim=(1, 3))
        maximum = x.amax(dim=(1, 3))
        descriptor = torch.stack((average, maximum), dim=1)
        return torch.sigmoid(self.gate(descriptor)).unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weights(x)


class AEGISModel(nn.Module):
    """Staged 2D convolutional autoencoder.

    The model accepts any practical log-Mel size. The decoder is resized back
    to the exact input shape, avoiding brittle assumptions about divisibility
    by the three stride-2 encoder blocks. Stage 2 inserts frequency-axis
    attention after the first encoder block. Stage 3 adds a self-supervised
    transformation classifier over globally pooled latent features.
    """

    def __init__(
        self,
        stage: int = 1,
        num_classes: int = 1,
        base_channels: int = 16,
        latent_channels: int = 64,
    ) -> None:
        super().__init__()
        if stage not in (1, 2, 3):
            raise ValueError("stage must be 1, 2, or 3")
        if base_channels < 1 or latent_channels < 1:
            raise ValueError("channel counts must be positive")

        self.stage = stage
        self.num_classes = num_classes
        first_encoder_layers: list[nn.Module] = [ConvBlock(1, base_channels)]
        if stage >= 2:
            first_encoder_layers.append(FrequencyAxisAttention())
        self.encoder = nn.Sequential(
            *first_encoder_layers,
            ConvBlock(base_channels, base_channels * 2),
            ConvBlock(base_channels * 2, latent_channels),
        )
        self.decoder = nn.Sequential(
            DeconvBlock(latent_channels, base_channels * 2),
            DeconvBlock(base_channels * 2, base_channels),
            nn.ConvTranspose2d(base_channels, 1, kernel_size=4, stride=2, padding=1),
        )
        self.classifier = None
        if stage >= 3:
            if num_classes < 2:
                raise ValueError("stage 3 needs at least two self-supervised classes")
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(latent_channels, num_classes),
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"expected [batch, 1, mel, time], got {tuple(x.shape)}")
        target_size = x.shape[-2:]
        latent = self.encoder(x)
        logits = self.classifier(latent) if self.classifier is not None else None
        reconstruction = self.decoder(latent)
        if reconstruction.shape[-2:] != target_size:
            reconstruction = F.interpolate(
                reconstruction,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        return reconstruction, logits
