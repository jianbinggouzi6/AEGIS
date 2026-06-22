"""Neural network definitions for AEGIS.

AEGISModel is controlled by three boolean flags matching the ablation table:
  use_freq_attention   — frequency-axis attention after the first encoder block
  use_classifier       — section-ID classification head (CE or ArcFace)

The model always uses a 2D log-Mel input [B, 1, n_mels, frames].
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DeconvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class FreqAxisAttention(nn.Module):
    """Per-frequency-bin gate, pooled over channel and time dimensions.

    Input  x : [B, C, F, T]
    Output   : [B, C, F, T]  (element-wise multiplied by per-bin scalar gate)
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.gate = nn.Conv1d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=(1, 3))            # [B, F]
        mx  = x.amax(dim=(1, 3))            # [B, F]
        desc    = torch.stack((avg, mx), dim=1)                 # [B, 2, F]
        weights = torch.sigmoid(self.gate(desc)).unsqueeze(-1)  # [B, 1, F, 1]
        return x * weights


# ---------------------------------------------------------------------------
# ArcFace classification head
# ---------------------------------------------------------------------------

class ArcFaceHead(nn.Module):
    """Angular-margin softmax (ArcFace) classification head.

    At training time (labels provided) the angular margin is applied.
    At inference (labels=None or eval mode) vanilla cosine-similarity
    logits are returned — softmax over these gives the confidence used
    for the anomaly score.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("ArcFaceHead requires num_classes >= 2")
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normed_x = F.normalize(features, p=2, dim=1)
        normed_w = F.normalize(self.weight,  p=2, dim=1)
        cos_theta = torch.mm(normed_x, normed_w.T).clamp(-1 + 1e-7, 1 - 1e-7)
        if labels is None or not self.training:
            return self.scale * cos_theta
        theta         = torch.acos(cos_theta)
        target_logits = torch.cos(theta + self.margin)
        one_hot       = F.one_hot(labels, self.num_classes).float()
        logits        = self.scale * (one_hot * target_logits + (1 - one_hot) * cos_theta)
        return logits


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class AEGISModel(nn.Module):
    """2D Conv-AE with optional frequency-axis attention and section classifier.

    Flags
    -----
    use_freq_attention : insert FreqAxisAttention after the first encoder block
    use_classifier     : attach a section-ID head over the pooled latent space
    num_classes        : number of sections; must be >= 2 when use_classifier=True
    loss_type          : "ce" (cross-entropy) or "arcface"
    base_channels      : channel width of the first encoder block
    latent_channels    : channel width of the bottleneck (deepest) encoder block
    """

    def __init__(
        self,
        use_freq_attention: bool = False,
        use_classifier: bool = False,
        num_classes: int = 1,
        base_channels: int = 16,
        latent_channels: int = 64,
        loss_type: str = "ce",
    ) -> None:
        super().__init__()
        if base_channels < 1 or latent_channels < 1:
            raise ValueError("channel counts must be positive")
        if use_classifier and num_classes < 2:
            raise ValueError("use_classifier requires num_classes >= 2")
        if loss_type not in ("ce", "arcface"):
            raise ValueError(f"loss_type must be 'ce' or 'arcface', got {loss_type!r}")

        self.use_freq_attention = use_freq_attention
        self.use_classifier     = use_classifier
        self.num_classes        = num_classes
        self.loss_type          = loss_type

        # ---- Encoder -------------------------------------------------------
        enc: list[nn.Module] = [ConvBlock(1, base_channels)]
        if use_freq_attention:
            enc.append(FreqAxisAttention())
        enc += [
            ConvBlock(base_channels, base_channels * 2),
            ConvBlock(base_channels * 2, latent_channels),
        ]
        self.encoder = nn.Sequential(*enc)

        # ---- Decoder -------------------------------------------------------
        self.decoder = nn.Sequential(
            DeconvBlock(latent_channels, base_channels * 2),
            DeconvBlock(base_channels * 2, base_channels),
            nn.ConvTranspose2d(base_channels, 1, kernel_size=4, stride=2, padding=1),
        )

        # ---- Pooling (shared between encoder and classifier) ---------------
        self.pool_flatten = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

        # ---- Classifier head (optional) ------------------------------------
        self.classifier_head: nn.Module | None = None
        if use_classifier:
            if loss_type == "arcface":
                self.classifier_head = ArcFaceHead(latent_channels, num_classes)
            else:
                self.classifier_head = nn.Linear(latent_channels, num_classes)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Parameters
        ----------
        x      : [B, 1, n_mels, frames]  log-Mel spectrogram patches
        labels : [B] int64, section indices — required by ArcFace at training time

        Returns
        -------
        reconstruction : [B, 1, n_mels, frames]
        logits         : [B, num_classes] or None if use_classifier=False
        """
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"expected [B, 1, mel, time], got {tuple(x.shape)}")

        target_size = x.shape[-2:]
        latent      = self.encoder(x)

        # ---- Decoder with exact-size restoration ---------------------------
        reconstruction = self.decoder(latent)
        if reconstruction.shape[-2:] != target_size:
            reconstruction = F.interpolate(
                reconstruction, size=target_size,
                mode="bilinear", align_corners=False,
            )

        # ---- Classifier head ----------------------------------------------
        logits: torch.Tensor | None = None
        if self.classifier_head is not None:
            feats = self.pool_flatten(latent)
            if isinstance(self.classifier_head, ArcFaceHead):
                logits = self.classifier_head(feats, labels)
            else:
                logits = self.classifier_head(feats)

        return reconstruction, logits
