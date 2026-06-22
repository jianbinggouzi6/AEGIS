"""Training, per-section calibration, checkpointing, and evaluation for AEGIS."""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import flat_windows_to_images
from .metrics import binary_metrics
from .models import AEGISModel


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


# ---------------------------------------------------------------------------
# Per-section anomaly-score calibration
# ---------------------------------------------------------------------------

@dataclass
class SectionStats:
    recon_mean: float = 0.0
    recon_std:  float = 1.0
    cls_mean:   float = 0.0
    cls_std:    float = 1.0


@dataclass
class SectionCalibration:
    """Z-score statistics and decision threshold derived from normal training data.

    section_stats maps integer section index (0-based, matching argmax of the
    one-hot condition vector in training batches) to per-section statistics.
    At test time the section index is the enumerate position in section_id_list.
    """
    section_stats:  dict[int, SectionStats] = field(default_factory=dict)
    threshold:      float = 0.0
    fusion_weight:  float = 0.0

    def score(
        self,
        recon_score: float,
        cls_score:   float,
        section_idx: int,
    ) -> float:
        """Compute the fused anomaly score with z-score normalization."""
        stats = self.section_stats.get(section_idx, SectionStats())
        z_recon = (recon_score - stats.recon_mean) / max(stats.recon_std, 1e-12)
        if self.fusion_weight == 0.0:
            return z_recon
        z_cls = (cls_score - stats.cls_mean) / max(stats.cls_std, 1e-12)
        return z_recon + self.fusion_weight * z_cls

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # section_stats keys are ints; JSON only accepts str keys
        d["section_stats"] = {str(k): asdict(v) for k, v in self.section_stats.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SectionCalibration":
        stats = {int(k): SectionStats(**v) for k, v in d.get("section_stats", {}).items()}
        return cls(
            section_stats=stats,
            threshold=float(d.get("threshold", 0.0)),
            fusion_weight=float(d.get("fusion_weight", 0.0)),
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Wraps AEGISModel with training, calibration, and scoring utilities."""

    def __init__(
        self,
        model: AEGISModel,
        n_mels: int,
        frames: int,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        lambda_cls: float = 0.2,
        fusion_weight: float = 0.3,
    ) -> None:
        self.model          = model.to(device)
        self.n_mels         = n_mels
        self.frames         = frames
        self.device         = device
        self.lambda_cls     = lambda_cls
        self.fusion_weight  = fusion_weight
        self.learning_rate  = learning_rate
        self.weight_decay   = weight_decay
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_images(
        self, batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a baseline batch to 2D images and integer section labels.

        batch[0] : [B, frames*n_mels] flattened mel features
        batch[2] : [B, n_sections]    one-hot condition vector (section ID)

        Returns
        -------
        images         : [B, 1, n_mels, frames]  float32
        section_labels : [B]                      int64 (argmax of one-hot)
        """
        windows = batch[0].to(self.device, dtype=torch.float32)
        images  = flat_windows_to_images(windows, self.n_mels, self.frames)
        section_labels = batch[2].float().argmax(dim=1).long().to(self.device)
        return images, section_labels

    def _compute_losses(
        self,
        images: torch.Tensor,
        section_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass and compute MSE + λ·CE.

        Returns (recon_loss, cls_loss, total_loss).
        """
        reconstruction, logits = self.model(
            images,
            labels=section_labels if self.model.training else None,
        )
        recon_loss = F.mse_loss(reconstruction, images)
        cls_loss   = torch.zeros((), device=self.device)
        if logits is not None and self.model.training:
            cls_loss = F.cross_entropy(logits, section_labels)
        total = recon_loss + self.lambda_cls * cls_loss
        return recon_loss, cls_loss, total

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        loader: Iterable[Any],
        max_batches: int | None = None,
    ) -> float:
        self.model.train()
        losses: list[float] = []
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            images, section_labels = self._to_images(batch)
            _, _, loss = self._compute_losses(images, section_labels)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if not losses:
            raise RuntimeError("training loader produced no batches")
        return float(np.mean(losses))

    @torch.no_grad()
    def validation_loss(
        self,
        loader: Iterable[Any],
        max_batches: int | None = None,
    ) -> float:
        self.model.eval()
        losses: list[float] = []
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            images, section_labels = self._to_images(batch)
            _, _, loss = self._compute_losses(images, section_labels)
            losses.append(float(loss.cpu()))
        return float(np.mean(losses)) if losses else float("nan")

    # ------------------------------------------------------------------
    # Staged training helpers
    # ------------------------------------------------------------------

    def train_ae_phase(
        self,
        loader: Iterable[Any],
        epochs: int,
        valid_loader: Iterable[Any] | None = None,
        max_batches: int | None = None,
    ) -> list[dict[str, float]]:
        """Train reconstruction only (λ=0), classifier head receives no gradient."""
        saved_lambda = self.lambda_cls
        self.lambda_cls = 0.0
        history = []
        for epoch in range(1, epochs + 1):
            tr = self.train_epoch(loader, max_batches)
            vl = self.validation_loss(valid_loader, max_batches) if valid_loader else float("nan")
            history.append({"epoch": epoch, "train_loss": tr, "valid_loss": vl})
        self.lambda_cls = saved_lambda
        return history

    def train_clf_phase(
        self,
        loader: Iterable[Any],
        epochs: int,
        valid_loader: Iterable[Any] | None = None,
        max_batches: int | None = None,
    ) -> list[dict[str, float]]:
        """Fine-tune classifier head only; encoder and decoder are frozen."""
        if self.model.classifier_head is None:
            return []
        # Freeze encoder + decoder
        for param in (*self.model.encoder.parameters(),
                      *self.model.decoder.parameters()):
            param.requires_grad_(False)
        # Optimizer covers only the classifier head
        clf_optimizer = torch.optim.Adam(
            self.model.classifier_head.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        saved_optimizer  = self.optimizer
        self.optimizer   = clf_optimizer
        history = []
        for epoch in range(1, epochs + 1):
            tr = self.train_epoch(loader, max_batches)
            vl = self.validation_loss(valid_loader, max_batches) if valid_loader else float("nan")
            history.append({"epoch": epoch, "train_loss": tr, "valid_loss": vl})
        # Restore
        self.optimizer = saved_optimizer
        for param in (*self.model.encoder.parameters(),
                      *self.model.decoder.parameters()):
            param.requires_grad_(True)
        return history

    # ------------------------------------------------------------------
    # Calibration (per-section z-score from normal training data)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def calibrate(
        self,
        loader: Iterable[Any],
        n_sections: int,
        quantile: float,
        max_batches: int | None = None,
    ) -> SectionCalibration:
        """Collect reconstruction and classification scores per section index.

        Uses training data (which is normal-only for DCASE) to compute
        per-section mean/std and a global fused-score threshold.
        """
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be in (0, 1)")
        self.model.eval()

        per_section_recon: dict[int, list[float]] = defaultdict(list)
        per_section_cls:   dict[int, list[float]] = defaultdict(list)

        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            images, section_labels = self._to_images(batch)
            reconstruction, logits = self.model(images)

            # Per-sample reconstruction score (mean over mel×time)
            per_sample_recon = F.mse_loss(
                reconstruction, images, reduction="none"
            ).mean(dim=(1, 2, 3))                              # [B]

            for s_idx, r in zip(
                section_labels.cpu().tolist(),
                per_sample_recon.cpu().tolist(),
            ):
                per_section_recon[int(s_idx)].append(float(r))

            # Per-sample classification score (1 − max softmax prob)
            if logits is not None:
                probs = F.softmax(logits, dim=1)               # [B, n_cls]
                cls_scores = 1.0 - probs.max(dim=1).values     # [B]
                for s_idx, c in zip(
                    section_labels.cpu().tolist(),
                    cls_scores.cpu().tolist(),
                ):
                    per_section_cls[int(s_idx)].append(float(c))

        # Build per-section statistics and collect all fused z-scores
        section_stats: dict[int, SectionStats] = {}
        all_fused: list[float] = []

        for s_idx in range(n_sections):
            rv = np.asarray(per_section_recon.get(s_idx, [0.0]), dtype=np.float64)
            cv = np.asarray(per_section_cls.get(s_idx, [0.0]), dtype=np.float64)

            rm = float(rv.mean())
            rs = max(float(rv.std()), 1e-12)
            cm = float(cv.mean())
            cs = max(float(cv.std()), 1e-12)
            section_stats[s_idx] = SectionStats(
                recon_mean=rm, recon_std=rs,
                cls_mean=cm,   cls_std=cs,
            )

            z_recon = (rv - rm) / rs
            if self.model.use_classifier and per_section_cls.get(s_idx):
                cv_section = np.asarray(per_section_cls[s_idx], dtype=np.float64)
                z_cls  = (cv_section - cm) / cs
                fused  = z_recon + self.fusion_weight * z_cls
            else:
                fused = z_recon
            all_fused.extend(fused.tolist())

        if not all_fused:
            raise RuntimeError("calibration loader produced no samples")

        threshold = float(np.quantile(all_fused, quantile))
        return SectionCalibration(
            section_stats=section_stats,
            threshold=threshold,
            fusion_weight=self.fusion_weight if self.model.use_classifier else 0.0,
        )

    # ------------------------------------------------------------------
    # Test scoring (one full file per batch from baseline test loaders)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_file_loader(
        self,
        loader: Iterable[Any],
        calibration: SectionCalibration,
        section_train_idx: int,
        max_files: int | None = None,
    ) -> list[dict[str, Any]]:
        """Score every test file and return one row per file.

        Parameters
        ----------
        loader            : test DataLoader (one batch = all windows of one file)
        calibration       : SectionCalibration from :meth:`calibrate`
        section_train_idx : position of this section in data.section_id_list,
                            used to look up the matching per-section statistics
        max_files         : cap on the number of files scored (debugging)
        """
        self.model.eval()
        rows: list[dict[str, Any]] = []

        for batch in loader:
            if max_files is not None and len(rows) >= max_files:
                break
            images, _ = self._to_images(batch)
            reconstruction, logits = self.model(images)

            # Whole-file reconstruction score (mean MSE over all windows)
            recon_score = float(
                F.mse_loss(reconstruction, images).item()
            )

            # Whole-file classification score (mean 1-maxprob over all windows)
            cls_score = 0.0
            if self.model.use_classifier and logits is not None:
                probs = F.softmax(logits, dim=1)             # [n_windows, n_cls]
                cls_score = float(
                    (1.0 - probs.max(dim=1).values).mean().item()
                )

            # Fused anomaly score with per-section z-score normalization
            score = calibration.score(recon_score, cls_score, section_train_idx)

            label    = int(batch[1][0].item())
            filename = str(batch[3][0])
            rows.append({
                "filename":            filename,
                "label":               label,
                "reconstruction_score": recon_score,
                "classification_score": cls_score,
                "score":               float(score),
                "decision":            int(score > calibration.threshold),
            })

        return rows


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Top-level per-machine runner
# ---------------------------------------------------------------------------

def run_machine(
    config: dict[str, Any],
    dataset_name: str,
    machine_type: str,
    *,
    max_train_batches: int | None = None,
    max_test_files: int | None = None,
) -> dict[str, Any]:
    """Train, calibrate, and evaluate one machine type.

    Returns a summary dict with dataset, machine_type, experiment_name,
    AUC, pAUC, and F1.
    """
    from .data import build_data_bundle

    seed_everything(int(config["seed"]))
    device = resolve_device(str(config.get("device", "auto")))

    # ---- Data --------------------------------------------------------------
    data = build_data_bundle(
        baseline_root=config["baseline_root"],
        dataset=dataset_name,
        machine_type=machine_type,
        feature=config["feature"],
        training=config["training"],
    )

    # ---- Model -------------------------------------------------------------
    model_cfg   = config["model"]
    clf_cfg     = config.get("classifier", {})
    fusion_cfg  = config.get("fusion", {})

    use_cls   = bool(model_cfg.get("classifier_fusion", False))
    n_classes = int(clf_cfg.get("num_classes", 0))
    if n_classes == 0:
        n_classes = int(data.num_classes)      # auto-detect from dataset

    if use_cls and n_classes < 2:
        print(
            f"[{machine_type}] WARNING: num_classes={n_classes} < 2; "
            "classifier_fusion disabled for this machine."
        )
        use_cls   = False
        n_classes = max(n_classes, 1)

    model = AEGISModel(
        use_freq_attention=bool(model_cfg.get("freq_attention", False)),
        use_classifier=use_cls,
        num_classes=n_classes,
        base_channels=int(model_cfg.get("base_channels", 16)),
        latent_channels=int(model_cfg.get("latent_channels", 64)),
        loss_type=str(clf_cfg.get("loss_type", "ce")),
    )

    # ---- Trainer -----------------------------------------------------------
    tr_cfg = config["training"]
    trainer = Trainer(
        model=model,
        n_mels=int(config["feature"]["n_mels"]),
        frames=int(config["feature"]["frames"]),
        device=device,
        learning_rate=float(tr_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(tr_cfg.get("weight_decay", 1e-5)),
        lambda_cls=float(clf_cfg.get("lambda_cls", 0.2)),
        fusion_weight=float(fusion_cfg.get("weight", 0.3)) if use_cls else 0.0,
    )

    # ---- Training loop -----------------------------------------------------
    experiment_name = str(config.get("experiment_name", "aegis"))
    staged   = bool(tr_cfg.get("staged", False))
    history: list[dict[str, Any]] = []

    if staged and use_cls:
        ae_epochs  = int(tr_cfg.get("ae_epochs",  30))
        clf_epochs = int(tr_cfg.get("clf_epochs", 20))
        print(f"[{machine_type}] staged training: {ae_epochs} AE + {clf_epochs} CLF epochs")
        h1 = trainer.train_ae_phase(
            data.train_loader, ae_epochs, data.valid_loader, max_train_batches
        )
        h2 = trainer.train_clf_phase(
            data.train_loader, clf_epochs, data.valid_loader, max_train_batches
        )
        history = h1 + h2
    else:
        total_epochs = int(tr_cfg.get("epochs", 50))
        for epoch in range(1, total_epochs + 1):
            tr_loss = trainer.train_epoch(data.train_loader, max_train_batches)
            vl_loss = trainer.validation_loss(data.valid_loader, max_train_batches)
            history.append({"epoch": epoch, "train_loss": tr_loss, "valid_loss": vl_loss})
            print(
                f"[{dataset_name}/{machine_type}/{experiment_name}] "
                f"epoch {epoch}/{total_epochs}  "
                f"train={tr_loss:.6f}  valid={vl_loss:.6f}"
            )

    # ---- Calibration -------------------------------------------------------
    eval_cfg = config.get("evaluation", {})
    calibration = trainer.calibrate(
        loader=data.train_loader,
        n_sections=len(data.section_id_list),
        quantile=float(eval_cfg.get("threshold_quantile", 0.9)),
        max_batches=max_train_batches,
    )

    # ---- Checkpoint --------------------------------------------------------
    run_dir = (
        Path(config["output_dir"]) / dataset_name / experiment_name / machine_type
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state":    model.state_dict(),
            "experiment":     experiment_name,
            "dataset":        dataset_name,
            "machine_type":   machine_type,
            "num_classes":    model.num_classes,
            "feature":        config["feature"],
            "model_config":   model_cfg,
            "calibration":    calibration.to_dict(),
        },
        run_dir / "model.pt",
    )
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    # ---- Evaluation --------------------------------------------------------
    all_rows: list[dict[str, Any]] = []
    section_metrics: list[dict[str, Any]] = []
    max_fpr = float(eval_cfg.get("max_fpr", 0.1))

    for s_idx, (section_id, test_loader) in enumerate(
        zip(data.section_id_list, data.test_loader)
    ):
        rows = trainer.score_file_loader(
            test_loader, calibration,
            section_train_idx=s_idx,
            max_files=max_test_files,
        )
        for row in rows:
            row["section"] = section_id
        _write_csv(run_dir / f"anomaly_score_section_{section_id}.csv", rows)
        all_rows.extend(rows)

        labels = [r["label"] for r in rows]
        scores = [r["score"]  for r in rows]
        if rows and len(set(labels)) == 2:
            m = binary_metrics(labels, scores, calibration.threshold, max_fpr).to_dict()
            m["section"] = section_id
            section_metrics.append(m)

    _write_csv(run_dir / "section_metrics.csv", section_metrics)

    if not all_rows or len({r["label"] for r in all_rows}) != 2:
        raise RuntimeError(
            f"[{machine_type}] test data must contain both normal and anomalous files"
        )

    overall = binary_metrics(
        [r["label"] for r in all_rows],
        [r["score"]  for r in all_rows],
        calibration.threshold,
        max_fpr,
    ).to_dict()

    summary = {
        "dataset":      dataset_name,
        "machine_type": machine_type,
        "experiment":   experiment_name,
        **overall,
    }
    _write_csv(run_dir / "summary.csv", [summary])
    return summary
