"""Training, calibration, checkpointing, and evaluation for AEGIS."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import flat_windows_to_images
from .metrics import binary_metrics
from .models import AEGISModel


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
        raise RuntimeError("CUDA was requested but is not available")
    return device


@dataclass
class ScoreCalibration:
    threshold: float
    mean: float
    std: float
    classification_mean: float = 0.0
    classification_std: float = 1.0
    fusion_weight: float = 0.0

    def normalize(self, score: float) -> float:
        return (score - self.mean) / max(self.std, 1e-12)

    def fuse(self, reconstruction: float, classification: float = 0.0) -> float:
        if self.fusion_weight == 0.0:
            return reconstruction
        classification_z = (classification - self.classification_mean) / max(
            self.classification_std, 1e-12
        )
        return self.normalize(reconstruction) + self.fusion_weight * classification_z


class Trainer:
    def __init__(
        self,
        model: AEGISModel,
        n_mels: int,
        frames: int,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        ssl_loss_weight: float = 0.2,
        fusion_weight: float = 0.3,
    ) -> None:
        self.model = model.to(device)
        self.n_mels = n_mels
        self.frames = frames
        self.device = device
        self.ssl_loss_weight = ssl_loss_weight
        self.fusion_weight = fusion_weight
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def _images(self, batch: Any) -> torch.Tensor:
        windows = batch[0].to(self.device, dtype=torch.float32)
        return flat_windows_to_images(windows, self.n_mels, self.frames)

    def _ssl_views(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Create balanced, automatically labelled transformation views."""
        if self.model.num_classes != 3:
            raise ValueError("the current SSL recipe requires num_classes=3")
        batch_size = images.shape[0]
        views = torch.cat(
            (images, torch.flip(images, dims=(-1,)), torch.flip(images, dims=(-2,))),
            dim=0,
        )
        targets = torch.cat(
            [
                torch.full((batch_size,), label, device=self.device, dtype=torch.long)
                for label in range(3)
            ]
        )
        return views, targets

    def _losses(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.model.stage < 3:
            reconstruction, _ = self.model(images)
            reconstruction_loss = F.mse_loss(reconstruction, images)
            zero = reconstruction_loss.new_zeros(())
            return reconstruction_loss, zero, reconstruction_loss

        views, targets = self._ssl_views(images)
        reconstruction, logits = self.model(views)
        if logits is None:
            raise RuntimeError("stage-3 model did not return classification logits")
        reconstruction_loss = F.mse_loss(reconstruction, views)
        classification_loss = F.cross_entropy(logits, targets)
        total = reconstruction_loss + self.ssl_loss_weight * classification_loss
        return reconstruction_loss, classification_loss, total

    def train_epoch(
        self,
        loader: Iterable[Any],
        max_batches: int | None = None,
    ) -> float:
        self.model.train()
        losses: list[float] = []
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = self._images(batch)
            _, _, loss = self._losses(images)
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
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = self._images(batch)
            _, _, loss = self._losses(images)
            losses.append(float(loss.cpu()))
        return float(np.mean(losses)) if losses else float("nan")

    @torch.no_grad()
    def calibrate(
        self,
        loader: Iterable[Any],
        quantile: float,
        max_batches: int | None = None,
    ) -> ScoreCalibration:
        if not 0.0 < quantile < 1.0:
            raise ValueError("threshold quantile must be between 0 and 1")
        self.model.eval()
        reconstruction_scores: list[float] = []
        classification_scores: list[float] = []
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = self._images(batch)
            reconstruction, _ = self.model(images)
            per_window = F.mse_loss(reconstruction, images, reduction="none").mean(
                dim=(1, 2, 3)
            )
            reconstruction_scores.extend(per_window.cpu().tolist())
            if self.model.stage >= 3:
                views, targets = self._ssl_views(images)
                _, logits = self.model(views)
                if logits is None:
                    raise RuntimeError("stage-3 model did not return logits")
                per_view = F.cross_entropy(logits, targets, reduction="none")
                per_window_classification = per_view.reshape(3, images.shape[0]).mean(0)
                classification_scores.extend(per_window_classification.cpu().tolist())
        if not reconstruction_scores:
            raise RuntimeError("calibration loader produced no samples")
        values = np.asarray(reconstruction_scores, dtype=np.float64)
        if self.model.stage < 3:
            return ScoreCalibration(
                threshold=float(np.quantile(values, quantile)),
                mean=float(values.mean()),
                std=float(values.std()),
            )

        classification_values = np.asarray(classification_scores, dtype=np.float64)
        reconstruction_z = (values - values.mean()) / max(values.std(), 1e-12)
        classification_z = (
            classification_values - classification_values.mean()
        ) / max(classification_values.std(), 1e-12)
        fused = reconstruction_z + self.fusion_weight * classification_z
        return ScoreCalibration(
            threshold=float(np.quantile(fused, quantile)),
            mean=float(values.mean()),
            std=float(values.std()),
            classification_mean=float(classification_values.mean()),
            classification_std=float(classification_values.std()),
            fusion_weight=self.fusion_weight,
        )

    @torch.no_grad()
    def score_file_loader(
        self,
        loader: Iterable[Any],
        calibration: ScoreCalibration,
        max_files: int | None = None,
    ) -> list[dict[str, Any]]:
        """Score baseline test batches (one complete WAV file per batch)."""
        self.model.eval()
        rows: list[dict[str, Any]] = []
        label_counts = {0: 0, 1: 0}
        for batch in loader:
            label = int(batch[1][0].item())
            if max_files is not None and label in label_counts:
                # Official loaders order normal files before anomalies. Keep a
                # balanced subset so a smoke-test limit can still compute AUC.
                quota = (max_files + (1 if label == 0 else 0)) // 2
                if label_counts[label] >= quota:
                    continue
            images = self._images(batch)
            reconstruction, _ = self.model(images)
            reconstruction_score = F.mse_loss(reconstruction, images).item()
            classification_score = 0.0
            if self.model.stage >= 3:
                views, targets = self._ssl_views(images)
                _, logits = self.model(views)
                if logits is None:
                    raise RuntimeError("stage-3 model did not return logits")
                classification_score = float(F.cross_entropy(logits, targets).item())
            score = calibration.fuse(reconstruction_score, classification_score)
            rows.append(
                {
                    "filename": str(batch[3][0]),
                    "label": label,
                    "reconstruction_score": float(reconstruction_score),
                    "classification_score": classification_score,
                    "score": float(score),
                }
            )
            if label in label_counts:
                label_counts[label] += 1
            if max_files is not None and len(rows) >= max_files:
                break
        return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_machine(
    config: dict[str, Any],
    dataset_name: str,
    machine_type: str,
    stage: int,
    *,
    epochs: int | None = None,
    max_train_batches: int | None = None,
    max_test_files: int | None = None,
) -> dict[str, Any]:
    """Train and evaluate one machine type, returning its summary row."""
    from .data import build_data_bundle

    seed_everything(int(config["seed"]))
    device = resolve_device(str(config["device"]))
    data = build_data_bundle(
        config["baseline_root"],
        dataset_name,
        machine_type,
        config["feature"],
        config["training"],
    )
    ssl = config["self_supervised"]
    model = AEGISModel(
        stage=stage,
        num_classes=int(ssl["num_classes"]),
        **config["model"],
    )
    trainer = Trainer(
        model,
        n_mels=int(config["feature"]["n_mels"]),
        frames=int(config["feature"]["frames"]),
        device=device,
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        ssl_loss_weight=float(ssl["loss_weight"]),
        fusion_weight=float(ssl["fusion_weight"]),
    )

    history: list[dict[str, float | int]] = []
    total_epochs = epochs if epochs is not None else int(config["training"]["epochs"])
    for epoch in range(1, total_epochs + 1):
        train_loss = trainer.train_epoch(data.train_loader, max_train_batches)
        valid_loss = trainer.validation_loss(data.valid_loader, max_train_batches)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        print(
            f"[{dataset_name}/{machine_type}/stage{stage}] epoch {epoch}/{total_epochs} "
            f"train={train_loss:.6f} valid={valid_loss:.6f}"
        )

    evaluation = config["evaluation"]
    calibration = trainer.calibrate(
        data.train_loader,
        float(evaluation["threshold_quantile"]),
        max_train_batches,
    )
    run_dir = Path(config["output_dir"]) / dataset_name / f"stage{stage}" / machine_type
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "stage": stage,
        "dataset": dataset_name,
        "machine_type": machine_type,
        "num_classes": model.num_classes,
        "feature": config["feature"],
        "model": config["model"],
        "calibration": calibration.__dict__,
    }
    torch.save(checkpoint, run_dir / "model.pt")
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    all_rows: list[dict[str, Any]] = []
    section_metrics: list[dict[str, Any]] = []
    for section_id, test_loader in zip(data.section_id_list, data.test_loader):
        rows = trainer.score_file_loader(test_loader, calibration, max_test_files)
        for row in rows:
            row["section"] = section_id
            row["decision"] = int(row["score"] > calibration.threshold)
        _write_csv(run_dir / f"anomaly_score_section_{section_id}.csv", rows)
        all_rows.extend(rows)
        if rows and len({row["label"] for row in rows}) == 2:
            measured = binary_metrics(
                [row["label"] for row in rows],
                [row["score"] for row in rows],
                calibration.threshold,
                float(evaluation["max_fpr"]),
            ).to_dict()
            measured["section"] = section_id
            section_metrics.append(measured)

    _write_csv(run_dir / "section_metrics.csv", section_metrics)
    if not all_rows or len({row["label"] for row in all_rows}) != 2:
        raise RuntimeError("test data does not contain both normal and anomalous files")
    overall = binary_metrics(
        [row["label"] for row in all_rows],
        [row["score"] for row in all_rows],
        calibration.threshold,
        float(evaluation["max_fpr"]),
    ).to_dict()
    summary = {
        "dataset": dataset_name,
        "machine_type": machine_type,
        "stage": stage,
        **overall,
    }
    _write_csv(run_dir / "summary.csv", [summary])
    return summary
