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

    def normalize(self, score: float) -> float:
        return (score - self.mean) / max(self.std, 1e-12)


class Trainer:
    def __init__(
        self,
        model: AEGISModel,
        n_mels: int,
        frames: int,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ) -> None:
        self.model = model.to(device)
        self.n_mels = n_mels
        self.frames = frames
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def _images(self, batch: Any) -> torch.Tensor:
        windows = batch[0].to(self.device, dtype=torch.float32)
        return flat_windows_to_images(windows, self.n_mels, self.frames)

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
            reconstruction, _ = self.model(images)
            loss = F.mse_loss(reconstruction, images)
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
            reconstruction, _ = self.model(images)
            losses.append(float(F.mse_loss(reconstruction, images).cpu()))
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
        scores: list[float] = []
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = self._images(batch)
            reconstruction, _ = self.model(images)
            per_window = F.mse_loss(reconstruction, images, reduction="none").mean(
                dim=(1, 2, 3)
            )
            scores.extend(per_window.cpu().tolist())
        if not scores:
            raise RuntimeError("calibration loader produced no samples")
        values = np.asarray(scores, dtype=np.float64)
        return ScoreCalibration(
            threshold=float(np.quantile(values, quantile)),
            mean=float(values.mean()),
            std=float(values.std()),
        )

    @torch.no_grad()
    def score_file_loader(
        self,
        loader: Iterable[Any],
        max_files: int | None = None,
    ) -> list[dict[str, Any]]:
        """Score baseline test batches (one complete WAV file per batch)."""
        self.model.eval()
        rows: list[dict[str, Any]] = []
        for file_index, batch in enumerate(loader):
            if max_files is not None and file_index >= max_files:
                break
            images = self._images(batch)
            reconstruction, _ = self.model(images)
            score = F.mse_loss(reconstruction, images).item()
            rows.append(
                {
                    "filename": str(batch[3][0]),
                    "label": int(batch[1][0].item()),
                    "score": float(score),
                }
            )
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
    model = AEGISModel(
        stage=stage,
        num_classes=data.num_classes,
        **config["model"],
    )
    trainer = Trainer(
        model,
        n_mels=int(config["feature"]["n_mels"]),
        frames=int(config["feature"]["frames"]),
        device=device,
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
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
        "num_classes": data.num_classes,
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
        rows = trainer.score_file_loader(test_loader, max_test_files)
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

