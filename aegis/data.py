"""Adapter around the unmodified official DCASE baseline data loader.

The baseline stores mel features as flat vectors of shape [frames * n_mels].
flat_windows_to_images restores them to 2D log-Mel images [B, 1, n_mels, frames]
for the convolutional model without modifying any baseline code.

Batch format returned by the baseline DataLoader
-------------------------------------------------
batch[0] : FloatTensor [B, frames*n_mels]  — flattened log-Mel windows
batch[1] : FloatTensor [B]                  — anomaly label (0=normal, 1=anomalous)
batch[2] : FloatTensor [B, n_sections]      — one-hot section/condition vector
batch[3] : list[str]  [B]                   — audio file basenames
batch[4] : LongTensor [B]                   — window start indices
"""
from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import torch


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _baseline_datasets_class(baseline_root: Path) -> type:
    """Import the baseline Datasets class without copying or modifying baseline code."""
    root_str = str(baseline_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        module = importlib.import_module("datasets.datasets")
    except ModuleNotFoundError as exc:
        missing = exc.name or "a dependency"
        raise RuntimeError(
            f"Cannot import the official baseline loader (missing {missing!r}). "
            "Install dependencies: pip install -r aegis/requirements.txt"
        ) from exc
    return module.Datasets


def build_data_bundle(
    baseline_root: str | Path,
    dataset: str,
    machine_type: str,
    feature: dict[str, Any],
    training: dict[str, Any],
    *,
    dev: bool = True,
    train_only: bool = False,
    use_ids: list[int] | None = None,
    auto_download: bool = False,
) -> Any:
    """Instantiate the official baseline data bundle for one machine type.

    Parameters
    ----------
    baseline_root : path to the cloned dcase2023_task2_baseline_ae repo
    dataset       : dataset prefix, e.g. "DCASE2020T2" or "DCASE2024T2"
    machine_type  : e.g. "fan", "pump", "ToyCar"
    feature       : dict with n_mels, frames, n_fft, hop_length, etc.
    training      : dict with batch_size, validation_split, etc.

    Returns
    -------
    data : DCASE202XT2 instance with attributes
        .train_loader        — DataLoader over normal training windows
        .valid_loader        — DataLoader over validation split
        .test_loader         — list of per-section DataLoaders
        .section_id_list     — list of section ID strings (e.g. ["00","02","04","06"])
        .num_classes         — number of sections (int)
    """
    baseline_root = Path(baseline_root).resolve()
    if not (baseline_root / "datasets" / "datasets.py").is_file():
        raise FileNotFoundError(
            f"Invalid baseline_root (datasets/datasets.py not found): {baseline_root}"
        )

    args = SimpleNamespace(
        dataset=f"{dataset}{machine_type}",
        dataset_directory="./data",
        eval=not dev,
        dev=dev,
        shuffle=True,
        batch_size=int(training["batch_size"]),
        validation_split=float(training["validation_split"]),
        train_only=train_only,
        use_ids=use_ids or [],
        is_auto_download=auto_download,
        mono=True,
        **feature,
    )
    datasets_class = _baseline_datasets_class(baseline_root)
    with _working_directory(baseline_root):
        return datasets_class(args.dataset).data(args)


def flat_windows_to_images(
    windows: torch.Tensor,
    n_mels: int,
    frames: int,
) -> torch.Tensor:
    """Reshape baseline [B, frames*n_mels] flat vectors to 2D log-Mel images.

    The baseline concatenates mel frames in time order:
        window = [mel_0..mel_{n_mels-1}@t0, mel_0..@t1, ..., @t_{frames-1}]

    Reshape to [B, frames, n_mels] then transpose to [B, n_mels, frames],
    add a channel dim to get [B, 1, n_mels, frames] — the expected Conv2D input
    with n_mels as the frequency axis and frames as the time axis.
    """
    expected_dim = n_mels * frames
    if windows.ndim != 2 or windows.shape[1] != expected_dim:
        raise ValueError(
            f"expected [B, {expected_dim}] (n_mels={n_mels} × frames={frames}), "
            f"got {tuple(windows.shape)}"
        )
    # [B, frames, n_mels] → [B, n_mels, frames] → [B, 1, n_mels, frames]
    return (
        windows
        .reshape(-1, frames, n_mels)
        .permute(0, 2, 1)
        .unsqueeze(1)
        .contiguous()
    )
