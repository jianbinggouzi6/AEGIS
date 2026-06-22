"""Adapter around the unmodified official DCASE baseline data loader."""

from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _baseline_datasets_class(baseline_root: Path) -> type:
    """Import the baseline loader without copying or changing baseline code."""
    root_text = str(baseline_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        module = importlib.import_module("datasets.datasets")
    except ModuleNotFoundError as exc:
        missing = exc.name or "a dependency"
        raise RuntimeError(
            f"Cannot import the official baseline loader (missing {missing!r}). "
            "Install dependencies with: pip install -r aegis/requirements.txt"
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
    """Build the exact loader object used by the official Dense-AE baseline."""
    baseline_root = Path(baseline_root).resolve()
    if not (baseline_root / "datasets" / "datasets.py").is_file():
        raise FileNotFoundError(f"Invalid baseline_root: {baseline_root}")

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
    windows: Any,
    n_mels: int,
    frames: int,
) -> Any:
    """Restore baseline `[time blocks × mel]` vectors to 2D images."""
    if windows.ndim != 2 or windows.shape[1] != n_mels * frames:
        raise ValueError(
            f"expected [batch, {n_mels * frames}] baseline vectors, "
            f"got {tuple(windows.shape)}"
        )
    return windows.reshape(-1, frames, n_mels).transpose(1, 2).unsqueeze(1)

