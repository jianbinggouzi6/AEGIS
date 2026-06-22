"""Entry point for AEGIS experiments.

Usage
-----
    python -m aegis.train --config configs/exp1_convae.yaml

The full resolved config is saved to <output_dir>/config.yaml at the start of
each run so results are always reproducible from the stored file.
"""
from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .engine import run_machine, seed_everything


# ---------------------------------------------------------------------------
# Config loading (supports an optional 'extends' key for inheritance)
# ---------------------------------------------------------------------------

def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    extends = raw.pop("extends", None)
    base_config: dict[str, Any] = load_config(config_path.parent / extends) if extends else {}
    # Resolve path-like values relative to the config file's directory
    for key in ("baseline_root", "output_dir"):
        if key in raw:
            raw[key] = str((config_path.parent / raw[key]).resolve())
    return _deep_merge(base_config, raw)


# ---------------------------------------------------------------------------
# Dataset name and default machine list
# ---------------------------------------------------------------------------

_DATASET_NAMES = {
    "a": "DCASE2020T2",
    "b": "DCASE2024T2",
}

_DEFAULT_MACHINES: dict[str, list[str]] = {
    "DCASE2020T2": ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"],
    "DCASE2024T2": ["bearing", "fan", "gearbox", "slider", "valve", "ToyCar", "ToyTrain"],
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train and evaluate an AEGIS ablation experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True,
                   help="Path to experiment YAML (e.g. configs/exp1_convae.yaml)")
    p.add_argument("--machine-types", nargs="+",
                   help="Override the machine list from the config")
    p.add_argument("--epochs", type=int,
                   help="Override training.epochs")
    p.add_argument("--batch-size", type=int,
                   help="Override training.batch_size")
    p.add_argument("--device",
                   help="Override device (auto | cpu | cuda | cuda:N)")
    p.add_argument("--max-train-batches", type=int,
                   help="Limit batches per epoch (debugging)")
    p.add_argument("--max-test-files", type=int,
                   help="Limit test files per section (debugging)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # ---- Load and patch config --------------------------------------------
    config = load_config(args.config)

    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.device is not None:
        config["device"] = args.device
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs

    # ---- Seed (before anything else) --------------------------------------
    seed_everything(int(config.get("seed", 13711)))

    # ---- Resolve dataset --------------------------------------------------
    dataset_key  = str(config.get("dataset", "a")).lower()
    dataset_name = _DATASET_NAMES.get(dataset_key, dataset_key)

    machine_types: list[str] = (
        args.machine_types
        or config.get("machine_types")
        or _DEFAULT_MACHINES.get(dataset_name, [])
    )
    if not machine_types:
        raise ValueError(
            f"No machine types found for dataset '{dataset_name}'. "
            "Add machine_types to the config or pass --machine-types."
        )

    experiment_name = str(config.get("experiment_name", "aegis"))
    output_dir      = Path(config["output_dir"])

    # ---- Save the full resolved config for reproducibility ----------------
    run_root = output_dir / dataset_name / experiment_name
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "config.yaml").write_text(
        yaml.dump(config, allow_unicode=True), encoding="utf-8"
    )

    print("=" * 60)
    print(f"Experiment : {experiment_name}")
    print(f"Dataset    : {dataset_name}")
    print(f"Machines   : {machine_types}")
    print(f"Output     : {run_root}")
    print("=" * 60)

    # ---- Per-machine training and evaluation ------------------------------
    summaries: list[dict[str, Any]] = []
    for machine_type in machine_types:
        print(f"\n>>> {machine_type}")
        summary = run_machine(
            config=config,
            dataset_name=dataset_name,
            machine_type=machine_type,
            max_train_batches=args.max_train_batches,
            max_test_files=args.max_test_files,
        )
        summaries.append(summary)
        print(
            f"    AUC={summary['auc']:.4f}  "
            f"pAUC={summary['pauc']:.4f}  "
            f"F1={summary['f1']:.4f}"
        )

    # ---- Aggregate results ------------------------------------------------
    metric_keys  = ["auc", "pauc", "precision", "recall", "f1"]
    mean_row: dict[str, Any] = {
        "dataset":      dataset_name,
        "machine_type": "arithmetic_mean",
        "experiment":   experiment_name,
        **{
            k: sum(float(s[k]) for s in summaries) / len(summaries)
            for k in metric_keys
        },
    }

    all_rows = summaries + [mean_row]
    result_csv = run_root / "result.csv"
    with result_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n" + "=" * 60)
    print(f"Results saved to: {result_csv}")
    print(f"\nMean AUC  = {mean_row['auc']:.4f}")
    print(f"Mean pAUC = {mean_row['pauc']:.4f}")
    print(f"Mean F1   = {mean_row['f1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
