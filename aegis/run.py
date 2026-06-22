"""Command-line entry point for staged AEGIS experiments."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .engine import run_machine


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    base = config_path.parent
    config["baseline_root"] = str((base / config["baseline_root"]).resolve())
    config["output_dir"] = str((base / config["output_dir"]).resolve())
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--dataset", required=True, choices=["DCASE2020T2", "DCASE2024T2"])
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--machine-types", nargs="+", help="override config machine list")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--max-train-batches", type=int, help="smoke-test limiter")
    parser.add_argument("--max-test-files", type=int, help="smoke-test limiter per section")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = deepcopy(load_config(args.config))
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        config["device"] = args.device
    machine_types = args.machine_types or config["datasets"][args.dataset]["machine_types"]

    summaries = [
        run_machine(
            config,
            args.dataset,
            machine_type,
            args.stage,
            epochs=args.epochs,
            max_train_batches=args.max_train_batches,
            max_test_files=args.max_test_files,
        )
        for machine_type in machine_types
    ]
    metric_names = ["auc", "pauc", "precision", "recall", "f1"]
    mean_row = {
        "dataset": args.dataset,
        "machine_type": "arithmetic mean",
        "stage": args.stage,
        **{
            name: sum(float(row[name]) for row in summaries) / len(summaries)
            for name in metric_names
        },
    }
    rows = summaries + [mean_row]
    output = Path(config["output_dir"]) / args.dataset / f"stage{args.stage}" / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary: {output}")


if __name__ == "__main__":
    main()
