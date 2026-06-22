"""Build the final cross-dataset comparison and AEGIS ablation tables."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METRICS = ("auc", "pauc", "f1")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_aegis(output_dir: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(Path(output_dir).glob("DCASE*T2/*/summary.csv")):
        for row in _read_csv(path):
            if row["machine_type"] == "arithmetic mean":
                continue
            experiment = row.get("experiment") or path.parent.name
            rows.append({**row, "method": f"AEGIS-{experiment}"})
    return rows


def _validate_reference(rows: Iterable[dict[str, str]], source: Path) -> None:
    required = {"dataset", "method", "machine_type", *METRICS}
    for index, row in enumerate(rows, start=2):
        missing = required.difference(row)
        if missing:
            raise ValueError(f"{source}:{index} missing columns: {sorted(missing)}")


def aggregate(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"])].append(row)
    result: dict[tuple[str, str], dict[str, float]] = {}
    for key, samples in grouped.items():
        result[key] = {
            metric: sum(float(sample[metric]) for sample in samples) / len(samples)
            for metric in METRICS
        }
    return result


def build_reports(
    output_dir: str | Path,
    report_dir: str | Path,
    reference_csvs: Iterable[str | Path] = (),
) -> tuple[Path, Path]:
    rows = collect_aegis(output_dir)
    for reference in reference_csvs:
        path = Path(reference)
        reference_rows = _read_csv(path)
        _validate_reference(reference_rows, path)
        rows.extend(reference_rows)
    if not rows:
        raise FileNotFoundError("no AEGIS summaries or reference rows were found")

    averages = aggregate(rows)
    datasets = sorted({key[0] for key in averages})
    methods = sorted({key[1] for key in averages})
    comparison_fields = ["method"] + [
        f"{dataset}_{metric}" for dataset in datasets for metric in METRICS
    ]
    comparison_rows: list[dict[str, Any]] = []
    for method in methods:
        row: dict[str, Any] = {"method": method}
        for dataset in datasets:
            values = averages.get((dataset, method), {})
            for metric in METRICS:
                row[f"{dataset}_{metric}"] = values.get(metric, "")
        comparison_rows.append(row)

    ablation_fields = ["dataset", "experiment", *METRICS]
    ablation_rows: list[dict[str, Any]] = []
    for (dataset, method), values in sorted(averages.items()):
        if not method.startswith("AEGIS-stage"):
            continue
        ablation_rows.append(
            {
                "dataset": dataset,
                "experiment": method.removeprefix("AEGIS-"),
                **values,
            }
        )

    report_dir = Path(report_dir)
    comparison_path = report_dir / "comparison.csv"
    ablation_path = report_dir / "ablation.csv"
    _write_csv(comparison_path, comparison_rows, comparison_fields)
    _write_csv(ablation_path, ablation_rows, ablation_fields)
    return comparison_path, ablation_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(Path(__file__).with_name("outputs")))
    parser.add_argument("--report-dir", default=str(Path(__file__).with_name("outputs") / "reports"))
    parser.add_argument("--reference-csv", action="append", default=[])
    args = parser.parse_args(argv)
    comparison, ablation = build_reports(
        args.output_dir,
        args.report_dir,
        args.reference_csv,
    )
    print(f"comparison: {comparison}")
    print(f"ablation: {ablation}")


if __name__ == "__main__":
    main()
