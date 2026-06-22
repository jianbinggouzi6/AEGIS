from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from aegis.report import build_reports


class ReportTest(unittest.TestCase):
    def test_cross_dataset_and_ablation_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "outputs"
            for dataset in ("DCASE2020T2", "DCASE2024T2"):
                for stage in (1, 2, 3):
                    path = output / dataset / f"stage{stage}" / "summary.csv"
                    path.parent.mkdir(parents=True)
                    with path.open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(
                            stream,
                            fieldnames=["dataset", "machine_type", "stage", "auc", "pauc", "precision", "recall", "f1"],
                        )
                        writer.writeheader()
                        writer.writerow(
                            {
                                "dataset": dataset,
                                "machine_type": "fan",
                                "stage": stage,
                                "auc": 0.8 + stage / 100,
                                "pauc": 0.7,
                                "precision": 0.6,
                                "recall": 0.6,
                                "f1": 0.6,
                            }
                        )
            comparison, ablation = build_reports(output, root / "reports")
            self.assertTrue(comparison.is_file())
            self.assertTrue(ablation.is_file())
            with ablation.open(encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 6)


if __name__ == "__main__":
    unittest.main()
