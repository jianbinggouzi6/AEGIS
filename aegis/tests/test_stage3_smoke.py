from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader

from aegis.engine import Trainer
from aegis.models import AEGISModel
from aegis.tests.test_stage1_smoke import SyntheticMachine


class Stage3SmokeTest(unittest.TestCase):
    def test_ssl_head_fused_score_and_short_training(self) -> None:
        torch.set_num_threads(1)
        n_mels, frames = 30, 15
        loader = DataLoader(SyntheticMachine(8, n_mels, frames), batch_size=4)
        model = AEGISModel(
            stage=3,
            num_classes=3,
            base_channels=4,
            latent_channels=8,
        )
        trainer = Trainer(
            model,
            n_mels,
            frames,
            torch.device("cpu"),
            ssl_loss_weight=0.2,
            fusion_weight=0.3,
        )

        loss = trainer.train_epoch(loader, max_batches=2)
        calibration = trainer.calibrate(loader, quantile=0.9, max_batches=2)
        rows = trainer.score_file_loader(loader, calibration, max_files=1)

        self.assertGreater(loss, 0.0)
        self.assertEqual(calibration.fusion_weight, 0.3)
        self.assertEqual(len(rows), 1)
        self.assertIn("classification_score", rows[0])
        x = torch.randn(2, 1, n_mels, frames)
        reconstruction, logits = model(x)
        self.assertEqual(reconstruction.shape, x.shape)
        self.assertEqual(logits.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
