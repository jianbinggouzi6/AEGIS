from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader

from aegis.engine import Trainer
from aegis.models import AEGISModel, FrequencyAxisAttention
from aegis.tests.test_stage1_smoke import SyntheticMachine


class Stage2SmokeTest(unittest.TestCase):
    def test_frequency_attention_and_short_training(self) -> None:
        torch.set_num_threads(1)
        n_mels, frames = 31, 19
        attention = FrequencyAxisAttention(kernel_size=3)
        features = torch.randn(2, 4, n_mels, 10)
        weights = attention.weights(features)
        self.assertEqual(weights.shape, (2, 1, n_mels, 1))
        self.assertTrue(torch.all((weights >= 0) & (weights <= 1)))

        loader = DataLoader(SyntheticMachine(8, n_mels, frames), batch_size=4)
        model = AEGISModel(stage=2, base_channels=4, latent_channels=8)
        trainer = Trainer(model, n_mels, frames, torch.device("cpu"))
        loss = trainer.train_epoch(loader, max_batches=2)

        self.assertGreater(loss, 0.0)
        x = torch.randn(2, 1, n_mels, frames)
        reconstruction, logits = model(x)
        self.assertEqual(reconstruction.shape, x.shape)
        self.assertIsNone(logits)


if __name__ == "__main__":
    unittest.main()
