from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from aegis.engine import Trainer
from aegis.models import AEGISModel


class SyntheticMachine(Dataset):
    """Tiny baseline-shaped dataset representing one machine type."""

    def __init__(self, samples: int, n_mels: int, frames: int) -> None:
        generator = torch.Generator().manual_seed(7)
        self.data = torch.randn(samples, n_mels * frames, generator=generator)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return (
            self.data[index],
            torch.tensor(index % 2),
            torch.tensor([1.0]),
            f"synthetic_{index}.wav",
            index,
        )


class Stage1SmokeTest(unittest.TestCase):
    def test_one_machine_one_short_epoch(self) -> None:
        torch.set_num_threads(1)
        n_mels, frames = 32, 17
        loader = DataLoader(SyntheticMachine(8, n_mels, frames), batch_size=4)
        model = AEGISModel(stage=1, base_channels=4, latent_channels=8)
        trainer = Trainer(model, n_mels, frames, torch.device("cpu"))

        loss = trainer.train_epoch(loader, max_batches=2)
        calibration = trainer.calibrate(loader, quantile=0.9, max_batches=2)

        self.assertGreater(loss, 0.0)
        self.assertGreater(calibration.threshold, 0.0)
        x = torch.randn(2, 1, n_mels, frames)
        reconstruction, logits = model(x)
        self.assertEqual(reconstruction.shape, x.shape)
        self.assertIsNone(logits)


if __name__ == "__main__":
    unittest.main()

