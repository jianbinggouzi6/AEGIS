from __future__ import annotations

import unittest
from pathlib import Path

from aegis.models import AEGISModel
from aegis.run import load_config


class AblationConfigTest(unittest.TestCase):
    def test_all_ablation_configs_load_and_build(self) -> None:
        config_dir = Path(__file__).parents[1] / "configs"
        expected = {
            "01_stage1_conv_ae.yaml": (False, False),
            "02_stage2_frequency_attention.yaml": (True, False),
            "03_stage3_full_aegis.yaml": (True, True),
        }
        for filename, flags in expected.items():
            with self.subTest(config=filename):
                config = load_config(config_dir / filename)
                model = AEGISModel(
                    stage=config["stage"],
                    num_classes=config["self_supervised"]["num_classes"],
                    **config["model"],
                )
                self.assertEqual(
                    (model.use_frequency_attention, model.use_ssl_head), flags
                )
                self.assertTrue(Path(config["baseline_root"]).is_absolute())
                self.assertTrue(Path(config["output_dir"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
