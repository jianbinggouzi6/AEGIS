# AEGIS

AEGIS keeps both reference implementations unchanged and reuses the official
Dense-AE repository's DCASE2020/2024 loader. Its first stage is a 2D
convolutional autoencoder over log-Mel windows.

## Setup

```bash
pip install -r aegis/requirements.txt
```

Place the datasets under the baseline's existing `data/dcase2020t2` and
`data/dcase2024t2` directories. The directory and WAV filename conventions are
the ones documented by the official baseline.

## Stage 1

Run one machine type first:

```bash
python -m aegis.run --dataset DCASE2020T2 --stage 1 \
  --machine-types fan --epochs 1 --max-train-batches 2 --max-test-files 10
```

Then run every machine type configured for the dataset:

```bash
python -m aegis.run --dataset DCASE2020T2 --stage 1
python -m aegis.run --dataset DCASE2024T2 --stage 1
```

Outputs are written below `aegis/outputs/<dataset>/stage1/`, including model
checkpoints, per-file anomaly scores, section metrics, and a cross-machine
summary. Use `--machine-types` to select a fixed six-machine subset for an
experiment; the official DCASE2024 development map currently contains seven.

Run the dependency-light model smoke test with:

```bash
python -m unittest aegis.tests.test_stage1_smoke -v
```
