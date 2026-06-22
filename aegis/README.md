# AEGIS

AEGIS keeps both reference implementations unchanged and reuses the official
Dense-AE repository's DCASE2020/2024 loader. Stage 1 is a 2D convolutional
autoencoder over log-Mel windows; stage 2 adds learned frequency-axis attention;
stage 3 adds self-supervised transformation classification and score fusion.

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

## Stage 2

The stage-2 attention gate pools over channel and time, preserves every Mel
bin, and uses local frequency context to reweight encoder features:

```bash
python -m aegis.run --dataset DCASE2020T2 --stage 2 \
  --machine-types fan --epochs 1 --max-train-batches 2 --max-test-files 10
```

## Stage 3

The auxiliary head identifies three labels generated from the input itself:
identity, time reversal, and frequency reversal. This stays useful on DCASE2024
where development machine types have only one section. Its cross-entropy score
is standardized on normal training data and fused with standardized
reconstruction error; `self_supervised.fusion_weight` controls its contribution.

```bash
python -m aegis.run --dataset DCASE2020T2 --stage 3 \
  --machine-types fan --epochs 1 --max-train-batches 2 --max-test-files 10
```

## Ablation configurations

Ready-to-run YAML files are in `aegis/configs/`. They form one cumulative
ablation chain: Conv-AE, then frequency attention, then SSL score fusion:

```bash
python -m aegis.run --config aegis/configs/01_stage1_conv_ae.yaml --dataset DCASE2020T2
python -m aegis.run --config aegis/configs/02_stage2_frequency_attention.yaml --dataset DCASE2020T2
python -m aegis.run --config aegis/configs/03_stage3_full_aegis.yaml --dataset DCASE2020T2
```

Each file inherits common settings from `aegis/config.yaml`; its experiment
name also isolates checkpoints and results from the other runs.

## Comparison and ablation tables

After running all stages, normalize the untouched Dense-AE and STgram-MFN
results to the schema in `reference_results.example.csv`, then generate both
final tables:

```bash
python -m aegis.report --reference-csv path/to/baseline_results.csv
```

This writes `comparison.csv` (methods across both datasets) and `ablation.csv`
(all configured variants) under `aegis/outputs/reports/`.

Outputs are written below `aegis/outputs/<dataset>/<experiment_name>/`, including model
checkpoints, per-file anomaly scores, section metrics, and a cross-machine
summary. Use `--machine-types` to select a fixed six-machine subset for an
experiment; the official DCASE2024 development map currently contains seven.

Run the dependency-light model smoke test with:

```bash
python -m unittest discover -s aegis/tests -p "test_*_smoke.py" -v
```
