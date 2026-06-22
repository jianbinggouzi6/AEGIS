# AEGIS — Acoustic Equipment anomaly detection with Guided Iterative Spectral attention

AEGIS is a research codebase for industrial sound anomaly detection built on top of
the [DCASE Task 2](https://dcase.community/challenge2023/task2-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring) benchmark.  
All model code lives in `aegis/`; two reference implementations are kept **unmodified** as git submodules.

## Repository layout

```
AEGIS/
├── aegis/                          # AEGIS source code (this is the only code you touch)
│   ├── models.py                   # Conv-AE + Freq-Attn + Section Classifier
│   ├── engine.py                   # Trainer, per-section calibration, scoring
│   ├── train.py                    # Entry point: python -m aegis.train
│   ├── data.py                     # Adapter around the unmodified baseline loader
│   ├── metrics.py                  # AUC / pAUC / F1 (mirrors baseline evaluation)
│   ├── report.py                   # Cross-dataset comparison + ablation table builder
│   └── requirements.txt
├── configs/                        # 6 ablation YAML files (3 settings × 2 datasets)
│   ├── exp1_convae.yaml
│   ├── exp2_convae_freqattn.yaml
│   ├── exp3_full_aegis.yaml
│   ├── exp1_convae_b.yaml
│   ├── exp2_convae_freqattn_b.yaml
│   └── exp3_full_aegis_b.yaml
├── run_all.sh                      # Run all 6 experiments sequentially
├── dcase2023_task2_baseline_ae/    # Official Dense-AE (git submodule, unmodified)
├── STgram-MFN/                     # STgram-MFN strong baseline (git submodule, unmodified)
├── CODEX.md                        # Research design notes
└── README.md                       # This file
```

---

## 1  Environment

```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:<user>/AEGIS.git
cd AEGIS

# Install dependencies
pip install -r aegis/requirements.txt
```

---

## 2  Data preparation

Place the **development** data inside the baseline repo's `data/` directory,
following the path convention the official loader expects:

```
dcase2023_task2_baseline_ae/data/
├── dcase2020t2/dev_data/raw/<machine_type>/train/   ← Dataset A training WAVs
│                                           test/
├── dcase2024t2/dev_data/raw/<machine_type>/train/   ← Dataset B training WAVs
│                                           test/
```

The loader caches mel features as `.pickle` files on first run.

**Dataset A** — DCASE2020 Task 2  
Machines: `fan  pump  slider  valve  ToyCar  ToyConveyor`

**Dataset B** — DCASE2024 Task 2  
Machines: `bearing  fan  gearbox  slider  valve  ToyCar  ToyTrain`

> **Note:** DCASE2024T2 dev data provides only one section per machine,
> so `classifier_fusion` is automatically disabled with a warning for Dataset B experiments.

---

## 3  Model switches (ablation dimensions)

All three boolean flags in `model:` of the YAML file control which components are active:

| Switch | Key in YAML | What it does |
|--------|-------------|--------------|
| `conv_ae` | `model.conv_ae` | 2D Conv-AE backbone — always `true` |
| `freq_attention` | `model.freq_attention` | FreqAxisAttention gate after the first encoder block |
| `classifier_fusion` | `model.classifier_fusion` | Section-ID classifier head + z-score fusion |

Additional tunables in `classifier:`:

| Key | Default | Meaning |
|-----|---------|---------|
| `num_classes` | `0` (auto) | Number of sections; `0` detects from data |
| `loss_type` | `"ce"` | `"ce"` for cross-entropy, `"arcface"` for ArcFace |
| `lambda_cls` | `0.2` | λ — weight of classifier loss in joint training |

And in `fusion:`:

| Key | Default | Meaning |
|-----|---------|---------|
| `weight` | `0.3` | w — `fused = z_recon + w * z_cls` |

Staged training (train AE first, then fine-tune classifier):

```yaml
training:
  staged:     true
  ae_epochs:  30    # AE-only phase
  clf_epochs: 20    # classifier fine-tuning phase (encoder frozen)
```

---

## 4  Running experiments

### Single experiment

```bash
python -m aegis.train --config configs/exp3_full_aegis.yaml
```

Override any setting from the command line:

```bash
python -m aegis.train --config configs/exp3_full_aegis.yaml \
    --device cuda --epochs 100 --batch-size 256
```

Restrict to a subset of machines:

```bash
python -m aegis.train --config configs/exp1_convae.yaml \
    --machine-types fan pump
```

### Full ablation suite (all 6 configs)

```bash
bash run_all.sh                    # CPU
bash run_all.sh --device cuda      # GPU
```

The six commands and their ablation-table rows:

| Shell step | Config | Dataset | Conv-AE | Freq-Attn | Cls-Fusion |
|-----------|--------|---------|:-------:|:---------:|:----------:|
| A-1 | `exp1_convae.yaml` | A | ✓ | ✗ | ✗ |
| A-2 | `exp2_convae_freqattn.yaml` | A | ✓ | ✓ | ✗ |
| A-3 | `exp3_full_aegis.yaml` | A | ✓ | ✓ | ✓ |
| B-1 | `exp1_convae_b.yaml` | B | ✓ | ✗ | ✗ |
| B-2 | `exp2_convae_freqattn_b.yaml` | B | ✓ | ✓ | ✗ |
| B-3 | `exp3_full_aegis_b.yaml` | B | ✓ | ✓ | ✓ |

---

## 5  Outputs

Each run writes to `outputs/<dataset>/<experiment_name>/`:

```
outputs/DCASE2020T2/exp3_full_aegis/
├── config.yaml                        ← full resolved config (reproducibility)
├── result.csv                         ← per-machine AUC/pAUC/F1 + arithmetic mean
└── <machine_type>/
    ├── model.pt                       ← checkpoint
    ├── history.json                   ← epoch-level train/valid losses
    ├── anomaly_score_section_<id>.csv ← per-file scores
    ├── section_metrics.csv            ← per-section AUC/pAUC/F1
    └── summary.csv                    ← machine-level aggregate
```

`result.csv` columns: `dataset, machine_type, experiment, auc, pauc, precision, recall, f1`  
The last row is the **arithmetic mean** across all machines — paste this row directly into the paper table.

---

## 6  Comparison and ablation tables

After running all experiments and collecting baseline results, generate the
final comparison and ablation CSVs:

```bash
python -m aegis.report \
    --output-dir outputs \
    --report-dir outputs/reports \
    --reference-csv path/to/baseline_dense_ae_results.csv \
    --reference-csv path/to/stgram_mfn_results.csv
```

Reference CSVs must have columns: `dataset, method, machine_type, auc, pauc, f1`.

---

## 7  Reproducing from GitHub

```bash
git clone --recurse-submodules git@github.com:<user>/AEGIS.git
cd AEGIS
pip install -r aegis/requirements.txt
# Place datasets as described in section 2
bash run_all.sh --device cuda
```

---

## 8  Citation / acknowledgements

This repository builds on:

- **DCASE2023 Task 2 Baseline AE** (NTT Media Intelligence Laboratories):  
  `dcase2023_task2_baseline_ae/` — kept unmodified.
- **STgram-MFN** (Liu et al., 2022):  
  `STgram-MFN/` — kept unmodified.
