#!/usr/bin/env bash
# run_all.sh — Run the full 2-dataset × 3-config ablation suite sequentially.
#
# Each command corresponds to one row (or dataset copy) in the ablation table:
#
#  Dataset A (DCASE2020 T2)
#  ┌─────────────────────────────────────┬─────────────┬────────────┬──────────────────┐
#  │ Config                              │ Conv-AE     │ Freq-Attn  │ Cls-Fusion       │
#  ├─────────────────────────────────────┼─────────────┼────────────┼──────────────────┤
#  │ exp1_convae          (Row A-1)      │     ✓       │     ✗      │       ✗          │
#  │ exp2_convae_freqattn (Row A-2)      │     ✓       │     ✓      │       ✗          │
#  │ exp3_full_aegis      (Row A-3)      │     ✓       │     ✓      │       ✓          │
#  └─────────────────────────────────────┴─────────────┴────────────┴──────────────────┘
#
#  Dataset B (DCASE2024 T2) — same ablation rows B-1, B-2, B-3
#  (Note: classifier_fusion auto-disabled for B because dev data has 1 section/machine)
#
# Usage:
#   bash run_all.sh
#   # or with GPU:
#   bash run_all.sh --device cuda
#
# Additional CLI options (--epochs, --batch-size, etc.) are forwarded as-is
# to every python invocation.

set -euo pipefail

EXTRA_ARGS="$*"

echo "=========================================================="
echo "AEGIS ablation suite — 6 experiments"
echo "=========================================================="

# ---------- Dataset A (DCASE2020 T2) ----------

echo ""
echo "[A-1] Conv-AE baseline  (DCASE2020T2)"
python -m aegis.train --config configs/exp1_convae.yaml          $EXTRA_ARGS

echo ""
echo "[A-2] +Frequency Attention  (DCASE2020T2)"
python -m aegis.train --config configs/exp2_convae_freqattn.yaml $EXTRA_ARGS

echo ""
echo "[A-3] +Classifier Fusion — Full AEGIS  (DCASE2020T2)"
python -m aegis.train --config configs/exp3_full_aegis.yaml      $EXTRA_ARGS

# ---------- Dataset B (DCASE2024 T2) ----------

echo ""
echo "[B-1] Conv-AE baseline  (DCASE2024T2)"
python -m aegis.train --config configs/exp1_convae_b.yaml          $EXTRA_ARGS

echo ""
echo "[B-2] +Frequency Attention  (DCASE2024T2)"
python -m aegis.train --config configs/exp2_convae_freqattn_b.yaml $EXTRA_ARGS

echo ""
echo "[B-3] +Classifier Fusion — Full AEGIS  (DCASE2024T2)"
python -m aegis.train --config configs/exp3_full_aegis_b.yaml      $EXTRA_ARGS

echo ""
echo "=========================================================="
echo "All experiments finished.  Results are in outputs/ and outputs_b/."
echo "=========================================================="
