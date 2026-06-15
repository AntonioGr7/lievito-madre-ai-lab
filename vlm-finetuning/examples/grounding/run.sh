#!/usr/bin/env bash
# Fine-tune a VLM for grounded PPE detection on CPPE-5.
#
# Run from anywhere (script cd's to repo root):
#   bash examples/grounding/run.sh
#
# Override the config:
#   CONFIG=examples/grounding/configs/cppe5_qwen25vl_3b_qlora.yaml bash examples/grounding/run.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CONFIG="${CONFIG:-examples/grounding/configs/cppe5_qwen25vl_3b.yaml}"

echo "============================================================"
echo "Step 1/3  Build the grounding dataset from CPPE-5"
echo "============================================================"
python examples/grounding/dataset/prepare_cppe5.py --out-dir data/processed/cppe5

echo
echo "============================================================"
echo "Step 2/3  Zero-shot baseline (the bar fine-tuning must clear)"
echo "============================================================"
python scripts/baseline_zeroshot.py --config "$CONFIG" --max-test-samples 100 || true

echo
echo "============================================================"
echo "Step 3/3  Fine-tune ($CONFIG)"
echo "============================================================"
python scripts/train_vlm.py --config "$CONFIG"
