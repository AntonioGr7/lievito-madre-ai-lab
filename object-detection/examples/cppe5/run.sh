#!/usr/bin/env bash
# Fine-tune a detector on CPPE-5 (medical PPE).
#
#   bash examples/cppe5/run.sh
#   CONFIG=examples/cppe5/configs/rtdetrv2_r50.yaml bash examples/cppe5/run.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CONFIG="${CONFIG:-examples/cppe5/configs/dfine_x.yaml}"

echo "============================================================"
echo "Step 1/3  Build the COCO-format dataset from CPPE-5"
echo "============================================================"
python examples/cppe5/dataset/prepare_cppe5.py --out-dir data/processed/cppe5

echo
echo "============================================================"
echo "Step 2/3  Pretrained baseline (pipeline sanity check)"
echo "============================================================"
python scripts/baseline_pretrained.py --config "$CONFIG" --max-test-samples 100 || true

echo
echo "============================================================"
echo "Step 3/3  Fine-tune ($CONFIG)"
echo "============================================================"
python scripts/train_detector.py --config "$CONFIG"
