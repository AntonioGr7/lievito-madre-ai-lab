#!/usr/bin/env bash
# Fine-tune a VLM for receipt -> JSON extraction on CORD-v2.
# This is the generic text-target example (vlm.task: text) — same engine as the
# grounding example, different output format + metric.
#
#   bash examples/json_extraction/run.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CONFIG="${CONFIG:-examples/json_extraction/configs/cord_qwen25vl_3b.yaml}"

echo "============================================================"
echo "Step 1/3  Build the text-target dataset from CORD-v2"
echo "============================================================"
python examples/json_extraction/dataset/prepare_cord.py --out-dir data/processed/cord

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
