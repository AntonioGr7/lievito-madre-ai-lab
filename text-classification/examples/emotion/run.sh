#!/usr/bin/env bash
# Fine-tune ModernBERT for 6-way emotion classification on dair-ai/emotion.
#
# Run from anywhere (script cd's to repo root):
#   bash examples/emotion/run.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CONFIG="examples/emotion/configs/emotion_bert.yaml"

echo "============================================================"
echo "Step 1/2  Prepare dataset (downloads, tokenizes, saves to Arrow)"
echo "============================================================"
python examples/emotion/dataset/prepare_emotion.py

echo
echo "============================================================"
echo "Step 2/2  Train"
echo "============================================================"
python scripts/train_text_classification.py --config "$CONFIG"
