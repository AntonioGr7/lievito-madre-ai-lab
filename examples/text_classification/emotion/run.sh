#!/usr/bin/env bash
# Fine-tune ModernBERT for 6-way emotion classification on dair-ai/emotion.
#
# Run from anywhere (script cd's to repo root):
#   bash examples/text_classification/emotion/run.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

CONFIG="examples/text_classification/emotion/configs/emotion_bert.yaml"

echo "============================================================"
echo "Step 1/2  Prepare dataset (downloads, tokenizes, saves to Arrow)"
echo "============================================================"
python examples/text_classification/emotion/dataset/prepare_emotion.py

echo
echo "============================================================"
echo "Step 2/2  Train"
echo "============================================================"
python scripts/text_classification/train_text_classification.py --config "$CONFIG"
