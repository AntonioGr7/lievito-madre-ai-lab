#!/usr/bin/env bash
# Fine-tune an encoder for PII token classification (BIO scheme). Two
# prepare scripts ship here — OpenPII (default) and Nemotron-PII —
# both produce the same subword-aligned BIO dataset shape.
#
# Run from anywhere (script cd's to repo root):
#   bash examples/pii/run.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CONFIG="examples/pii/configs/pii_mbert.yaml"

echo "============================================================"
echo "Step 1/2  Prepare dataset (downloads, tokenizes, aligns BIO)"
echo "============================================================"
python examples/pii/dataset/prepare_openpii.py

echo
echo "============================================================"
echo "Step 2/2  Train"
echo "============================================================"
python scripts/train_token_classification.py --config "$CONFIG"
