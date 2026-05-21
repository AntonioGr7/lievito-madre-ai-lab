#!/usr/bin/env bash
# Fine-tune GLiNER for PII entity extraction. Two prepare scripts ship
# here — OpenPII (default) and Nemotron-PII — both produce the same
# char-offset dataset shape the trainer consumes.
#
# Run from anywhere (script cd's to repo root):
#   bash examples/gliner_entity_extraction/pii/run.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

CONFIG="examples/gliner_entity_extraction/pii/configs/pii_gliner.yaml"

echo "============================================================"
echo "Step 1/2  Build the char-offset dataset from OpenPII"
echo "============================================================"
python examples/gliner_entity_extraction/pii/prepare_openpii.py \
    --out-dir data/processed/pii-gliner

echo
echo "============================================================"
echo "Step 2/2  Fine-tune GLiNER"
echo "============================================================"
python scripts/gliner_entity_extraction/train_gliner.py --config "$CONFIG"
