#!/usr/bin/env bash
# Smoke test for Recipe 1 (pair data → MNRL).
# Builds a 100-row synthetic dataset, trains for 1 epoch, then verifies
# the saved model loads and produces sensible embeddings.
#
# Run from anywhere (script cd's to repo root):
#   bash examples/embedding_bi_encoder/smoke/run.sh
#
# Total runtime: ~1-2 min on CPU, ~30s on a recent GPU.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

DATA_DIR="data/processed/smoke-bi-encoder"
OUT_DIR="outputs/smoke_bi_encoder_r1/final"
CONFIG="examples/embedding_bi_encoder/smoke/configs/smoke_r1.yaml"

echo "============================================================"
echo "Step 1/3  Build the synthetic dataset"
echo "============================================================"
python examples/embedding_bi_encoder/smoke/prepare_smoke.py \
    --out-dir "$DATA_DIR" \
    --n-train 100 \
    --n-val 24 \
    --n-test 24

echo
echo "============================================================"
echo "Step 2/3  Train the bi-encoder (1 epoch, batch 4)"
echo "============================================================"
python scripts/embedding_bi_encoder/train_bi_encoder.py --config "$CONFIG"

echo
echo "============================================================"
echo "Step 3/3  Verify the saved model loads + encodes"
echo "============================================================"
python -c "
import sys
from pathlib import Path
from lievito_madre_ai_lab.finetuning.embedding.bi_encoder.serve import BiEncoderPredictor

out_dir = '$OUT_DIR'
if not Path(out_dir).exists():
    sys.exit(f'final/ dir missing at {out_dir} — training did not persist the model')

predictor = BiEncoderPredictor(out_dir, use_compile=False, warmup_steps=0)
texts = [
    'How do vector databases work?',
    'What is a vector database?',
    'Mount Everest is the tallest mountain on Earth.',
]
emb = predictor.encode(texts)
sims = predictor.similarity([texts[0]], texts)

print()
print('Embeddings shape:', tuple(emb.shape))
print('Self-similarity row:', sims[0].tolist())

# Recipe-1 sanity check: anchor[0] should be more similar to the paraphrase
# (texts[1]) than to the unrelated text (texts[2]).
sim_paraphrase = float(sims[0, 1])
sim_unrelated  = float(sims[0, 2])
print(f'sim(anchor, paraphrase) = {sim_paraphrase:.4f}')
print(f'sim(anchor, unrelated)  = {sim_unrelated:.4f}')

if sim_paraphrase <= sim_unrelated:
    sys.exit(
        f'SMOKE TEST FAILED — paraphrase should outscore the unrelated text. '
        f'Got paraphrase={sim_paraphrase:.4f} <= unrelated={sim_unrelated:.4f}. '
        f'The training loop ran, but the model did not learn the task on the '
        f'synthetic data. Check for tokenizer / loss config drift.'
    )
print()
print('SMOKE TEST PASSED — paraphrase outscored the unrelated text.')
"
