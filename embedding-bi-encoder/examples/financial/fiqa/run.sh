#!/usr/bin/env bash
# Fine-tune an Ettin-68m bi-encoder on FiQA-2018, then benchmark against
# the RTEB-finance English open subset (FinanceBench, HC3Finance, FinQA).
#
# Run from anywhere (script cd's to repo root):
#   bash examples/financial/fiqa/run.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

CONFIG="examples/financial/fiqa/configs/fiqa_ettin68m.yaml"
MODEL_DIR="outputs/bi_encoder_fiqa_ettin68m/exp_01/final"

echo "============================================================"
echo "Step 1/3  Build FiQA (anchor, positive) pairs from BeIR"
echo "============================================================"
python examples/financial/dataset/prepare_fiqa.py

echo
echo "============================================================"
echo "Step 2/3  Train the bi-encoder"
echo "============================================================"
python scripts/train_bi_encoder.py --config "$CONFIG"

echo
echo "============================================================"
echo "Step 3/3  Benchmark on RTEB-finance English open subset"
echo "============================================================"
python examples/financial/eval/eval_rteb_finance.py --model-dir "$MODEL_DIR"
