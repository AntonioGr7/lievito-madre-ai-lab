#!/usr/bin/env bash
# Worked example: build (anchor, positive) pairs from a custom corpus using
# the synthetic-data pipeline (chunk → LLM query gen → filter → split). The
# corpus here is a small SEC-EDGAR sample, but any jsonl of {id, text} works.
#
# Requires OPENAI_API_KEY (or set gen_llm.base_url in the config to point at
# a local vLLM/Ollama). The default `gen_llm.model: gpt-4o-mini` + the
# `max_documents: 20` cap keep the demo's LLM spend at a few cents.
#
# Run from anywhere (script cd's to the pipelines project root):
#   bash examples/custom_pairs/run.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CONFIG="examples/custom_pairs/configs/sec_edgar_pairs.yaml"
INPUT_DOCS="data/raw/sec-edgar-1000.jsonl"
PAIRS_DIR="data/processed/sec-edgar-pairs"

echo "============================================================"
echo "Step 1/2  Download a small SEC-EDGAR sample (skipped if cached)"
echo "============================================================"
if [[ -f "$INPUT_DOCS" ]]; then
    echo "[skip] $INPUT_DOCS exists — reusing."
else
    python examples/custom_pairs/dataset/download_sec_edgar.py \
        --sample-size 1000 \
        --out-path "$INPUT_DOCS"
fi

echo
echo "============================================================"
echo "Step 2/2  Generate (anchor, positive) pairs via the pipeline"
echo "============================================================"
python scripts/generate_bi_encoder_pairs.py --config "$CONFIG"

echo
echo "Pairs written to $PAIRS_DIR (DatasetDict with train/dev/test splits)."
echo "Feed them into the embedding-bi-encoder project, e.g. (from that project):"
echo "  python scripts/train_bi_encoder.py \\"
echo "      --config <your-train-config>.yaml \\"
echo "      --processed-dir <path-to>/sec-edgar-pairs"
