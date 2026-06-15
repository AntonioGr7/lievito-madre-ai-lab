# Synthetic Bi-Encoder Pairs from a Custom Corpus

End-to-end example: take a corpus of raw documents, run the synthetic-data pipeline (chunk → LLM query generation → heuristic + judge filter → doc-level split), and get back a `DatasetDict` of `(anchor, positive)` pairs. The pairs are produced here by the **bi_encoder_dataset** pipeline; train on them in the separate **embedding-bi-encoder** project (its `scripts/train_bi_encoder.py` / `scripts/mine_hard_negatives.py`).

The demo corpus is a 1k-row sample of SEC-EDGAR — pulled by [download_sec_edgar.py](dataset/download_sec_edgar.py). Any jsonl with `{id, text}` per line works the same way.

## Requirements

```bash
pip install -e ".[dev]"
export OPENAI_API_KEY=sk-...   # or set gen_llm.base_url in the config for local vLLM/Ollama
```

The default `gen_llm.model` is `gpt-4o-mini` and `max_documents` is capped at **20** — the whole demo costs a few cents. Drop `max_documents` to `null` once you're happy with the output.

## Run

```bash
bash examples/custom_pairs/run.sh
```

Or step by step:

```bash
# 1. Get the input corpus (jsonl with {id, text} per line).
python examples/custom_pairs/dataset/download_sec_edgar.py \
    --sample-size 1000 \
    --out-path data/raw/sec-edgar-1000.jsonl

# 2. Run the pipeline.
python scripts/generate_bi_encoder_pairs.py \
    --config examples/custom_pairs/configs/sec_edgar_pairs.yaml
```

Output lands in `data/processed/sec-edgar-pairs/` as a `DatasetDict` with `train/dev/test` splits — doc-level (chunks of the same document never leak across splits).

## Adapting to your own corpus

1. **Format your docs** as a jsonl with `{id, text}` per line, or save a HF dataset to disk with at least `id` and `text` columns.
2. **Copy this folder** to `examples/<your_name>/`.
3. **Edit `configs/<your_name>.yaml`**: set `input_path`, `output_dir`, and lift the `max_documents` cap when ready.
4. **Tune knobs** that matter for your data:
   - `chunking.chunk_tokens` — drop to 256 for short docs (tweets, abstracts); keep at 512 for filings/papers.
   - `query_gen.styles` — fewer styles = fewer queries per chunk = cheaper.
   - `multi_hop.enabled` — disable for single-hop-only (cheaper, narrower coverage).
   - `judge.enabled` — disable for ~50% LLM spend reduction at a noticeable quality cost.
5. **Switch to a local LLM** by setting `gen_llm.base_url: http://localhost:8000/v1` (vLLM, Ollama, etc.) and picking a stronger open-weight `gen_llm.model` — concurrency can go much higher than the 16 default.

## Layout

```
custom_pairs/
├── README.md
├── run.sh
├── dataset/
│   └── download_sec_edgar.py    # self-contained corpus downloader
└── configs/
    └── sec_edgar_pairs.yaml      # the worked-example config — copy + edit for your corpus
```
