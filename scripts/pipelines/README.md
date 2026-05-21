# Pipelines — synthetic data generation

Module: [lievito_madre_ai_lab/pipelines/](../../lievito_madre_ai_lab/pipelines/)

LLM-driven synthetic data generation. The first pipeline here builds
`(anchor, positive)` training pairs from a corpus of domain documents — the
canonical first step before bi-encoder fine-tuning when you have documents
but no labelled queries.

## Install

```bash
pip install -e ".[pipelines]"
export OPENAI_API_KEY=sk-...
```

Set `OPENAI_API_KEY` (or point `base_url` in the YAML at a local vLLM/Ollama
endpoint — the OpenAI SDK is wire-compatible with both).

## Pipeline: bi-encoder pairs

End-to-end: raw documents → token-aware chunks → multi-style LLM-generated
queries (single-chunk + **intra-doc multi-hop**) → heuristic + LLM-judge
filtering → doc-level train/dev/test split → HF `DatasetDict` in the
`(anchor, positive)` shape that
[`scripts/embedding_bi_encoder/train_bi_encoder.py`](../embedding_bi_encoder/train_bi_encoder.py)
already consumes.

### Single-hop vs multi-hop queries

The pipeline generates two flavours of training pair in one run:

- **Single-hop**: one query per chunk, in N styles (natural question,
  keyword, paraphrase, messy). Teaches the model the most common pattern
  — a query that one passage can answer.
- **Multi-hop (intra-document)**: one query per adjacent K-chunk window,
  emitted as K rows `(Q, C_i)` so the model learns Q close to every chunk
  in the window. Teaches the model that a single query can match multiple
  complementary passages — important for technical / structured docs
  where one answer spans sections.

Multi-hop is on by default (`multi_hop.k=2`, `multi_hop.target_share=0.2`,
i.e. ~20% of the generated pairs come from multi-hop groups — "2 out of
every 10"). Disable with `--no-multi-hop` to fall back to single-hop only.
The pipeline converts `target_share` into an absolute window count using
the corpus stats, so the share stays stable whether you change the number
of styles, chunk size, or k. Cross-document multi-hop is intentionally out
of scope for this v1 — it needs chunk clustering and co-positive-aware
hard-negative mining, and is best added once the single + intra-doc setup
has been validated on your domain.

### Resume after a crash

Every LLM call is checkpointed per-batch under `output_dir/_checkpoints/`:

```
output_dir/
  _checkpoints/
    single_hop.jsonl     # keyed by chunk_id
    multi_hop.jsonl      # keyed by group_id
    judge.jsonl          # keyed by row_id (anchor::chunk_id or group::chunk)
  chunks.jsonl
  raw_pairs.jsonl
  preprocessing.json
  train/  dev/  test/
```

If the process dies mid-run, restart with the same command — the next run
loads the checkpoint, skips completed work, and continues. No flag is
needed; resumption is the default.

Notes:

- **API errors** (timeouts, 5xx, rate-limit floods) leave the affected
  chunk uncheckpointed so the next run retries it. **Parse failures** /
  refusals ARE checkpointed (with an empty rows list) so we don't retry a
  doomed call forever; use `--fresh` to invalidate.
- The judge stage stores raw scores. You can tweak `judge.min_score`
  between runs and re-run without paying for any new LLM calls — only the
  threshold is re-applied to the persisted scores.
- Pass `--fresh` to delete the checkpoint directory and rebuild from
  scratch. Useful when you've changed the prompt or model and want every
  call redone.

### Quickstart

```bash
# 1. Stage your corpus as JSONL — one {id, text} per line.
#    HF DatasetDict on disk also works; set input_format: hf_dataset in the YAML.
head -1 data/raw/my-corpus.jsonl
# {"id": "doc-0001", "text": "..."}

# 2. Dry-run on 50 docs without the judge to sanity-check the prompts.
python scripts/pipelines/generate_bi_encoder_pairs.py \
    --config examples/embedding_bi_encoder/custom_pairs/configs/sec_edgar_pairs.yaml \
    --max-documents 50 \
    --no-judge \
    --output data/processed/my-pairs-dryrun

# 3. Inspect the intermediate artefacts:
#    - data/processed/my-pairs-dryrun/chunks.jsonl       (post-chunking)
#    - data/processed/my-pairs-dryrun/raw_pairs.jsonl    (pre-filter)
#    - data/processed/my-pairs-dryrun/preprocessing.json (every knob recorded)
#    - data/processed/my-pairs-dryrun/{train,dev,test}/  (final DatasetDict)

# 4. Production run on the full corpus.
python scripts/pipelines/generate_bi_encoder_pairs.py \
    --config examples/embedding_bi_encoder/custom_pairs/configs/sec_edgar_pairs.yaml \
    --input data/raw/my-corpus.jsonl \
    --output data/processed/my-pairs
```

### From pairs to a trained model

The pair dataset drops straight into the existing bi-encoder workflow:

```bash
# Stage 1 — train on pairs with in-batch negatives (MNRL).
python scripts/embedding_bi_encoder/train_bi_encoder.py \
    --config examples/embedding_bi_encoder/smoke/configs/smoke_r1.yaml \
    --processed-dir data/processed/my-pairs

# Stage 2 — mine hard negatives with the freshly-tuned model.
python scripts/embedding_bi_encoder/mine_hard_negatives.py \
    --input-dataset data/processed/my-pairs \
    --output-dir data/processed/my-pairs-mined \
    --strategy dense \
    --retriever <your-fine-tuned-model> \
    --cross-encoder BAAI/bge-reranker-v2-m3

# Stage 3 — fine-tune again on the triplet dataset for the final model.
```

Why train on pairs *first* and only then mine: hard-negative mining with an
off-the-shelf retriever produces weak negatives in a domain it doesn't know
yet. Train once on pairs, then mine — your `mine_hard_negatives.py` will
surface much sharper negatives once the retriever has been domain-adapted.

## Config knobs that matter most

[examples/embedding_bi_encoder/custom_pairs/configs/sec_edgar_pairs.yaml](../../examples/embedding_bi_encoder/custom_pairs/configs/sec_edgar_pairs.yaml)

| Knob | Default | When to change |
| --- | --- | --- |
| `chunking.chunk_tokens` | 256 | 128 for FAQ-style short answers; 512 for technical docs where context spans paragraphs. |
| `query_gen.styles` | 3 styles | Drop styles that don't match your inference traffic. Each style → one LLM-emitted query per chunk. |
| `multi_hop.enabled` / `multi_hop.k` / `multi_hop.target_share` | true / 2 / 0.2 | Disable for pure single-hop training. Bump `k` to 3 for harder reasoning patterns (costs more tokens per call). Raise `target_share` (e.g. 0.3-0.4) if your inference traffic is multi-hop heavy. |
| `judge.min_score` | 4 (of 5) | 3 keeps more pairs but admits noise; 5 is too strict for most domains. |
| `filter.max_verbatim_overlap` | 6 words | Lower for technical jargon where reuse is hard to avoid; higher in narrative domains. |
| `gen_llm.model` | gpt-4o-mini | gpt-4o for higher-quality queries; a local 70B for full data sovereignty (set `base_url`). |
| `judge_llm.model` | gpt-4o-mini | Worth upgrading to gpt-4o here — judge errors are more expensive than generator errors. |
| `train_ratio` / `dev_ratio` / `test_ratio` | 0.9 / 0.05 / 0.05 | Splits are doc-level (never row-level) so chunks of one doc can't leak across splits. |

## Cost budgeting

Per 10k documents at default settings (256-token chunks, 4 styles, judge on,
gpt-4o-mini for both stages):

- ~25k chunks (depends on doc length)
- ~25k generation calls (one batched call per chunk, 4 queries each)
- ~100k judge calls (one per candidate pair)
- ~125k total LLM calls → roughly $5–$15 of gpt-4o-mini at current pricing.

`--max-documents 50 --no-judge` is your friend for iteration — ~$0.05 to
validate the prompts, chunking and parsing end-to-end before paying for the
production run.

## Layout

```
lievito_madre_ai_lab/pipelines/
├── llm/
│   ├── base.py              # LLMClient protocol + request/response types
│   └── providers.py         # OpenAI async client (covers vLLM/Ollama via base_url)
└── embedding/
    ├── synthetic/
    │   ├── chunking.py          # tiktoken-based token-aware splitter
    │   ├── checkpoint.py        # append-only JSONL store for resumable LLM stages
    │   ├── query_generation.py  # one LLM call per chunk, multi-style JSON output
    │   ├── multi_hop.py         # adjacent K-chunk windows → multi-hop queries
    │   ├── filtering.py         # heuristic dedup + verbatim-overlap + LLM judge
    │   └── pipeline.py          # end-to-end orchestration → DatasetDict
    └── prompts/
        ├── query_gen.yaml             # editable single-chunk generation prompt
        ├── multi_hop_query_gen.yaml   # editable multi-chunk generation prompt
        └── judge.yaml                 # editable judge rubric

scripts/pipelines/
└── generate_bi_encoder_pairs.py   # CLI entry point

examples/embedding_bi_encoder/custom_pairs/
├── README.md
├── run.sh                          # download corpus → run pipeline
└── configs/
    └── sec_edgar_pairs.yaml        # worked-example config; copy + edit for your corpus
```
