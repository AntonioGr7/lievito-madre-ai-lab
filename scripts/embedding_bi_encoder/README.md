# Bi-Encoder Fine-Tuning

Fine-tuning a [`sentence-transformers`](https://www.sbert.net/) bi-encoder for semantic search, retrieval, and similarity. Driven by the same `TrainConfig` YAML the encoder pipelines use, plus a `bi_encoder:` block for loss, sampler, mining metadata, Matryoshka (1D / 2D), and instruction prompts.

```bash
pip install -e ".[embedding]"
```

The default backbone is [`nomic-ai/modernbert-embed-base`](https://huggingface.co/nomic-ai/modernbert-embed-base) — ModernBERT-based, 8k context out of the box.

---

## Decision guide — what should I run?

Pick the row that matches your starting point. Each cell links to the recipe below.

**GradCache is on by default** in every shipped YAML — see [the reference section](#gradcache-gradient-caching-the-memory-trick). The lab assumes you want to train at the batch size that produces good embeddings, not the batch size your GPU happens to fit.

| You have… | You want… | Recipe |
|---|---|---|
| `(anchor, positive)` pairs, no time to mine | Working baseline today | [Recipe 1 — Quick MNRL baseline](#recipe-1--quick-mnrl-baseline) |
| Same, want stronger retrieval | Hard-negative mined model | [Recipe 2 — + Hard negatives](#recipe-2--hard-negatives-the-single-biggest-lever) |
| Same, want SOTA quality | Cross-encoder distilled | [Recipe 3 — + Listwise KL distillation](#recipe-3--listwise-kl-distillation) |
| Same, plus a tight serve budget | One model, many dims | [Recipe 4 — Matryoshka (1D)](#recipe-4--matryoshka-1d-one-model-many-dims) |
| Same, plus heterogeneous serve hardware | Trade dim *and* depth | [Recipe 5 — Matryoshka 2D](#recipe-5--matryoshka-2d-dim--depth) |
| Heterogeneous failure modes (lexical + semantic) | Maximum negative diversity | [Recipe 6 — Ensemble mining (dense + BM25)](#recipe-6--ensemble-mining-dense--bm25) |
| Small gold set + a big unlabelled pool | Bootstrap with silver labels | [Recipe 7 — Silver labels from a cross-encoder](#recipe-7--silver-labels-from-a-cross-encoder) |
| You're using E5 / Nomic / BGE-M3 | Don't drift off the protocol | [Recipe 8 — Instruction prompts](#recipe-8--instruction-prompts) |

Stack the recipes. The full SOTA setup is **2 + 3 + 5 + 6 + 8** in one run.

**Before you start**, sanity-check the install with the smoke test — builds a synthetic 100-row dataset, runs Recipe 1 end-to-end in ~2 minutes, and verifies the saved model can tell a paraphrase from an unrelated sentence:

```bash
bash scripts/embedding_bi_encoder/smoke_test.sh
```

If that passes, the pipeline is wired correctly and you can move to the recipes below with real data.

---

## Recipes

### Recipe 1 — Quick MNRL baseline

**Use when**: just getting the pipeline working, or your corpus is small enough that in-batch negatives are enough signal.

```bash
# 1. Build a DatasetDict on disk with two string columns: anchor, positive.
# 2. Train.
python scripts/embedding_bi_encoder/train_bi_encoder.py \
    --config configs/embedding/bi_encoder/modernbert_mnrl.yaml
```

The trainer auto-picks `MultipleNegativesRankingLoss` with `NO_DUPLICATES` batch sampling. **GradCache is already on** in the shipped YAML — the loss is transparently swapped to its cached variant, so `per_device_train_batch_size: 256` is the default and memory stays bounded by `mini_batch_size: 32`.

If you OOM, lower `mini_batch_size` first (peak memory drops). Only lower `per_device_train_batch_size` once `mini_batch_size` is already tiny — shrinking the logical batch directly hurts model quality.

See [the GradCache reference](#gradcache-gradient-caching-the-memory-trick) for the full story.

---

### Recipe 2 — + Hard negatives (the single biggest lever)

**Use when**: you've seen Recipe 1 work and want quality up. Hard negatives are the highest-impact addition; in-batch negatives only test the model against *random* documents.

```bash
# 1. Mine 5 hard negatives per anchor with a strong dense retriever +
#    cross-encoder filter. The 60%-of-positive rule lives in --relative-margin 0.4.
python scripts/embedding_bi_encoder/mine_hard_negatives.py \
    --input-dataset data/processed/my-pairs \
    --output-dir    data/processed/my-pairs-mined \
    --strategy dense \
    --retriever     BAAI/bge-large-en-v1.5 \
    --cross-encoder BAAI/bge-reranker-v2-m3 \
    --num-negatives 5 \
    --relative-margin 0.4 \
    --range-min 10 --range-max 100

# 2. Train. The trainer auto-detects the multi_negative shape and uses
#    every mined negative as a hard sample in the MNRL in-batch loss.
python scripts/embedding_bi_encoder/train_bi_encoder.py \
    --config configs/embedding/bi_encoder/modernbert_mnrl.yaml
```

Point `processed_dir` in the YAML at `data/processed/my-pairs-mined`.

**The retriever should not be the model you're fine-tuning.** Use a strong general embedder (BGE-large, E5-large) for mining. Mining with the student model is circular and produces weak negatives.

---

### Recipe 3 — + Listwise KL distillation

**Use when**: Recipe 2 isn't enough, and you want to extract every drop of signal from your cross-encoder. KL distillation teaches the *full ranking shape*, not just the order — it outperforms margin-based losses in the recipe's experiments.

```bash
# 1. Mine as in Recipe 2.

# 2. Score every (anchor, candidate) pair with the cross-encoder. This
#    appends a `label` column (list of teacher scores).
python scripts/embedding_bi_encoder/score_with_cross_encoder.py \
    --input-dataset data/processed/my-pairs-mined \
    --output-dir    data/processed/my-pairs-distill \
    --cross-encoder BAAI/bge-reranker-v2-m3

# 3. Switch the loss in the YAML to DistillKLDivLoss, then train.
#    bi_encoder.loss.name: DistillKLDivLoss
python scripts/embedding_bi_encoder/train_bi_encoder.py \
    --config configs/embedding/bi_encoder/modernbert_mnrl.yaml
```

The trainer hard-errors if you point `DistillKLDivLoss` at a non-distill dataset (or vice versa). The teacher must be **stronger** than the student bi-encoder — otherwise distillation just imitates a weaker model.

**Recommended teachers (2026)**:

| Teacher | Speed | Domain |
|---|---|---|
| `BAAI/bge-reranker-v2-m3` | medium | general English + multilingual |
| `mixedbread-ai/mxbai-rerank-large-v2` | medium | strong English |
| `Alibaba-NLP/gte-multilingual-reranker-base` | fast | multilingual |

---

### Recipe 4 — Matryoshka (1D): one model, many dims

**Use when**: the same model needs to serve different latency/cost tiers (e.g. a 64-dim ANN index for first-stage retrieval and a 768-dim reranker pass).

```yaml
# In your YAML — add to the bi_encoder block:
bi_encoder:
  matryoshka:
    enabled: true
    mode: "1d"
    dims: [768, 512, 256, 128, 64]   # always include the native dim first
    weights: null                      # equal weighting
```

```bash
python scripts/embedding_bi_encoder/train_bi_encoder.py --config <your>.yaml
```

At inference, truncate to any trained dim. The predictor renormalises so dot product = cosine similarity at the truncated dim:

```python
predictor = BiEncoderPredictor("outputs/<run>/final")
emb_768 = predictor.encode(texts)                    # full quality
emb_64  = predictor.encode(texts, truncate_dim=64)   # ~12× smaller, ~12× faster downstream
```

The smallest dim lands in `preprocessing.json` as `default_truncate_dim` so storage-tight serve setups pick it up automatically.

---

### Recipe 5 — Matryoshka 2D: dim × depth

**Use when**: you want to serve from heterogeneous hardware — strong GPUs run the full model at full dim, edge devices run a shallow truncated version of the *same* artifact.

```yaml
bi_encoder:
  matryoshka:
    enabled: true
    mode: "2d"                          # depth × dim
    dims: [768, 512, 256, 128, 64]
    n_layers_per_step: 1                # train every layer; 4 = train every 4th (cheaper)
    last_layer_weight: 1.0
    prior_layers_weight: 1.0
    kl_div_weight: 1.0                  # KL term aligning earlier layers' outputs with the last
    kl_temperature: 0.3
```

At inference, drop encoder layers **before** loading the predictor (it's applied at load time):

```python
# Full model, full dim
fast_full = BiEncoderPredictor("outputs/<run>/final")

# Half-depth, quarter-dim — for a tiny edge serve
edge = BiEncoderPredictor(
    "outputs/<run>/final",
    truncate_layers=6,         # keep only the first 6 transformer layers
    truncate_dim=128,
)
```

The layer-truncation helper supports BERT / RoBERTa / DeBERTa / ModernBERT / DistilBERT layer-list paths out of the box ([model.py](../../lievito_madre_ai_lab/embedding/bi_encoder/model.py)).

**Layer truncation only works on Matryoshka2d / AdaptiveLayer-trained models.** Truncating a normally-trained model produces near-random embeddings — the earlier layers were never asked to emit useful sentence vectors.

---

### Recipe 6 — Ensemble mining (dense + BM25)

**Use when**: a single retriever's negatives are all of the same type. Dense retrievers flag *semantically*-close-but-wrong passages; BM25 flags *lexically*-close-but-wrong ones. Unioning the two and re-filtering with a cross-encoder produces a strictly broader hardness distribution.

```bash
python scripts/embedding_bi_encoder/mine_hard_negatives.py \
    --input-dataset data/processed/my-pairs \
    --output-dir    data/processed/my-pairs-ensemble \
    --strategy ensemble \
    --retriever     BAAI/bge-large-en-v1.5 \
    --retriever     intfloat/e5-large-v2 \
    --bm25 \
    --cross-encoder BAAI/bge-reranker-v2-m3 \
    --num-negatives 5 \
    --relative-margin 0.4
```

How it works ([mining.py](../../lievito_madre_ai_lab/embedding/bi_encoder/mining.py)):

1. Each retriever (+ BM25) mines `2 × num_negatives` candidates per anchor — over-mining gives headroom for dedup.
2. Per-row candidate lists are unioned across sources and deduped by negative text.
3. The cross-encoder filter (60%-of-positive rule) runs once on the union — cheaper than per-source filtering, and consistent.
4. Top `num_negatives` per row are kept.

You can also run BM25 alone (`--strategy bm25`) — useful when your domain has strong lexical signal (legal, medical, code).

---

### Recipe 7 — Silver labels from a cross-encoder

**Use when**: you have a small gold-labelled set and a *much larger* unlabelled pool of documents/queries. Train a cross-encoder on the gold, label the pool with it, distill into a bi-encoder. End result: a bi-encoder stronger than what the gold set alone could produce.

```bash
# Step A — train a cross-encoder on the gold set (out of scope here;
# see the cross-encoder README once it ships, or use HF's run_clm.py).
# Result: outputs/my-cross-encoder/final

# Step B — for the unlabelled pool, you need at least (query, candidate)
# pairs. The simplest path: use the unlabelled documents as queries and
# mine candidates from your gold positives' corpus.
python scripts/embedding_bi_encoder/mine_hard_negatives.py \
    --input-dataset data/processed/my-unlabelled-pairs \
    --output-dir    data/processed/my-silver-mined \
    --strategy ensemble \
    --retriever     BAAI/bge-large-en-v1.5 \
    --bm25

# Step C — score the mined candidates with your custom cross-encoder.
#         This produces SILVER teacher scores.
python scripts/embedding_bi_encoder/score_with_cross_encoder.py \
    --input-dataset data/processed/my-silver-mined \
    --output-dir    data/processed/my-silver-distill \
    --cross-encoder outputs/my-cross-encoder/final

# Step D — distil into the bi-encoder.
#         Set loss.name: DistillKLDivLoss in the YAML, processed_dir: …silver-distill
python scripts/embedding_bi_encoder/train_bi_encoder.py --config <your>.yaml
```

Tips for silver labels:
- The cross-encoder doesn't need to be perfect; it needs to be *better than the bi-encoder*. Even a moderate cross-encoder yields strong silver labels.
- Mix gold + silver in the same `processed_dir` (concatenate the DatasetDicts) — the bi-encoder benefits from learning to match both.

---

### Recipe 8 — Instruction prompts

**Use when**: your backbone is E5, Nomic Embed, BGE-M3, or any model that was pre-trained with explicit query/document prefixes. Fine-tuning *without* the prefixes drifts the model off its own protocol and silently degrades retrieval quality.

```yaml
bi_encoder:
  prompts:
    # Applied per column during training — maps column name → prefix string.
    columns:
      anchor:   "search_query: "
      positive: "search_document: "
      negative: "search_document: "
    # Saved on the model so the predictor can do
    #   predictor.encode(texts, prompt_name="query")
    # If omitted, auto-derived from `columns` (query=anchor, document=positive).
    inference:
      query:    "search_query: "
      document: "search_document: "
```

At serve time:

```python
emb = predictor.encode(["how do vector DBs work?"], prompt_name="query")
hits = predictor.search(
    queries=["how do vector DBs work?"],
    corpus=corpus,
    query_prompt_name="query",
    corpus_prompt_name="document",
)
```

Common prefix sets:

| Family | Query prefix | Document prefix |
|---|---|---|
| `nomic-ai/modernbert-embed-base`, Nomic Embed v1+ | `search_query: ` | `search_document: ` |
| `intfloat/e5-*` | `query: ` | `passage: ` |
| `BAAI/bge-*` (asymmetric variants) | `Represent this sentence for searching relevant passages: ` | _(empty)_ |
| `Alibaba-NLP/gte-large-en-v1.5` | _(none — symmetric)_ | _(none)_ |
| `mixedbread-ai/mxbai-embed-large-v1` | `Represent this sentence for searching relevant passages: ` | _(empty)_ |

---

## Reference

### Dataset shapes

Sentence-Transformers reads columns *positionally*; the count + the trailing column's type decide the shape ([dataset.py:`infer_shape`](../../lievito_madre_ai_lab/embedding/bi_encoder/dataset.py)):

| Shape | Columns | Used by |
|---|---|---|
| `pair` | 2 strings `(anchor, positive)` | MNRL with in-batch negatives only |
| `triplet` | 3 strings `(anchor, positive, negative)` | MNRL + 1 hard negative |
| `multi_negative` | 4+ strings `(anchor, positive, neg_1, …, neg_N)` | MNRL with mined hard negatives |
| `distill` | multi_negative + trailing `label: list[float]` | `DistillKLDivLoss` (listwise KL) |

Splits within a single `DatasetDict` must share a shape.

### Loss matrix

Configured under `bi_encoder.loss.name`. Registry in [trainer.py](../../lievito_madre_ai_lab/embedding/bi_encoder/trainer.py):

| Loss | Shape | GradCache | When to use |
|---|---|---|---|
| `MultipleNegativesRankingLoss` *(default)* | pair / triplet / multi_negative | ✓ auto-swap | Standard retrieval fine-tuning. Bigger batches = more negatives. |
| `CachedMultipleNegativesRankingLoss` | same as MNRL | (already cached) | Explicit cached MNRL. Prefer the auto-swap below — same effect, cleaner config. |
| `DistillKLDivLoss` | distill | ✗ no-op + warn | Listwise KL distillation. Recipe 3 / Recipe 7. KL is structurally incompatible with the GradCache pattern. |

Each loss can be optionally wrapped in `MatryoshkaLoss` (1D) or `Matryoshka2dLoss` (2D) via the `bi_encoder.matryoshka` block.

### GradCache (gradient caching) — the memory trick

**Always on** in every shipped config. Plain in-batch-negatives losses (MNRL and its cousins) require every (anchor, positive) embedding in the batch to live in GPU memory *simultaneously*. That fundamentally limits the batch size on a single GPU — and **batch size is the single biggest quality lever for in-batch-negatives training** (more negatives per anchor = harder learning signal).

Concrete cap, single GPU:

| Backbone | GPU | Plain-MNRL batch cap | What you actually want |
|---|---|---|---|
| MiniLM-L6 | 24 GB | ~256 | ~256 ✓ |
| ModernBERT-base / BGE-base | 24 GB | ~64 | ~512 |
| BGE-large / ModernBERT-large | 24 GB | ~16 | ~512 |
| any large model | 80 GB | ~64 | ~1024 |

GradCache fixes this by splitting the batch into `mini_batch_size` chunks, running forward per-chunk to collect embeddings + their gradients, then backpropagating chunk-by-chunk. Memory cost is bounded by `mini_batch_size`, **not** by the logical batch size — so you can train with effective batch 1024 on a GPU that would normally cap at 32.

Trade-off: each step is ~30–50% slower (the model is re-run during the second pass). For any memory-bound setup, this is a net quality win — *not* training at large batch hurts the model far more than a slower step.

Shipped defaults (see [modernbert_mnrl.yaml](../../configs/embedding/bi_encoder/modernbert_mnrl.yaml)):

```yaml
per_device_train_batch_size: 256       # logical batch the loss sees
bi_encoder:
  loss:
    name: MultipleNegativesRankingLoss # auto-swapped to CachedMNRL
  gradient_caching:
    enabled: true                      # always on
    mini_batch_size: 32                # the largest chunk that fits without OOM
```

The trainer prints `[gradcache] MultipleNegativesRankingLoss → CachedMultipleNegativesRankingLoss (mini_batch_size=32)` at startup so the swap is visible.

When `gradient_caching.enabled` is paired with a loss that has no cached variant (e.g. `DistillKLDivLoss`), the trainer prints a `[warn]` and proceeds without caching. The distillation recipe already uses much smaller batches by design — caching matters less there.

**How to tune** for your hardware:

| GPU | Backbone | Suggested batch | Suggested mini_batch_size |
|---|---|---|---|
| T4 / 16 GB | ModernBERT-base | 64-128 | 16 |
| 3090 / 4090 / A6000 (24 GB) | ModernBERT-base | 256 (default) | 32 |
| A100 (40 GB) | BGE-large | 256-512 | 32 |
| A100 (80 GB) / H100 | any | 512-1024 | 64-128 |
| CPU only | MiniLM-L6 | 32-64 | 8-16 |

Rule of thumb:
- **OOM → halve `mini_batch_size` first**, not `per_device_train_batch_size`. The logical batch is what controls quality; the mini-batch is just the memory dial.
- Larger `mini_batch_size` = faster per step (fewer chunks to re-run).
- The effective batch (`per_device_train_batch_size × gradient_accumulation_steps × world_size`) is what the loss sees — push it to 256+ for serious training.

### Temperature scaling

Three independent temperatures shape the learning signal — tuning any one of them changes how the model trains, often more than the choice of loss itself.

| Where | YAML field | Default | What it controls | Useful range |
|---|---|---|---|---|
| **MNRL / CachedMNRL** softmax sharpness | `bi_encoder.loss.kwargs.scale` (= 1 / temperature) | `20.0` (≈ temp 0.05) | How sharply the contrastive softmax peaks on the positive vs. in-batch negatives. Higher scale = harder learning signal but more gradient noise. | `10`–`50` |
| **DistillKLDivLoss** teacher softmax temperature | `bi_encoder.loss.kwargs.temperature` | `1.0` | How soft the teacher's score distribution is. Higher = student sees more of the teacher's "dark knowledge" — the relative rankings of *low-scoring* candidates, not just which is on top. Most retrieval distillation papers use `2.0`–`4.0`, not `1.0`. | `1.0`–`4.0` |
| **Matryoshka 2D** layer-alignment KL | `bi_encoder.matryoshka.kl_temperature` | `0.3` | Sharpness of the KL term that pulls earlier layers' output distributions toward the last layer's. Lower = sharper alignment, less tolerance for divergent intermediate representations. | `0.1`–`1.0` |

When to actually tune:

- **MNRL scale**: leave at `20.0` unless you see (a) training loss plateauing instantly (lower scale to `10`) or (b) loss exploding with large effective batches (raise to `30`–`50`).
- **DistillKLDivLoss temperature**: this one *matters*. Sweep `[1.0, 2.0, 4.0]` on a small slice — the differences are often 1–2 MRR points. Higher temperature is especially helpful when the teacher is much stronger than the student.
- **Matryoshka 2D kl_temperature**: rarely needs tuning. Only revisit if 2D-trained layer-truncated inference is much worse than the same model at full depth.

The three temperatures don't interact directly — they live in different parts of the loss graph. You can change one without re-tuning the others.

### Evaluator selection

Auto-picked per [evaluate.py](../../lievito_madre_ai_lab/embedding/bi_encoder/evaluate.py):

| Shape | Evaluator | Primary metric (auto-filled into `metric_for_best_model`) |
|---|---|---|
| `triplet` | `TripletEvaluator` | `eval_<split>_cosine_accuracy` |
| anything else | `InformationRetrievalEvaluator` (anchor → positive) | `eval_<split>_cosine_mrr@10` |

When `metric_for_best_model: f1` (the encoder default) appears in the YAML, the trainer prints an `[info]` line and swaps it for the matching ST key — no manual edit needed.

### Mining strategy reference

| Strategy | Best for | Pros | Cons |
|---|---|---|---|
| `--strategy dense` | Most fine-tunes | Easy, strong baseline | Single failure mode (only semantic-close negatives) |
| `--strategy bm25` | Lexical-signal domains (legal, medical, code) | No GPU needed, very fast | Misses semantic negatives |
| `--strategy ensemble` | Best quality | Diverse hardness, complementary failure modes | More config + compute |

Knob reference:

| Flag | Default | Effect |
|---|---|---|
| `--num-negatives` | 5 | Negatives per anchor in the output dataset. |
| `--relative-margin` | 0.4 | Drop negatives with cross-encoder score > `(1-margin) × positive_score`. 0.4 = "60% of positive" rule. |
| `--margin` | None | Absolute alternative to `--relative-margin`. Mutually exclusive. |
| `--max-score` / `--min-score` | None | Absolute caps on negative scores. |
| `--range-min` | 10 | Skip top-K candidates (defeat near-duplicates). 0 on clean curated data. |
| `--range-max` | 100 | Sample from top-K. Smaller = harder. |
| `--sampling-strategy` | `top` | `top` (K hardest after filter) vs `random` (uniform). |
| `--use-faiss` | off | ANN retrieval for large corpora. |
| `--corpus-file` | _(unset)_ | External pool, one doc per line. Defaults to the union of positives. |

### Sidecar (`preprocessing.json`)

| Field | Written by | Read by |
|---|---|---|
| `source` | prep script | informational |
| `shape` / `columns` / `loss` | train | informational |
| `tokenizer` | train (= `cfg.model_name`) | tokenizer-mismatch guard |
| `max_seq_length` | prep / train | predictor default |
| `matryoshka_dims` | train (when enabled) | informational |
| `matryoshka_mode` | train (when enabled) | informational |
| `default_truncate_dim` | train (when 1D / 2D enabled) | predictor default |
| `matryoshka_n_layers_per_step` | train (when 2D enabled) | informational |
| `inference_prompts` | train | informational (actual prompts persist on the model) |
| `mining.*` | mining CLI | provenance |
| `distillation.*` | scoring CLI | provenance |

Drift guards:

- **Tokenizer mismatch (hard error)**: prep wrote `tokenizer: X`, YAML says `model_name: Y` → train bails before any GPU work.
- **Loss-vs-shape (hard error)**: `DistillKLDivLoss` on a non-distill dataset crashes immediately. Non-distill loss on a distill dataset prints a warning and ignores the `label` column.

### Operations

```bash
# Resume from the latest checkpoint
python scripts/embedding_bi_encoder/train_bi_encoder.py --config <…>.yaml --resume

# Resume from a specific checkpoint
python scripts/embedding_bi_encoder/train_bi_encoder.py --config <…>.yaml \
    --resume outputs/<run>/<exp>/checkpoint-1000

# Smoke-test on a slice
python scripts/embedding_bi_encoder/train_bi_encoder.py --config <…>.yaml \
    --max-train-samples 1000 --max-eval-samples 500
```

### Inference

```python
from lievito_madre_ai_lab.embedding.bi_encoder.serve import BiEncoderPredictor

predictor = BiEncoderPredictor("outputs/<run>/final")

# Plain encode (L2-normalised by default)
emb = predictor.encode(["semantic search"])

# Matryoshka — speed/storage trade
emb_64 = predictor.encode(["semantic search"], truncate_dim=64)

# Matryoshka 2D — shallow + small
shallow = BiEncoderPredictor("outputs/<run>/final", truncate_layers=6, truncate_dim=128)

# Instruction prompts (saved on the model at training time)
emb = predictor.encode(["how do vector DBs work?"], prompt_name="query")

# Asymmetric retrieval
hits = predictor.search(
    queries=["how do vector DBs work?"],
    corpus=corpus,
    query_prompt_name="query",
    corpus_prompt_name="document",
    top_k=5,
)
```

CLI:

```bash
python -m lievito_madre_ai_lab.embedding.bi_encoder.serve outputs/<run>/final \
    "semantic search" --prompt-name query --truncate-dim 64

python -m lievito_madre_ai_lab.embedding.bi_encoder.serve outputs/<run>/final \
    "edge text" --truncate-layers 6 --truncate-dim 128

python -m lievito_madre_ai_lab.embedding.bi_encoder.serve outputs/<run>/final --benchmark
```

Hardware-aware automation:

| Hardware | Optimisations |
|---|---|
| CUDA (Ampere+) | FP16 weights · Flash Attention 2 · `torch.compile` · BF16 autocast · L2-normalised outputs |
| CUDA (Turing/T4) | FP16 weights · SDPA · `torch.compile` · FP16 autocast |
| CPU | FP32 weights · dynamic INT8 quantisation of Linear layers |

### Bringing your own data

```python
from datasets import Dataset, DatasetDict
from lievito_madre_ai_lab.embedding.bi_encoder.dataset import (
    save_preprocessing_meta,
    validate_dataset,
)

raw = DatasetDict({
    "train":      Dataset.from_dict({"anchor": [...], "positive": [...]}),
    "validation": Dataset.from_dict({"anchor": [...], "positive": [...]}),
})
shape = validate_dataset(raw)             # "pair"
raw.save_to_disk("data/processed/my-bi-encoder")
save_preprocessing_meta(
    "data/processed/my-bi-encoder",
    source="my-corpus",
    max_seq_length=512,
)
```

Then plug into Recipes 2 → 6 to add mining and (optionally) distillation.

### Recommended public starter datasets

| Dataset | Native shape | Use case |
|---|---|---|
| [`sentence-transformers/all-nli`](https://huggingface.co/datasets/sentence-transformers/all-nli) `triplet` | triplet | Generic similarity — the canonical MNRL starter |
| [`sentence-transformers/quora-duplicates`](https://huggingface.co/datasets/sentence-transformers/quora-duplicates) | triplet | Paraphrase / duplicate detection |
| [`sentence-transformers/msmarco-bm25`](https://huggingface.co/datasets/sentence-transformers/msmarco-bm25) | triplet | Web-passage retrieval with BM25-mined hard negatives |
| [`sentence-transformers/gooaq`](https://huggingface.co/datasets/sentence-transformers/gooaq) | pair | Question → answer retrieval — feed straight into mining |

### Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Eval metric plateaus instantly | Mining produces near-duplicates of the positive | Raise `--range-min` (default 10 → try 20-50) |
| Many rows dropped during mining | Filter too aggressive | Loosen `--relative-margin` (0.4 → 0.3) or raise `--range-max` |
| Training loss flat from step 1 | Tokenizer mismatch (silent before this lab) | Re-check `model_name` vs `preprocessing.json` — the guard now hard-errors |
| Quality plateaus far below published numbers | Effective batch too small (in-batch negatives need ≥256) | Bump `per_device_train_batch_size`. GradCache is already on by default, so memory stays bounded by `mini_batch_size`. |
| OOM | `mini_batch_size` too large for the GPU | Halve `mini_batch_size` first — logical batch can stay where it is. Only shrink `per_device_train_batch_size` once `mini_batch_size` is tiny. |
| Inference quality much worse than eval | Forgot prompts on an instruction-tuned model | Add `bi_encoder.prompts.columns` and re-train |
| `truncate_layers` produces garbage embeddings | Model trained without `mode: "2d"` | Layer truncation requires Matryoshka2d / AdaptiveLayer training |
| `DistillKLDivLoss` raises shape error | Dataset doesn't have a teacher-score `label` column | Run `score_with_cross_encoder.py` first |
| Ensemble mining drops all rows | All sources mined the same candidates → dedup leaves <num_negatives | Add more retriever diversity, raise `--num-negatives` to over-mine more |
