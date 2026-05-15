# GLiNER Entity Extraction

Fine-tuning [GLiNER](https://github.com/urchade/GLiNER) for open-vocabulary entity extraction. Entity types are passed as prompts at inference time — the same model can extract any label you ask for, including ones it never saw at training.

**Different transformers version.** GLiNER 0.2.x requires `transformers<5.2.0`, which conflicts with the encoder/decoder/vision groups. Install in its own virtualenv:

```bash
pip install -e ".[gliner]"
```

The default backbone is [`knowledgator/gliner-multitask-large-v0.5`](https://huggingface.co/knowledgator/gliner-multitask-large-v0.5) — the strongest public GLiNER for zero-shot/few-shot at the time of writing. The trainer auto-detects precision (bf16 on Ampere+, fp16 on Turing/T4, fp32 on CPU), so the same YAML runs on a T4 and on a B100 without edits.

## Dataset contract

Every prepare script you write must produce the following on-disk layout:

```
data/processed/<task>/
├── train/             # HF DatasetDict shards (load_from_disk-compatible)
├── validation/
├── test/              # single split; closed-set + zero-shot are runtime views
├── train_types.json   # ["PERSON", "ORG", ...]
└── holdout_types.json # [] if no zero-shot probe desired
```

Every row in every split has the same shape — char-offset spans, half-open `[start, end)`:

```python
{
    "text": "Maria Rossi lives in Rome.",
    "spans": [
        {"start": 0,  "end": 11, "label": "PERSON"},
        {"start": 21, "end": 25, "label": "CITY"},
    ],
    # any extra columns are silently kept (ignored by trainer)
}
```

**Filtering rules** the prep script enforces:
- `train` + `validation`: keep only spans whose label is in `train_types`. Held-out spans are **dropped**, not relabeled — GLiNER must see no signal on held-out types during training.
- `test`: keep all spans. Eval-time label-set filtering produces the closed-set and zero-shot views.

`examples/gliner_entity_extraction/prepare_openpii.py` is the worked example targeting OpenPII. Use it as a template.

## Quickstart (OpenPII reference)

```bash
# 1. Build the char-offset dataset from the ai4privacy OpenPII corpus
python examples/gliner_entity_extraction/prepare_openpii.py \
    --out-dir data/processed/pii-gliner

# 2. Train
python scripts/gliner_entity_extraction/train_gliner.py \
    --config configs/encoder/gliner_entity_extraction/pii_gliner.yaml
```

Holdout types default to `PASSPORTNUM DRIVERLICENSENUM AGE` (overridable via `--holdout-types`). Final test metrics land in `outputs/<run>/final/test_metrics.json` with two prefixed sections:
- `test_closed_*` — F1 on `train_types`, scored against the test split.
- `test_zeroshot_*` — F1 on `holdout_types` (open-vocabulary probe), scored against the same test split with the holdout labels as the prompt set.

Per-label F1 is reported under `f1_<LABEL>`, `precision_<LABEL>`, `recall_<LABEL>` for both views. The output span schema (`label`, `text`, `start`, `end`, `score`) matches `TokenClassificationPredictor` so the same downstream code consumes both.

## Long documents (sliding-window chunking)

GLiNER differs architecturally from text/token classification in two ways that change *where* chunking belongs:

1. **The dataset is tokenizer-agnostic.** The prepare script emits char-offset rows (`{"text": str, "spans": [...]}`); the same processed dataset works with any GLiNER backbone. Adding tokenization at prep would lock the dataset to one model.
2. **GLiNER operates on word-tokens, not subwords.** Its splitter lives on the model (`model.data_processor.words_splitter`) and the truncation cap is `model.config.max_len`, measured in word-tokens.

So sliding-window chunking happens **at trainer-build time**, not at prep. The trainer (and the predictor) split long word-sequences into overlapping windows of `max_words` with `stride` overlap. Each chunk becomes a self-contained char-offset document with `text` sliced to the chunk's char range and `spans` re-based to chunk-local offsets — so the same row feeds GLiNER's loss (via `tokenized_text`/`ner`) and the eval callback (via `text`/`spans`) without re-truncating.

Configure it in the YAML's `gliner:` block:

```yaml
gliner:
  chunking:
    stride: 64                # default. Set to -1 to disable chunking entirely.
    max_words: 384            # optional. Omit to use model.config.max_len.
```

What gets chunked:

| Split | Chunked? | Effect |
|---|---|---|
| `train` | Yes | All words contribute to loss; entities past `max_len` are no longer silently dropped. |
| `validation` (during training) | Yes | `eval_f1` reflects performance on what the model actually sees — chunk-level. |
| `test` (final metrics) | Yes | `train_gliner.py` re-applies the same chunking before calling `evaluate_split` so `test_closed_*` and `test_zeroshot_*` use the same units as `eval_f1`. |

The prep script and trainer print expansion counts when chunking fires:

```
[train] 50000 docs → 73412 chunks (+23412 from long documents)
```

### Chunk-level vs document-level f1

We compute f1 at the **chunk level**, not the document level. An entity sitting in the overlap region appears in two chunks; the model has to predict it correctly in *both* to score full marks. That's slightly stricter than doc-level f1, but:

- It matches what the model sees at inference (one chunk at a time).
- It's cheap to implement — no aggregation pass.
- At corpus scale, the aggregate tracks doc-level performance closely.

If you need true document-level f1, post-process the per-chunk predictions yourself: group by source document, merge overlapping same-label spans, score against the original gold set.

### What happens at inference time

`GLiNERPredictor` does its own chunking transparently. Long inputs get split into word-windows, each chunk goes through `batch_predict_entities`, and the returned spans have their char offsets shifted back to the original text. Same-label adjacent/overlapping spans (the boundary-region duplicates) get merged into a single span with the higher score. The caller sees one clean span list per input, in original-text coordinates — exactly what they'd expect from `predict_one(long_text)`.

The predictor reads `max_words` and `stride` from `preprocessing.json` (see below) on load. Pass `max_words=...` / `stride=...` (or `--max-words` / `--stride` on the CLI) to override; `stride=-1` disables chunking.

## Preprocessing metadata (`preprocessing.json`)

Unlike the other two pipelines, GLiNER's prep doesn't pick a tokenizer — so the metadata flow has a slight twist:

- **Prep** writes a tokenizer-agnostic `preprocessing.json` (`source`, `languages`, `train_types`, `holdout_types`).
- **Train** augments it with the train-time choices (`tokenizer` = `cfg.model_name`, effective `max_words`, `stride`, train/holdout types) and writes the combined file into `outputs/<run>/final/preprocessing.json`.
- **Serve** reads it on load and uses the recorded `max_words`/`stride` as defaults.

```json
{
  "source": "ai4privacy/open-pii-masking-500k-ai4privacy",
  "languages": ["en"],
  "tokenizer": "knowledgator/gliner-multitask-large-v0.5",
  "max_words": 384,
  "stride": 64,
  "train_types": ["GIVENNAME", "SURNAME", "EMAIL", ...],
  "holdout_types": ["PASSPORTNUM", "DRIVERLICENSENUM", "AGE"]
}
```

Two guards run before any GPU work (same as the other pipelines):

- **Tokenizer mismatch (hard error)**: if `cfg.model_name` differs from the recorded tokenizer the train script bails.
- **`max_length`/`max_words` vs model capacity (soft warning)**: fires when the prep-recorded cap exceeds the model's `max_position_embeddings`.

## Bringing your own data

Copy `examples/gliner_entity_extraction/prepare_openpii.py` as a starting point and adapt it to your corpus. The simplest pattern:

```python
import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from lievito_madre_ai_lab.encoder.gliner_entity_extraction.dataset import (
    collect_entity_types, partition_entity_types, validate_row,
)

# 1. Load whatever raw corpus you have, however you have it. Just produce a
#    DatasetDict with train / validation / test splits whose rows already
#    follow the char-offset schema (or convert them to it here).
raw = DatasetDict({
    "train":      Dataset.from_list(my_train_rows),       # [{"text", "spans"}, ...]
    "validation": Dataset.from_list(my_val_rows),
    "test":       Dataset.from_list(my_test_rows),
})

# 2. Collect the full label vocabulary and partition into train + holdout.
all_types = collect_entity_types(raw, span_col="spans")
train_types, holdout_types = partition_entity_types(all_types, ["AGE", "PASSPORTNUM"])

# 3. Filter spans so train/val only carry train_types; test keeps everything.
train_set = frozenset(train_types)
full_set  = frozenset(all_types)

def _filter(row, allowed):
    return {"text": row["text"],
            "spans": [s for s in row["spans"] if s["label"] in allowed]}

processed = DatasetDict({
    "train":      raw["train"]     .map(lambda r: _filter(r, train_set)),
    "validation": raw["validation"].map(lambda r: _filter(r, train_set)),
    "test":       raw["test"]      .map(lambda r: _filter(r, full_set)),
})

# 4. Fail loudly at prep time, not training step 1.
for name, split in processed.items():
    if len(split) and (errs := validate_row(split[0])):
        raise ValueError(f"{name} row 0: {errs}")

# 5. Save.
out = Path("data/processed/my-gliner")
out.mkdir(parents=True, exist_ok=True)
processed.save_to_disk(str(out))
(out / "train_types.json").write_text(json.dumps(train_types))
(out / "holdout_types.json").write_text(json.dumps(holdout_types))

# 6. Sidecar — tokenizer-agnostic at prep time (the train script adds the
#    model_name / chunking fields when you actually train).
from lievito_madre_ai_lab.shared.preprocessing import save_preprocessing_meta
save_preprocessing_meta(
    out,
    source="my-corpus",
    train_types=train_types,
    holdout_types=holdout_types,
)
```

Then point a YAML config at `processed_dir: data/processed/my-gliner` and train.

`validate_row` checks the schema invariants (`text` is a string, every span has `end > start`, `start >= 0`, `end <= len(text)`, `label` is a non-empty string). Call it on every row during prep — schema bugs surface at prep time, not at step 1 of training.

## Fine-tuning paths — LoRA vs full FT

The YAML's `gliner.peft.enabled` switch picks between the two. Choose based on memory, dataset size, and how far the backbone needs to move.

### LoRA (recommended default — fits T4)

LoRA wraps the encoder with low-rank adapters and freezes the original encoder weights; the span scoring head stays fully trainable. On `knowledgator/gliner-multitask-large-v0.5` (~660M params), LoRA cuts the trainable parameter count by ~99% and fits comfortably on a 16GB T4 with batch 8 + grad-checkpointing.

```yaml
# configs/encoder/gliner_entity_extraction/<your>.yaml
model_name: knowledgator/gliner-multitask-large-v0.5
per_device_train_batch_size: 8
gradient_accumulation_steps: 2          # effective batch = 16
gradient_checkpointing: true            # ~40% memory cut
precision: auto                         # bf16 on Ampere+, fp16 on T4

gliner:
  peft:
    enabled: true
    r: 16                               # rank — 16 is the "first try" default
    alpha: 32                           # scaling = alpha / r
    dropout: 0.1
    target_modules: auto                # auto-detect; falls back to DeBERTa-v3 list
```

**When to use LoRA**: most fine-tuning runs (custom domain, smaller training set, T4-class hardware), or any time you'd want to ship multiple per-customer adapters off one base model.

**Tuning notes**:
- Bump `r` to 32 or 64 if LoRA underperforms full-FT by more than 1-2 F1 points after enough epochs.
- `target_modules: auto` resolves via `peft`'s mapping with a hardcoded DeBERTa-v3 fallback (`query_proj, key_proj, value_proj, dense`). Override with an explicit list if you're on an unusual backbone.

### Full fine-tuning

Full FT updates every encoder parameter alongside the span head. Strongest ceiling, biggest memory footprint, slowest training.

```yaml
gliner:
  peft:
    enabled: false                      # the only change vs LoRA
```

Plus on T4-class hardware, lean on memory tricks:

```yaml
per_device_train_batch_size: 2          # batch must shrink without LoRA
gradient_accumulation_steps: 8          # effective batch = 16 still
gradient_checkpointing: true
torch_compile: true                     # ~25% speedup once warm
```

**When to use full FT**: large training set (>100k rows), big domain shift from the base GLiNER's pre-training distribution, or you have an A100/H100/B100 and want the strongest possible ceiling.

## GLiNER-specific knobs

All under the `gliner:` block in the YAML. The defaults in `pii_gliner.yaml` are good starting points; tune from there.

```yaml
gliner:
  max_span_width: 16              # longest span (tokens) the scorer considers.
                                  # Set above the longest expected entity in
                                  # your data — addresses, multi-word names.

  loss:
    funcs: ["focal"]              # ["bce"] | ["focal"]  — focal is recommended
    focal_alpha: 0.75             # only used when funcs contains "focal"
    focal_gamma: 2.0

  sampling:
    max_types: 30                 # entity-prompt cap per batch
    max_neg_type_ratio: 1.0       # negative distractor labels per positive
    random_drop: 0.1              # chance to drop a gold label from the prompt

  head_lr_multiplier: 5.0         # head LR = learning_rate × this. Drops
                                  # straight into gliner's TrainingArguments
                                  # as `others_lr`. 3-10x is typical.

  label_aliases:                  # canonical label -> natural-language prompt
    GIVENNAME: "given name"       # GLiNER reads label names as prompts, so
    DRIVERLICENSENUM: "driver license number"   # natural-language helps zero-shot.
    EMAIL: "email address"        # Predictions are un-aliased back to canonical.

  chunking:                       # Sliding-window chunking for long docs.
    stride: 64                    # word-token overlap; -1 disables.
    max_words: 384                # optional; defaults to model.config.max_len.
                                  # See "Long documents" above for the full story.
```

Loss + negative sampling fields go onto `model.config` via `load_gliner` (consumed by GLiNER's collator at batch time). Focal alpha/gamma also propagate to `gliner.training.TrainingArguments.focal_loss_alpha` / `focal_loss_gamma` (consumed by `compute_loss`). Discriminative LR is handled natively by gliner's Trainer via the `others_lr` field, derived from `head_lr_multiplier`.

## Resuming after a failure

```bash
# Auto-detect the latest checkpoint
python scripts/gliner_entity_extraction/train_gliner.py \
    --config configs/encoder/gliner_entity_extraction/pii_gliner.yaml --resume

# Or point at a specific checkpoint
python scripts/gliner_entity_extraction/train_gliner.py \
    --config configs/encoder/gliner_entity_extraction/pii_gliner.yaml \
    --resume outputs/pii_gliner/gliner_exp_01/checkpoint-500
```

Smoke-test on a small slice before committing to a full run:

```bash
python scripts/gliner_entity_extraction/train_gliner.py \
    --config configs/encoder/gliner_entity_extraction/pii_gliner.yaml \
    --max-train-samples 200 --max-eval-samples 100 --max-test-samples 100
```

## Efficient serving

[`GLiNERPredictor`](../../lievito_madre_ai_lab/encoder/gliner_entity_extraction/serve.py) is the single inference entrypoint. It loads a full-FT save and a LoRA save through the same constructor, and applies the right optimisation stack for the detected hardware.

```python
from lievito_madre_ai_lab.encoder.gliner_entity_extraction.serve import GLiNERPredictor

predictor = GLiNERPredictor("outputs/pii_gliner/gliner_exp_01/final")

# Closed-set: uses train_types stamped on the model at training time
spans = predictor.predict_one("Email me at jane@example.com.")
# [{"label": "EMAIL", "text": "jane@example.com", "start": 12, "end": 28, "score": 0.97}]

# Open-vocabulary: pass any labels — even ones the model never trained on
spans = predictor.predict_one(text, labels=["passport number", "driver license"])

# Batched (sorted-by-length internally so longer texts cluster together)
results = predictor.predict([
    "Send the report to Maria Rossi at maria.rossi@example.com.",
    "John called from +1-800-555-0199.",
])
```

### What the predictor does automatically

| Concern | What happens |
|---|---|
| **Device** | CUDA if available, else MPS, else CPU. Override with `device="cuda:1"` / `"cpu"`. |
| **Precision (CUDA)** | bf16 autocast on Ampere+ (capability ≥ 8.0), fp16 autocast otherwise. |
| **Batching** | `batch_predict_entities` under the hood; sorted-by-length so similar-length texts batch together. |
| **`torch.compile`** | Applied to `model.model` (the encoder) on CUDA / MPS by default. Disable with `use_compile=False`. |
| **LoRA detection** | If `adapter_config.json` exists in the model dir, the base model is loaded and the adapter is wrapped with `PeftModel.from_pretrained`, then `merge_and_unload()` folds the LoRA weights into the base — zero inference overhead vs full FT. Disable the merge with `merge_lora_on_load=False` if you need to swap adapters at runtime. |
| **CPU quantisation** | Opt-in via `quantize_cpu=True` — applies dynamic INT8 to FFN Linears only (the span scorer's small margins do not survive INT8 on the attention path). |
| **Label aliases** | If `label_aliases` was stamped at training time, prompts go out aliased and predictions come back un-aliased — caller sees only canonical labels. |

### Constructor knobs worth knowing

```python
predictor = GLiNERPredictor(
    "outputs/pii_gliner/gliner_exp_01/final",
    device=None,                  # auto-detect
    batch_size=16,                # tune to your GPU; lower for long texts
    use_compile=True,             # torch.compile the encoder
    compile_mode="default",       # "default" | "reduce-overhead" | "max-autotune"
    amp_dtype=None,               # auto: bf16 on Ampere+, fp16 elsewhere
    quantize_cpu=False,           # True for CPU-only servers
    warmup_steps=3,               # compile warmup passes
    default_threshold=0.5,        # default span-score cutoff
    merge_lora_on_load=True,      # fold LoRA into base for fastest inference
)
```

### CLI

```bash
python -m lievito_madre_ai_lab.encoder.gliner_entity_extraction.serve \
    outputs/pii_gliner/gliner_exp_01/final \
    "Send it to Maria Rossi at maria.rossi@example.com."

# Open-vocabulary
python -m lievito_madre_ai_lab.encoder.gliner_entity_extraction.serve \
    outputs/pii_gliner/gliner_exp_01/final \
    "Driver license #ABC123 expired" \
    --labels "driver license number" "expiry date"
```

### Serving LoRA without merging

If you need to swap adapters at runtime (one base model, many per-tenant adapters):

```python
predictor = GLiNERPredictor("outputs/tenant_a/final", merge_lora_on_load=False)
# predictor._model.model is now a PeftModel; you can call .set_adapter() etc.
```

The trade-off: keeping the adapter unmerged costs ~5-10% inference latency vs the merged path.
