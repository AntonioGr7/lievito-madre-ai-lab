# Token Classification

Fine-tuning encoder models for token-level tasks like NER and PII detection. Labels follow the BIO scheme (`O`, `B-<entity>`, `I-<entity>`). By default every subword of an entity carries the BIO tag (first subword → `B-`, continuations → `I-`); set `--no-label-all-tokens` on the prepare script to fall back to the legacy "label only the first subword" recipe.

## Quickstart

```bash
# 1. Prepare the dataset (downloads, tokenizes, aligns BIO labels to subwords).
#    Two prep scripts ship with this repo:
#      - prepare_openpii.py    (ai4privacy/open-pii-masking-500k-ai4privacy)
#      - prepare_nemotron_pii.py (nvidia/Nemotron-PII)
python examples/token_classification/prepare_openpii.py

# 2. Train
python scripts/token_classification/train_token_classification.py \
    --config configs/encoder/token_classification/pii_mbert.yaml
```

The default dataset is `ai4privacy/open-pii-masking-500k-ai4privacy` (8 languages, 19 PII entity types). The default model is `microsoft/mdeberta-v3-base` (despite the `pii_mbert.yaml` filename, which is historical).
Metrics tracked at each epoch via `seqeval` (entity-level, not token-level): `precision`, `recall`, `f1`, `accuracy`.
Final test metrics are saved to `outputs/<run>/final/test_metrics.json`.

## Long documents (sliding-window chunking)

A 2048-token document tokenized with `max_length=512` would lose three-quarters of its supervision to plain truncation. The prepare scripts use **HuggingFace's overflow tokenization with stride** instead: long inputs are split into overlapping `max_length`-token chunks, each becoming its own training row.

```bash
python examples/token_classification/prepare_nemotron_pii.py \
    --max-length 512 \
    --stride 128         # 128 = default; -1 disables (legacy truncate-only behaviour)
```

The prep script prints the expansion when it happens:

```
[train] 50000 docs → 73412 chunks (+23412 from long documents)
```

What goes where:

| Stage | What chunking does |
|---|---|
| Prep | `tokenize_for_trainer(..., stride=128)` chunks rows. Each chunk gets aligned BIO labels for *its* tokens; entities in chunk-overlap regions appear in both chunks (B- inside each — independent training signal). |
| Train | Sees each chunk as a separate row. No special handling needed. |
| Serve | `TokenClassificationPredictor` runs the same overflow-tokenization at inference time and **merges** same-label adjacent spans back to original-text coordinates, so the caller still gets one span list per input text. |

**Cost / caveat**: chunks multiply the effective batch when many texts are long. If you OOM at training or inference, lower `per_device_train_batch_size` / `batch_size`. Stride 128 with `max_length=512` triples-to-quadruples the row count for 2k-token docs.

## Preprocessing metadata (`preprocessing.json`)

Every prepare script writes a sidecar `preprocessing.json` next to the processed dataset, recording the exact tokenizer, `max_length`, `stride`, label set, and label-alignment policy used. The train script copies it into `outputs/<run>/final/preprocessing.json`. The predictor reads it on load.

```json
{
  "source": "nvidia/Nemotron-PII",
  "tokenizer": "microsoft/mdeberta-v3-base",
  "max_length": 512,
  "stride": 128,
  "label_all_tokens": true,
  "entity_types": ["first_name", "last_name", ...]
}
```

Two guards protect against config drift:

- **Tokenizer mismatch (hard error)**: if `cfg.model_name` in the YAML doesn't match `preprocessing.json["tokenizer"]`, the train script bails before any GPU work. Saved `input_ids` reference one vocabulary; loading a different model would train on nonsense embeddings.
- **`max_length` vs model capacity (soft warning)**: if the dataset was prepared with `max_length=2048` but the chosen model only supports `max_position_embeddings=512`, a warning fires after the model loads. Architectures with relative or rotary position embeddings (T5, ModernBERT with RoPE extrapolation) are exempt.

At serve time you get the right `max_length` and `stride` automatically — no need to remember what prep was run with. Explicit constructor args still win.

## Switching to a longer-context model (e.g. ModernBERT)

The metadata system makes this safe, but you still need to keep three things in sync:

1. **Prep**: re-run the prepare script with `--model answerdotai/ModernBERT-base --max-length 2048`. The sidecar records this.
2. **YAML config**: `model_name` must match exactly. Also flip `attn_implementation: sdpa` (ModernBERT supports SDPA / FA2; mDeBERTa-V2 required `eager`).
3. **Batch size**: 2048-token sequences use ~16× the activations of 512-token. Drop `per_device_train_batch_size` and bump `gradient_accumulation_steps` to keep the effective batch constant.

The tokenizer-mismatch guard catches step 2 if you forget. The capacity warning catches step 1 if you over-bump `max_length`.

## Resuming after a failure

Pass `--resume` to continue from the latest checkpoint automatically:

```bash
python scripts/token_classification/train_token_classification.py \
    --config configs/encoder/token_classification/pii_mbert.yaml \
    --resume
```

Or point at a specific checkpoint:

```bash
python scripts/token_classification/train_token_classification.py \
    --config configs/encoder/token_classification/pii_mbert.yaml \
    --resume outputs/pii_mbert/checkpoint-5000
```

## Inference

Load the trained model for high-performance NER inference. The predictor decodes BIO predictions into entity spans with character offsets back into the original text:

```python
from lievito_madre_ai_lab.encoder.token_classification.serve import TokenClassificationPredictor

predictor = TokenClassificationPredictor("outputs/pii_mbert/final")

# batch
results = predictor.predict([
    "Send the report to Maria Rossi at maria.rossi@example.com.",
    "John called from +1-800-555-0199.",
])
# [
#   [{"text": "Maria Rossi", "label": "GIVENNAME", "start": 19, "end": 30, "score": 0.98}, ...],
#   [{"text": "+1-800-555-0199", "label": "TELEPHONENUM", "start": 17, "end": 32, "score": 0.99}],
# ]

# single
spans = predictor.predict_one("Email me at jane@example.com.")
```

Run from the CLI:

```bash
python -m lievito_madre_ai_lab.encoder.token_classification.serve \
    outputs/pii_mbert/final "Send it to John Doe at john@example.com."
```

Benchmark throughput:

```bash
python -m lievito_madre_ai_lab.encoder.token_classification.serve \
    outputs/pii_mbert/final --benchmark
```

The same hardware-aware optimisation stack as text classification applies (FP16/BF16 weights, SDPA / Flash Attention 2, `torch.compile`, CPU INT8 quantisation).

## Switching model or dataset

Duplicate the YAML config and edit `model_name`, `processed_dir`, and `output_dir`:

```bash
cp configs/encoder/token_classification/pii_mbert.yaml \
   configs/encoder/token_classification/ner_xlmr.yaml
```

For multilingual NER, `microsoft/mdeberta-v3-base` (the current default) and `xlm-roberta-base` are stronger backbones than mBERT. For long documents, `answerdotai/ModernBERT-base` supports up to 8192 tokens — see [Switching to a longer-context model](#switching-to-a-longer-context-model-eg-modernbert) above.

Then run with `--config configs/encoder/token_classification/ner_xlmr.yaml`.

## Bringing your own data

The training script only requires a tokenized Arrow dataset with three columns: `input_ids`, `attention_mask`, and `labels` (a per-token sequence of BIO label ids, with `-100` for special and non-first-subword tokens). There are two paths to get there.

**Path 1 — your data has text + character-span annotations** (the default ai4privacy format):

Each row needs a text column and a list of entity spans:

```python
{
    "source_text": "Send it to John Doe at john@example.com.",
    "privacy_mask": [
        {"label": "GIVENNAME", "value": "John Doe", "start": 11, "end": 19},
        {"label": "EMAIL",     "value": "john@example.com", "start": 23, "end": 39},
    ],
}
```

Use `tokenize_for_trainer` to tokenize and align labels to subwords:

```python
from datasets import DatasetDict, Dataset
from lievito_madre_ai_lab.encoder.token_classification.dataset import tokenize_for_trainer

raw = DatasetDict({
    "train": Dataset.from_dict({"source_text": [...], "privacy_mask": [...]}),
    "test":  Dataset.from_dict({"source_text": [...], "privacy_mask": [...]}),
})

processed = tokenize_for_trainer(raw, model_name="microsoft/mdeberta-v3-base")
processed.save_to_disk("data/processed/my_ner_dataset")
# Don't forget to write the sidecar so train/serve can rediscover the settings:
from lievito_madre_ai_lab.encoder.token_classification.dataset import save_preprocessing_meta
save_preprocessing_meta(
    "data/processed/my_ner_dataset",
    tokenizer="microsoft/mdeberta-v3-base",
    max_length=512, stride=128,
    text_col="source_text", mask_col="privacy_mask",
    entity_types=[...],  # the labels you used
)
```

If your label set differs from the default PII entities, edit `ENTITY_TYPES` in [lievito_madre_ai_lab/encoder/token_classification/dataset.py](../../lievito_madre_ai_lab/encoder/token_classification/dataset.py) — `LABEL_NAMES` is derived from it as `["O"] + B-/I- pairs`.

**Path 2 — your data is already token-and-tag aligned** (e.g. CoNLL format):

Build the `labels` column directly using `LABEL2ID` and save the Arrow dataset — see `tokenize_for_trainer` in the same module for the expected schema (`input_ids`, `attention_mask`, `labels` with `labels` cast to `Sequence(ClassLabel(names=LABEL_NAMES))` so label names survive `save_to_disk`).

Point `processed_dir` in your YAML config at `data/processed/my_ner_dataset` and run the training script as normal.
