# Token Classification

Fine-tuning encoder models for token-level tasks like NER and PII detection. Labels follow the BIO scheme (`O`, `B-<entity>`, `I-<entity>`); only the first subword of each entity carries the BIO tag, continuation subwords are ignored in the loss by default.

## Quickstart

```bash
# 1. Prepare the dataset (downloads, tokenizes, aligns BIO labels to subwords)
python examples/token_classification/prepare_openpii.py

# 2. Train
python scripts/token_classification/train_token_classification.py \
    --config configs/encoder/token_classification/pii_mbert.yaml
```

The default dataset is `ai4privacy/open-pii-masking-500k-ai4privacy` (8 languages, 19 PII entity types). The default model is `bert-base-multilingual-cased`.
Metrics tracked at each epoch via `seqeval` (entity-level, not token-level): `precision`, `recall`, `f1`, `accuracy`.
Final test metrics are saved to `outputs/<run>/final/test_metrics.json`.

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

For multilingual NER, `microsoft/mdeberta-v3-base` and `xlm-roberta-base` are stronger backbones than mBERT.

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

processed = tokenize_for_trainer(raw, model_name="bert-base-multilingual-cased")
processed.save_to_disk("data/processed/my_ner_dataset")
```

If your label set differs from the default PII entities, edit `ENTITY_TYPES` in [lievito_madre_ai_lab/encoder/token_classification/dataset.py](../../lievito_madre_ai_lab/encoder/token_classification/dataset.py) — `LABEL_NAMES` is derived from it as `["O"] + B-/I- pairs`.

**Path 2 — your data is already token-and-tag aligned** (e.g. CoNLL format):

Build the `labels` column directly using `LABEL2ID` and save the Arrow dataset — see `tokenize_for_trainer` in the same module for the expected schema (`input_ids`, `attention_mask`, `labels` with `labels` cast to `Sequence(ClassLabel(names=LABEL_NAMES))` so label names survive `save_to_disk`).

Point `processed_dir` in your YAML config at `data/processed/my_ner_dataset` and run the training script as normal.
