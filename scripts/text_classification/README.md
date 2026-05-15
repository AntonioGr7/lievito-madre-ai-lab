# Text Classification

Fine-tuning encoder models for sequence classification.

## Quickstart

```bash
# 1. Prepare the dataset (downloads, tokenizes, saves to Arrow)
python examples/text_classification/prepare_emotion.py

# 2. Train
python scripts/text_classification/train_text_classification.py \
    --config configs/encoder/text_classification/emotion_bert.yaml
```

The default dataset is `dair-ai/emotion` (6 emotion classes). The default model is `answerdotai/ModernBERT-base`.
Metrics tracked at each epoch: `accuracy`, `f1` (weighted), and `f1_<class>` per label.
Final test metrics are saved to `outputs/<run>/final/test_metrics.json`.

## Long documents — what we do (and don't do)

`tokenize_for_trainer` truncates at `max_length`. We **don't** apply sliding-window chunking the way the token-classification and GLiNER pipelines do — and that's deliberate, not an oversight.

Text classification produces one label *per document*. Splitting a long input into chunks raises a design decision that depends on the task: is the document's label detectable from any chunk (true for most sentiment / topic), or does it need full context (legal-document categorisation)? Mean-pool / max-pool / vote across chunks all make implicit assumptions, and the right one isn't universal. So we leave it as plain truncation and document the trade-off, rather than baking in an assumption.

If your inputs reliably exceed `max_length`, your options are:

1. **Pick a long-context backbone** — `answerdotai/ModernBERT-base` handles up to 8192 tokens out of the box. Re-prep with `--model answerdotai/ModernBERT-base --max-length 2048` (or larger). The `preprocessing.json` flow below makes this safe.
2. **Pre-split documents yourself** before prep, with a custom aggregation policy that fits your task.

## Preprocessing metadata (`preprocessing.json`)

The prepare script writes a sidecar `preprocessing.json` next to the processed dataset recording the tokenizer, `max_length`, text/label columns, and source. The train script copies it into `outputs/<run>/final/preprocessing.json`. The predictor reads it on load, so you don't have to remember which `max_length` you trained with.

```json
{
  "source": "dair-ai/emotion",
  "tokenizer": "answerdotai/ModernBERT-base",
  "max_length": 128,
  "text_col": "text",
  "label_col": "label"
}
```

Two guards protect against config drift:

- **Tokenizer mismatch (hard error)**: if `cfg.model_name` in the YAML doesn't match `preprocessing.json["tokenizer"]`, the train script bails before any GPU work — saved `input_ids` reference one vocabulary; loading a different model would train on nonsense embeddings.
- **`max_length` vs model capacity (soft warning)**: if the dataset was prepared with `max_length=2048` but the chosen model only supports `max_position_embeddings=512`, a warning fires after the model loads. Architectures with rotary/relative position embeddings are exempt.

Explicit constructor args (`max_length=...`) on `TextClassificationPredictor` still win — the metadata only fills in the gap when the caller didn't ask for anything.

## Resuming after a failure

Pass `--resume` to continue from the latest checkpoint automatically:

```bash
python scripts/text_classification/train_text_classification.py \
    --config configs/encoder/text_classification/emotion_bert.yaml \
    --resume
```

Or point at a specific checkpoint:

```bash
python scripts/text_classification/train_text_classification.py \
    --config configs/encoder/text_classification/emotion_bert.yaml \
    --resume outputs/emotion_bert/checkpoint-1000
```

## Inference

Load the trained model for high-performance inference:

```python
from lievito_madre_ai_lab.encoder.text_classification.serve import TextClassificationPredictor

predictor = TextClassificationPredictor("outputs/emotion_bert/final")

# batch
results = predictor.predict(["I love this!", "I'm furious."])
# [{"label": "joy", "score": 0.97, "scores": {"joy": 0.97, "anger": 0.01, ...}}, ...]

# single
result = predictor.predict_one("What a lovely day.")
```

Run from the CLI:

```bash
python -m lievito_madre_ai_lab.encoder.text_classification.serve \
    outputs/emotion_bert/final "I love this" "I'm so angry"
```

Benchmark throughput:

```bash
python -m lievito_madre_ai_lab.encoder.text_classification.serve \
    outputs/emotion_bert/final --benchmark
```

The predictor automatically applies the best available optimisations for the detected hardware:

| Hardware | Optimisations applied |
|---|---|
| CUDA (Ampere+) | FP16 weights · Flash Attention 2 · `torch.compile` · BF16 autocast |
| CUDA (older) | FP16 weights · SDPA · `torch.compile` · FP16 autocast |
| CPU | FP32 weights · dynamic INT8 quantisation of Linear layers |

## Switching model or dataset

Duplicate the YAML config and edit `model_name`, `processed_dir`, and `output_dir`:

```bash
cp configs/encoder/text_classification/emotion_bert.yaml \
   configs/encoder/text_classification/emotion_roberta.yaml
```

Then run with `--config configs/encoder/text_classification/emotion_roberta.yaml`.

## Bringing your own data

The training script only requires a tokenized Arrow dataset with three columns: `input_ids`, `attention_mask`, and `labels`. There are two paths to get there.

**Path 1 — your files already have `text` and `label` columns** (CSV, JSON, or Parquet):

Copy `examples/text_classification/prepare_emotion.py`, point it at your data, and run it:

```bash
python my_prepare_script.py \
    --source local \
    --local-path data/raw/my_dataset \
    --text-col my_text_column \
    --label-col my_label_column
```

**Path 2 — your data needs transformation first** (string labels → int, cleaning, custom splits, etc.):
Write a script that produces a `DatasetDict` and calls `tokenize_for_trainer`:

```python
from datasets import DatasetDict, Dataset
from lievito_madre_ai_lab.encoder.text_classification.dataset import tokenize_for_trainer

raw = DatasetDict({
    "train": Dataset.from_dict({"text": [...], "label": [...]}),
    "test":  Dataset.from_dict({"text": [...], "label": [...]}),
})

processed = tokenize_for_trainer(raw, model_name="bert-base-uncased")
processed.save_to_disk("data/processed/my_dataset")

# Write the sidecar so train/serve can rediscover the settings.
from lievito_madre_ai_lab.encoder.text_classification.dataset import save_preprocessing_meta
save_preprocessing_meta(
    "data/processed/my_dataset",
    tokenizer="bert-base-uncased",
    max_length=128,
    text_col="text",
    label_col="label",
)
```

Point `processed_dir` in your YAML config at `data/processed/my_dataset` and run the training script as normal. The `model_name` in the YAML must match the tokenizer string in the sidecar — the train script will refuse to start otherwise.
