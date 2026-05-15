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
```

Point `processed_dir` in your YAML config at `data/processed/my_dataset` and run the training script as normal.
