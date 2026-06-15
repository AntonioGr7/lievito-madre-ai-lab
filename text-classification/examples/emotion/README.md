# Emotion Text Classification

Fine-tune an encoder for 6-way emotion classification on [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion). Default backbone: [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base) (8192-token context).

Metrics tracked at each epoch: `accuracy`, `f1` (weighted), and `f1_<class>` per label. Final test metrics land in `outputs/<run>/final/test_metrics.json`.

## Run

```bash
bash examples/emotion/run.sh
```

Or step by step:

```bash
python examples/emotion/dataset/prepare_emotion.py
python scripts/train_text_classification.py \
    --config examples/emotion/configs/emotion_bert.yaml
```
