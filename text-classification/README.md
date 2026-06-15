# Text Classification

Encoder fine-tuning (BERT, RoBERTa, DeBERTa, ModernBERT, …) for single- and multi-class text
classification, driven by the HuggingFace `Trainer`. Every run is configured by a single YAML
file that loads into the [`TrainConfig`](text_classification/shared/config.py) dataclass.

A self-contained project: it carries its own vendored copy of the lab's shared config and
preprocessing helpers under [`text_classification/shared/`](text_classification/shared/), so it
has no dependency on the rest of the lab.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
# GPU: install torch first with the right CUDA index
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
```

## Quickstart

```bash
# 1. Prepare a dataset (example: dair-ai/emotion)
python examples/emotion/dataset/prepare_emotion.py
# 2. Train
python scripts/train_text_classification.py --config examples/emotion/configs/emotion_bert.yaml
```

See [scripts/README.md](scripts/README.md) for the full quickstart, resuming, inference, and
bring-your-own-data guides, and [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) for every YAML field.

## Layout

```
text_classification/   importable package (dataset, model, trainer, evaluate, serve + shared/)
scripts/               train / inference entry points
examples/              ready-to-run configs and dataset-prep scripts
tests/                 unit tests
data/                  tokenized datasets (gitignored)
```
