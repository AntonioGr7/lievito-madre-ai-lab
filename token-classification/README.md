# Token Classification (NER)

Encoder fine-tuning for token classification / named-entity recognition (BERT, mDeBERTa,
XLM-R, ModernBERT, …) with the HuggingFace `Trainer`. Runs are configured by a single YAML
file that loads into the [`TrainConfig`](token_classification/shared/config.py) dataclass.

A self-contained project: it carries its own vendored copy of the lab's shared config,
preprocessing, and data-source helpers under
[`token_classification/shared/`](token_classification/shared/). The
[`pii_corpus/`](token_classification/pii_corpus/) subpackage assembles a combined PII NER
corpus from several public sources.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # GPU
pip install -e ".[dev]"
```

## Quickstart

```bash
# 1. Prepare a dataset (example: ai4privacy open-pii)
python examples/pii/dataset/prepare_openpii.py
# 2. Train
python scripts/train_token_classification.py --config examples/pii/configs/pii_mbert.yaml
```

See [scripts/README.md](scripts/README.md) for resuming, inference and bring-your-own-data
guides, and [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) for every YAML field.

## Layout

```
token_classification/   importable package (dataset, model, trainer, evaluate, serve,
                        pii_corpus/ + vendored shared/)
scripts/                train / inference entry points
examples/               configs and dataset-prep scripts
tests/                  unit tests
data/                   tokenized datasets (gitignored)
```
