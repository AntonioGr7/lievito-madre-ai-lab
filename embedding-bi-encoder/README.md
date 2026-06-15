# Embedding Bi-Encoder

Fine-tuning bi-encoders for semantic search and retrieval, built on
[sentence-transformers](https://www.sbert.net/) (v3+ runs on the HuggingFace `Trainer`).
Includes hard-negative mining, cross-encoder distillation, and retrieval evaluation. Runs are
configured by a single YAML file that loads into the
[`TrainConfig`](bi_encoder/shared/config.py) dataclass.

A self-contained project: it carries its own vendored copy of the lab's shared config and
preprocessing helpers under [`bi_encoder/shared/`](bi_encoder/shared/).

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # GPU
pip install -e ".[dev]"
```

## Quickstart

```bash
# Train on (anchor, positive[, negative]) pairs
python scripts/train_bi_encoder.py --config examples/smoke/configs/smoke_r1.yaml
# Mine hard negatives, then score them with a cross-encoder for distillation
python scripts/mine_hard_negatives.py  --help
python scripts/score_with_cross_encoder.py --help
```

See [scripts/README.md](scripts/README.md) for the full guide (mining, distillation, losses,
evaluation) and [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) for every YAML field.

## Layout

```
bi_encoder/   importable package (dataset, model, trainer, distill, mining, evaluate, serve
              + vendored shared/)
scripts/      train / inference / mining / scoring entry points
examples/     ready-to-run configs and dataset-prep scripts (smoke, financial, custom pairs)
tests/        unit tests
data/         processed datasets (gitignored)
```
