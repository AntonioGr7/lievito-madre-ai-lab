# GLiNER Entity Extraction

Fine-tuning [GLiNER](https://github.com/urchade/GLiNER) for open-vocabulary / zero-shot named
entity recognition. Full fine-tuning and LoRA are both supported; runs are configured by a
single YAML file that loads into the [`TrainConfig`](gliner_entity_extraction/shared/config.py)
dataclass.

A self-contained project: it carries its own vendored copy of the lab's shared config and
preprocessing helpers under
[`gliner_entity_extraction/shared/`](gliner_entity_extraction/shared/).

> **Isolated environment required.** `gliner` 0.2.x pins `transformers<5.2.0`, which conflicts
> with the other lab projects. Always install this project in its own virtualenv.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # GPU
pip install -e ".[dev]"
```

## Quickstart

```bash
# 1. Prepare a dataset
python examples/pii/dataset/prepare_openpii.py
# 2. Train
python scripts/train_gliner.py --config examples/pii/configs/pii_gliner.yaml
# Zero-shot baseline (no training)
python scripts/baseline_zeroshot.py --config examples/pii/configs/smoke.yaml
```

See [scripts/README.md](scripts/README.md) for the full guide and
[CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) for every YAML field.

## Tests

```bash
pytest                       # fast unit tests
RUN_GLINER_SMOKE=1 pytest    # also runs the end-to-end smoke test (~30-60s CPU)
```

The smoke test uses a tiny GLiNER dataset under `tests/fixtures/gliner_tiny/`. It is **generated,
not committed** — `tests/conftest.py` rebuilds it automatically on first use from
[tests/fixtures/build_gliner_tiny.py](tests/fixtures/build_gliner_tiny.py) (whose ~10 rows are
defined inline and easy to read/edit). To (re)build it manually:

```bash
python tests/fixtures/build_gliner_tiny.py
```

## Layout

```
gliner_entity_extraction/   importable package (dataset, model, trainer, evaluate, serve
                            + vendored shared/)
scripts/                    train / inference / zero-shot-baseline entry points
examples/                   configs and dataset-prep scripts
tests/                      unit tests (+ auto-built tests/fixtures/gliner_tiny, gitignored)
data/                       processed datasets (gitignored)
```
