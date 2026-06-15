# Lievito Madre AI LAB

Lievito Madre isn't just an ingredient; it is a living, evolving ecosystem of knowledge. Here is
where we cook.

This repository is a **lab of independent projects**. Each use case is fully self-contained — its
own `setup.py`, package, `scripts/`, `examples/`, `tests/`, `data/`, and a vendored copy of the
shared config/preprocessing helpers — so any one of them could be lifted out into a standalone
repository with no changes. Install each in its **own virtualenv** (their dependency sets are
deliberately isolated; GLiNER in particular pins an older `transformers` than the rest).

## Projects

| Project | What it does |
|---|---|
| [text-classification/](text-classification/) | Encoder fine-tuning for single-/multi-class text classification. |
| [token-classification/](token-classification/) | Encoder fine-tuning for token classification / NER (incl. a PII corpus builder). |
| [gliner-entity-extraction/](gliner-entity-extraction/) | Fine-tuning GLiNER for open-vocabulary / zero-shot entity extraction. |
| [embedding-bi-encoder/](embedding-bi-encoder/) | Bi-encoder fine-tuning for semantic search (sentence-transformers), with mining + distillation. |
| [vlm-finetuning/](vlm-finetuning/) | LoRA/QLoRA fine-tuning of vision-language models (image→text SFT): free-form targets (tool calls, JSON) or grounding (labels + `<box>` coordinate tokens). |
| [object-detection/](object-detection/) | Fine-tuning canonical DETR-family detectors (D-FINE, RT-DETRv2, …) with COCO mAP eval, discriminative LR, and weight EMA. |

Future use cases (LLM fine-tuning, …) get added the same way: a new top-level
project directory with its own `setup.py`.

## Utilities (`utils/`)

Support code that isn't a model-training use case:

| Path | What it is |
|---|---|
| [utils/tools/](utils/tools/) | Loose, dependency-light dev scripts (e.g. `data_preview.py`, an Arrow dataset previewer). |
| [utils/pipelines/bi_encoder_dataset/](utils/pipelines/bi_encoder_dataset/) | Installable pipeline that synthesises `(anchor, positive)` bi-encoder training datasets from raw documents via an LLM. Feeds `embedding-bi-encoder`. |

Each pipeline under `utils/pipelines/<name>/` is its own self-contained, installable project
(own `setup.py` + venv); add new ones as new named subdirectories.

## Getting started

Pick a project and follow its README:

```bash
cd text-classification
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # GPU
pip install -e ".[dev]"
```

## Repo conventions

- Each project carries a private `<package>/shared/` copy of the lab's `config.py` /
  `preprocessing.py` (and `sources.py` where used). The copies are intentionally independent — a
  fix to shared logic must be applied in each project that vendors it.
- `utils/tools/data_preview.py` — a dependency-free Arrow dataset previewer, handy across projects.
- Datasets and model weights are never committed (`data/`, `outputs/` are gitignored per project).
