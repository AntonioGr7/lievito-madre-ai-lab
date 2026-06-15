"""Sidecar metadata describing how a model was trained / a dataset prepared.

The prepare script writes ``preprocessing.json`` next to the processed dataset
(processor-agnostic fields: source, task, coordinate scheme), the train script
augments it with the train-time choices and copies it next to the saved model,
and the predictor reads it on load so inference defaults to the exact settings
used at training time (processor id, coordinate bins, default prompt,
generation length, …).

Kept task-agnostic on purpose: this is the same sidecar format the other lab
projects use, so a new pipeline only needs to call the same two helpers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PREPROCESSING_META_FILE = "preprocessing.json"


def save_preprocessing_meta(out_dir: str | Path, **fields: Any) -> Path:
    """Persist preprocessing settings as ``<out_dir>/preprocessing.json``.

    Use it from prepare-* scripts to record what the dataset was built with,
    and from the trainer to forward the same record into the final model dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / PREPROCESSING_META_FILE
    path.write_text(json.dumps(fields, indent=2, sort_keys=True))
    return path


def load_preprocessing_meta(in_dir: str | Path) -> dict | None:
    """Return the metadata for a processed-dataset or saved-model directory,
    or ``None`` if no ``preprocessing.json`` is present (legacy artefacts)."""
    path = Path(in_dir) / PREPROCESSING_META_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text())


def assert_processor_matches(meta: dict, *, model_name: str) -> None:
    """Fail fast when the base model picked at training time uses a different
    processor than the one recorded in the model directory's metadata.

    A vision-language model's image-token expansion and chat template come from
    its processor; loading a checkpoint trained for processor A against the
    weights of model B produces silently garbled inputs (wrong number of image
    placeholder tokens, wrong special tokens). The metadata file exists
    specifically to catch this — we just have to read it.
    """
    expected = meta.get("processor") or meta.get("model_name")
    if expected and expected != model_name:
        raise ValueError(
            f"Processor mismatch between the saved model and the requested base:\n"
            f"  metadata recorded:  {expected!r}\n"
            f"  you asked to load:  {model_name!r}\n"
            f"The image-token layout and chat template differ across model "
            f"families; loading mismatched weights trains/serves against "
            f"garbled inputs. Load the recorded base, or re-train with "
            f"model_name={model_name}."
        )
