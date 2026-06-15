"""Sidecar metadata describing how a model was trained / a dataset prepared.

The prepare script writes ``preprocessing.json`` next to the processed dataset
(processor-agnostic provenance: source, categories), the train script augments
it with the train-time choices and copies it next to the saved model, and the
predictor reads it on load so inference defaults to the settings used at
training time (image-processor id, image size, default score threshold,
id2label).

Kept task-agnostic on purpose: this is the same sidecar format the other lab
projects use.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PREPROCESSING_META_FILE = "preprocessing.json"


def save_preprocessing_meta(out_dir: str | Path, **fields: Any) -> Path:
    """Persist preprocessing settings as ``<out_dir>/preprocessing.json``."""
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
    image processor than the one recorded in the model directory's metadata.

    A detector's resize/normalize/pad policy and box post-processing come from
    its image processor; loading a checkpoint trained for processor A against
    the weights of model B silently garbles inputs (wrong normalization, wrong
    box decoding). The metadata file exists to catch this — we just read it.
    """
    expected = meta.get("processor") or meta.get("model_name")
    if expected and expected != model_name:
        raise ValueError(
            f"Image-processor mismatch between the saved model and the requested base:\n"
            f"  metadata recorded:  {expected!r}\n"
            f"  you asked to load:  {model_name!r}\n"
            f"Resize/normalize and box decoding differ across detector families; "
            f"loading mismatched weights serves against garbled inputs. Load the "
            f"recorded base, or re-train with model_name={model_name}."
        )
