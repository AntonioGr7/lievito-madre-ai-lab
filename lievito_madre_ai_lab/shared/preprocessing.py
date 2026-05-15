"""Sidecar metadata describing how a dataset was prepared.

Both prepare-* scripts write ``preprocessing.json`` next to the processed
dataset, the train script copies it next to the saved model, and the
predictor reads it on load so inference defaults to the exact settings used
at training time (tokenizer, max_length, sliding-window stride, …).

Kept task-agnostic on purpose: text- and token-classification share this
file format, and a third pipeline added later only needs to call the same
two helpers.
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


def warn_if_max_length_exceeds_model_capacity(meta: dict, *, model_config) -> None:
    """Soft warning when the dataset was tokenized at a longer max_length than
    the model's positional capacity supports.

    Catches the symmetric mistake to the tokenizer check: e.g. bumping
    ``--max-length`` to 2048 at prep time but leaving the YAML pointing at
    mDeBERTa (512). Without this, training fails much later with a cryptic
    position-id out-of-range error from inside the model forward pass.

    Architectures that don't report a static cap — T5 with relative bias,
    models with rotary embeddings that can extrapolate — are skipped because
    a one-off override is legitimate there.
    """
    prep_max = meta.get("max_length")
    if not prep_max:
        return
    model_max = getattr(model_config, "max_position_embeddings", None)
    if not model_max or model_max <= 0:
        return
    if prep_max > model_max:
        name = getattr(model_config, "_name_or_path", "?")
        print(
            f"[warn] dataset tokenized with max_length={prep_max} but "
            f"{name} reports max_position_embeddings={model_max}. Training "
            f"will likely crash with a position-id out-of-range error. "
            f"Either re-prep with --max-length <={model_max} or pick a "
            f"longer-context model."
        )


def assert_tokenizer_matches(meta: dict, *, model_name: str) -> None:
    """Fail fast when the model picked at training time uses a different
    tokenizer than the one the dataset was tokenized with.

    A silent vocabulary mismatch is the single worst failure mode in this
    pipeline: input_ids saved by the prep script reference one vocabulary,
    the loaded model embeds them against a different one, training looks
    fine but produces nonsense. The metadata file exists specifically to
    catch this — we just have to read it.
    """
    expected = meta.get("tokenizer")
    if expected and expected != model_name:
        raise ValueError(
            f"Tokenizer mismatch between dataset prep and training config:\n"
            f"  dataset was tokenized with: {expected!r}\n"
            f"  training config says:        {model_name!r}\n"
            f"input_ids in the processed dataset reference {expected}'s "
            f"vocabulary; loading {model_name} would train against nonsense "
            f"embeddings. Either re-run the prep script with "
            f"--model {model_name}, or change model_name in the YAML to "
            f"{expected}."
        )
