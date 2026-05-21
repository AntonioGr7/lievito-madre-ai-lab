"""Dataset contracts for bi-encoder fine-tuning with `sentence-transformers`.

Sentence-Transformers v3+ wraps HF Trainer and reads columns positionally:
the *order* of columns in the dataset must match the loss function's
expected inputs. Column names themselves are free.

Four supported shapes (all column-count based):

- ``pair``           : 2 cols ``(anchor, positive)`` — MNRL with in-batch negatives.
- ``triplet``        : 3 cols ``(anchor, positive, negative)`` — MNRL + 1 hard neg.
- ``multi_negative`` : 4+ cols ``(anchor, positive, neg_1, …, neg_N)`` — output
                       of hard-negative mining. MNRL uses every column as a neg
                       source for the in-batch loss.
- ``distill``        : multi_negative + a trailing list-of-float ``label`` column
                       carrying cross-encoder teacher scores. Used by
                       `DistillKLDivLoss` for listwise distillation.

Shape detection looks at the *type* of the last column: if it's a
``Sequence[float]`` we're in distill territory, otherwise we count string
columns.
"""
from __future__ import annotations

from typing import Literal

from datasets import Dataset, DatasetDict, Sequence, Value

# Re-exported so prepare scripts can use a single import. The shared helper
# is task-agnostic on purpose — text/token/gliner pipelines all use it too.
from lievito_madre_ai_lab.shared.preprocessing import (  # noqa: F401
    PREPROCESSING_META_FILE,
    load_preprocessing_meta,
    save_preprocessing_meta,
)

BiEncoderShape = Literal["pair", "triplet", "multi_negative", "distill"]

PAIR_COLUMNS = ("anchor", "positive")
TRIPLET_COLUMNS = ("anchor", "positive", "negative")
LABEL_COLUMN = "label"  # convention for the teacher-score column in `distill`


def _is_label_column(feature) -> bool:
    """A `label` column for DistillKLDivLoss is a Sequence of floats."""
    if isinstance(feature, Sequence):
        inner = feature.feature
        return isinstance(inner, Value) and inner.dtype.startswith("float")
    return False


def infer_shape(dataset: Dataset) -> BiEncoderShape:
    """Detect the dataset shape from columns + dtypes.

    The last column is inspected first: a ``Sequence[float]`` indicates a
    distillation dataset. Otherwise every column must be string-typed and
    the count picks pair / triplet / multi_negative.
    """
    cols = dataset.column_names
    n = len(cols)
    if n < 2:
        raise ValueError(
            f"Bi-encoder dataset needs at least 2 columns; got {n}: {cols}."
        )

    last_feature = dataset.features[cols[-1]]
    if _is_label_column(last_feature):
        # All text columns before the label must be strings.
        for col in cols[:-1]:
            _require_string(dataset.features[col], col)
        if n - 1 < 3:
            raise ValueError(
                f"Distill dataset needs ≥3 text columns + 1 label; got "
                f"{n - 1} text columns: {cols[:-1]}."
            )
        return "distill"

    # No teacher-score column — ordinary text-only shape.
    for col in cols:
        _require_string(dataset.features[col], col)
    if n == 2:
        return "pair"
    if n == 3:
        return "triplet"
    return "multi_negative"


def _require_string(feature, col: str) -> None:
    dtype = getattr(feature, "dtype", None)
    if dtype != "string":
        raise ValueError(
            f"Column {col!r} has dtype {dtype!r}; bi-encoder text columns "
            f"must be 'string'. Cast non-string columns in the prepare script."
        )


def validate_dataset(datasets: DatasetDict) -> BiEncoderShape:
    """Validate a DatasetDict for bi-encoder training and return its shape.

    Splits must share a shape — mixing pair/triplet/multi_negative/distill
    across splits silently breaks the trainer (loss vs. evaluator look at
    different columns).
    """
    if "train" not in datasets:
        raise ValueError(
            f"Bi-encoder dataset must have a 'train' split; got {list(datasets.keys())}."
        )

    train_shape = infer_shape(datasets["train"])
    for split_name, split in datasets.items():
        shape = infer_shape(split)
        if shape != train_shape:
            raise ValueError(
                f"Split '{split_name}' has shape {shape!r} but 'train' has "
                f"{train_shape!r}. All splits must share the same shape."
            )
    return train_shape


def text_columns(dataset: Dataset) -> list[str]:
    """All columns except the trailing `label` (when present)."""
    cols = dataset.column_names
    if cols and _is_label_column(dataset.features[cols[-1]]):
        return cols[:-1]
    return cols
