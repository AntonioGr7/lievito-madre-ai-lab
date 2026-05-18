"""Cross-encoder scoring for listwise distillation.

`DistillKLDivLoss` minimises the KL divergence between the teacher's
softmaxed score distribution over (query, candidate) pairs and the
student bi-encoder's. The teacher is a cross-encoder — much stronger and
much slower than the student. Scoring is an offline, one-time cost; the
student then trains on the soft labels at bi-encoder speed.

Input dataset shape: a multi-negative dataset
    ``(anchor, positive, neg_1, …, neg_N)``
(produced by `mine_hard_negatives_for_dataset` with ``output_format="n-tuple"``).

Output dataset shape: same columns plus a trailing ``label`` column
    ``label = [score(anchor, positive), score(anchor, neg_1), …]``
which `infer_shape` then classifies as the ``distill`` shape.

The teacher cross-encoder should be **stronger** than the student
bi-encoder (otherwise distillation just imitates a weaker model). Good
choices in 2026: BAAI/bge-reranker-v2-m3, mixedbread-ai/mxbai-rerank-large-v2,
Alibaba-NLP/gte-multilingual-reranker-base.

This also enables a SOTA trick referenced in the recipe: train a
cross-encoder on a small labelled set, label a large unlabelled pool with
it (silver labels), then distill into the bi-encoder. The bi-encoder
ends up stronger than what the small labelled set alone could produce.
"""
from __future__ import annotations

from dataclasses import dataclass

from datasets import Dataset, Sequence, Value
from sentence_transformers import CrossEncoder

from lievito_madre_ai_lab.embedding.bi_encoder.dataset import (
    LABEL_COLUMN,
    text_columns,
)


@dataclass
class ScoringConfig:
    batch_size: int = 64
    activation_fn: str | None = None  # "sigmoid" | "softmax" | None (raw logits)
    label_column: str = LABEL_COLUMN
    show_progress_bar: bool = True


def add_cross_encoder_scores(
    dataset: Dataset,
    *,
    cross_encoder: CrossEncoder,
    cfg: ScoringConfig | None = None,
) -> Dataset:
    """Append a teacher-score list column to a multi-negative dataset.

    The dataset must have ≥3 string columns (anchor + positive + ≥1 negative)
    and no trailing label column yet. The new ``label`` column is a list of
    floats: one score per candidate column, in the same column order.

    The activation matters: KL distillation operates on a *distribution*
    over candidates. Leaving scores as raw logits and applying a single
    softmax across the row preserves the relative scale (which is what
    KL needs); passing them through sigmoid first collapses dynamic range.
    `DistillKLDivLoss` softmaxes internally, so the recommended default is
    raw logits (`activation_fn=None`).
    """
    cfg = cfg or ScoringConfig()
    text_cols = text_columns(dataset)
    if len(text_cols) < 3:
        raise ValueError(
            f"Distillation needs ≥3 text columns (anchor, positive, ≥1 negative); "
            f"got {len(text_cols)}: {text_cols}."
        )
    if cfg.label_column in dataset.column_names:
        raise ValueError(
            f"Dataset already has a {cfg.label_column!r} column. "
            f"Drop it before re-scoring, or pick another label_column name."
        )

    anchor_col = text_cols[0]
    candidate_cols = text_cols[1:]
    n_candidates = len(candidate_cols)

    # Score every (anchor, candidate_col_i) pair across the full split, then
    # transpose so each row holds the per-candidate list the loss expects.
    per_candidate_scores: list[list[float]] = []
    for cand_col in candidate_cols:
        pairs = list(zip(dataset[anchor_col], dataset[cand_col]))
        scores = cross_encoder.predict(
            pairs,
            batch_size=cfg.batch_size,
            activation_fn=cfg.activation_fn,
            show_progress_bar=cfg.show_progress_bar,
        )
        per_candidate_scores.append([float(s) for s in scores])

    # Transpose: (n_candidates, n_rows) → (n_rows, n_candidates).
    n_rows = len(dataset)
    labels: list[list[float]] = [
        [per_candidate_scores[c][r] for c in range(n_candidates)]
        for r in range(n_rows)
    ]

    return dataset.add_column(
        cfg.label_column,
        labels,
        feature=Sequence(Value("float32")),
    )
