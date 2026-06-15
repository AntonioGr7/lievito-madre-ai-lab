"""Evaluator factory for bi-encoder training.

Sentence-Transformers reports metrics via `SentenceEvaluator` objects rather
than HF's `compute_metrics`. We pick the evaluator from the dataset shape:

| Shape           | Evaluator                       | Primary metric                       |
|-----------------|---------------------------------|--------------------------------------|
| triplet         | TripletEvaluator                | cosine_accuracy                      |
| pair            | InformationRetrievalEvaluator   | cosine_mrr@k / ndcg@k / recall@k     |
| multi_negative  | InformationRetrievalEvaluator   | (anchor → positive, IR-style)        |
| distill         | InformationRetrievalEvaluator   | (anchor → positive, IR-style)        |

For multi_negative / distill we ignore the mined hard negatives at eval
time: their role is during the *training* loss, but for "did the model
learn anything?" the right signal is whether the anchor still retrieves
its true positive against the corpus of all positives. The result is a
metric that's directly comparable across all shapes.
"""
from __future__ import annotations

from datasets import Dataset
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SentenceEvaluator,
    TripletEvaluator,
)

from bi_encoder.dataset import (
    BiEncoderShape,
    infer_shape,
    text_columns,
)


def build_evaluator(
    dataset: Dataset,
    *,
    name: str = "eval",
    ir_top_k: int = 10,
    batch_size: int = 64,
) -> tuple[SentenceEvaluator, BiEncoderShape, str]:
    """Build a SentenceEvaluator that matches the dataset shape.

    Returns `(evaluator, shape, primary_metric_key)`. The primary metric key
    is the suggested value for `metric_for_best_model` in the YAML.
    """
    shape = infer_shape(dataset)
    cols = text_columns(dataset)

    if shape == "triplet":
        evaluator = TripletEvaluator(
            anchors=dataset[cols[0]],
            positives=dataset[cols[1]],
            negatives=dataset[cols[2]],
            name=name,
            batch_size=batch_size,
        )
        return evaluator, shape, f"eval_{name}_cosine_accuracy"

    # pair / multi_negative / distill — always anchor (col 0) → positive (col 1).
    # Build an IR eval where the corpus is the unique set of positives and the
    # gold for each anchor is its own positive.
    anchors = dataset[cols[0]]
    positives = dataset[cols[1]]

    corpus: dict[str, str] = {}
    positive_to_cid: dict[str, str] = {}
    for pos in positives:
        if pos not in positive_to_cid:
            cid = f"c{len(positive_to_cid)}"
            positive_to_cid[pos] = cid
            corpus[cid] = pos

    queries: dict[str, str] = {f"q{i}": q for i, q in enumerate(anchors)}
    relevant_docs: dict[str, set[str]] = {
        f"q{i}": {positive_to_cid[pos]} for i, pos in enumerate(positives)
    }

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        batch_size=batch_size,
        mrr_at_k=[ir_top_k],
        ndcg_at_k=[ir_top_k],
        accuracy_at_k=[ir_top_k],
        precision_recall_at_k=[ir_top_k],
        map_at_k=[ir_top_k],
        corpus_chunk_size=50_000,
        show_progress_bar=False,
    )
    return evaluator, shape, f"eval_{name}_cosine_mrr@{ir_top_k}"
