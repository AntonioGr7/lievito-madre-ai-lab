from __future__ import annotations

from typing import Callable

import evaluate


def build_compute_metrics(label_names: list[str]) -> Callable:
    """Return a compute_metrics function compatible with HF Trainer.

    Uses seqeval for entity-level evaluation (span-based, not token-based):
    precision, recall, f1, accuracy — all computed over full entity spans.

    Tokens labelled -100 (special tokens and non-first subwords) are
    automatically excluded from evaluation.

    Expects ``predictions`` to be label ids (post-argmax), produced by the
    Trainer's ``preprocess_logits_for_metrics`` hook — accumulating raw
    float logits across the full eval set OOMs on a 16 GB GPU.
    """
    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_pred) -> dict:
        predictions, labels = eval_pred

        true_labels = [
            [label_names[l] for l in row if l != -100]
            for row in labels
        ]
        true_preds = [
            [label_names[p] for p, l in zip(pred_row, label_row) if l != -100]
            for pred_row, label_row in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_metrics
