from __future__ import annotations

from typing import Callable

import evaluate
import numpy as np


def build_compute_metrics(label_names: list[str] | None = None) -> Callable:
    """Return a compute_metrics function compatible with HF Trainer.

    Tracks:
    - accuracy
    - f1 (weighted average)
    - f1_<class> per label (e.g. f1_joy, f1_anger, …)
    """
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred) -> dict:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        weighted_f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
        per_class_f1 = f1_metric.compute(predictions=predictions, references=labels, average=None)["f1"]

        results = {
            "accuracy": acc["accuracy"],
            "f1": weighted_f1["f1"],
        }

        for i, score in enumerate(per_class_f1):
            key = label_names[i] if label_names else str(i)
            results[f"f1_{key}"] = round(float(score), 4)

        return results

    return compute_metrics
