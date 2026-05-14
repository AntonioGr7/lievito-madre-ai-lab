"""Strict char-offset entity-set F1 + the TrainerCallback that runs it.

Eval metric is the standard "did the model predict the same (start, end, label)
triples as gold?" comparison. No BIO projection — predictions and gold come
back in char-offset space already.

`build_eval_callback` wraps `evaluate_split` in a `TrainerCallback` that fires
on `on_evaluate`, so `metric_for_best_model="eval_f1"` and `EarlyStoppingCallback`
work normally.
"""
from __future__ import annotations

from typing import Any


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def score_predictions(
    pred_per_text: list[list[dict]],
    gold_per_text: list[list[dict]],
    *,
    labels: list[str] | None = None,
) -> dict[str, float]:
    """Strict exact-match P/R/F1 on (start, end, label) char-offset sets.

    Returns micro-aggregated `precision`, `recall`, `f1`, `tp`, `fp`, `fn`
    plus per-label `f1_<LABEL>`, `precision_<LABEL>`, `recall_<LABEL>` if
    *labels* is provided.
    """
    if len(pred_per_text) != len(gold_per_text):
        raise ValueError(
            f"pred/gold length mismatch: {len(pred_per_text)} vs {len(gold_per_text)}"
        )

    def _triples(span_list: list[dict]) -> set[tuple[int, int, str]]:
        return {(int(s["start"]), int(s["end"]), s["label"]) for s in span_list}

    tp = fp = fn = 0
    per_label: dict[str, dict[str, int]] = {}

    for pred, gold in zip(pred_per_text, gold_per_text):
        pred_set = _triples(pred)
        gold_set = _triples(gold)
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

        if labels:
            for lbl in labels:
                pred_l = {t for t in pred_set if t[2] == lbl}
                gold_l = {t for t in gold_set if t[2] == lbl}
                d = per_label.setdefault(lbl, {"tp": 0, "fp": 0, "fn": 0})
                d["tp"] += len(pred_l & gold_l)
                d["fp"] += len(pred_l - gold_l)
                d["fn"] += len(gold_l - pred_l)

    precision, recall, f1 = _prf(tp, fp, fn)
    out: dict[str, Any] = {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
    }
    for lbl, d in per_label.items():
        p, r, fm = _prf(d["tp"], d["fp"], d["fn"])
        out[f"precision_{lbl}"] = p
        out[f"recall_{lbl}"] = r
        out[f"f1_{lbl}"] = fm
    return out


def evaluate_split(
    model,
    dataset,
    labels: list[str],
    *,
    threshold: float = 0.5,
    batch_size: int = 16,
    label_aliases: dict[str, str] | None = None,
) -> dict[str, float]:
    """Run GLiNER inference over *dataset* and score char-offset entity sets.

    `dataset` rows must follow the char-offset contract from dataset.py:
    `{"text": str, "spans": [{"start", "end", "label"}, ...]}`.

    `label_aliases` maps canonical → prompt labels. The prompt is built
    from aliased labels; predictions are un-aliased before scoring so the
    per-label breakdown uses canonical names.
    """
    aliases = label_aliases or {}
    reverse = {v: k for k, v in aliases.items()}
    prompt_labels = [aliases.get(lbl, lbl) for lbl in labels]

    texts = [row["text"] for row in dataset]
    gold_per_text = [list(row["spans"]) for row in dataset]

    pred_per_text: list[list[dict]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        batch = model.batch_predict_entities(chunk, prompt_labels, threshold=threshold)
        for spans in batch:
            normalised = [
                {
                    "start": int(s["start"]),
                    "end": int(s["end"]),
                    "label": reverse.get(s["label"], s["label"]),
                }
                for s in spans
            ]
            pred_per_text.append(normalised)

    return score_predictions(pred_per_text, gold_per_text, labels=labels)


def build_eval_callback(
    eval_dataset,
    train_types: list[str],
    *,
    prefix: str = "eval",
    threshold: float = 0.5,
    batch_size: int = 16,
):
    """Build a TrainerCallback that populates `metrics[<prefix>_*]` from evaluate_split."""
    from transformers import TrainerCallback

    class _EvalCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, model=None, metrics=None, **_):
            if model is None or metrics is None:
                return
            import torch
            model.eval()
            with torch.inference_mode():
                aliases = getattr(model.config, "label_aliases", {}) or {}
                results = evaluate_split(
                    model,
                    eval_dataset,
                    labels=train_types,
                    threshold=threshold,
                    batch_size=batch_size,
                    label_aliases=aliases,
                )
            for k, v in results.items():
                metrics[f"{prefix}_{k}"] = v

    return _EvalCallback()


# Kept for backwards compatibility with train_gliner.py's import. New code
# should call `evaluate_split` directly.
def build_compute_metrics(train_types: list[str]):
    def _compute_metrics(eval_pred):
        return {}
    return _compute_metrics
