"""Grounding metrics + the TrainerCallback that runs them.

Eval is generation-based: the model decodes a ``label<box> ... </box>`` string,
we parse it back to structured objects, and match predictions to gold by label +
geometry. For boxes that's IoU matching (a prediction is a true positive when it
shares the gold's label and overlaps it at ``iou_threshold``); for points it's
point-in-gold-box (the standard "pointing accuracy" criterion). Greedy one-to-one
matching prevents one prediction from claiming several gold boxes.

We report exact-match P/R/F1 at the operating IoU plus a COCO-style ``f1_iou_avg``
averaged over IoU 0.50:0.95. We deliberately do *not* report mAP: free-form
generation yields no calibrated per-box confidence to rank by, so an
AP-under-the-PR-curve number would be meaningless. F1 at a fixed IoU is the
honest metric for a generative detector.

``build_eval_callback`` wraps :func:`evaluate_split` in a ``TrainerCallback`` that
fires on ``on_evaluate``, so ``metric_for_best_model="eval_f1"`` and
``EarlyStoppingCallback`` work normally.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from vlm_finetuning.dataset import iou, point_in_box

_WS = re.compile(r"\s+")


def _normalize_text(s: str) -> str:
    """Whitespace-normalize for text-target comparison (collapse runs, strip)."""
    return _WS.sub(" ", str(s)).strip()


def score_text(pred_per_row: list[str], gold_per_row: list[str]) -> dict[str, float]:
    """Metrics for the free-form ``task="text"`` path.

    Returns ``exact_match`` (fraction of rows whose whitespace-normalized output
    equals gold) and ``token_f1`` (mean SQuAD-style multiset-token F1 — a softer
    signal that credits near-misses, e.g. one wrong field in a JSON object).
    The honest default for generic SFT; swap in a task-specific scorer (JSON
    structural match, BLEU, …) by editing this function.
    """
    if len(pred_per_row) != len(gold_per_row):
        raise ValueError(
            f"pred/gold length mismatch: {len(pred_per_row)} vs {len(gold_per_row)}"
        )
    n = len(gold_per_row) or 1
    exact = 0
    f1_sum = 0.0
    for pred, gold in zip(pred_per_row, gold_per_row):
        p, g = _normalize_text(pred), _normalize_text(gold)
        if p == g:
            exact += 1
        pt, gt = Counter(p.split()), Counter(g.split())
        overlap = sum((pt & gt).values())
        if overlap == 0:
            continue
        precision = overlap / max(sum(pt.values()), 1)
        recall = overlap / max(sum(gt.values()), 1)
        f1_sum += 2 * precision * recall / (precision + recall)
    return {"exact_match": exact / n, "token_f1": f1_sum / n, "n": len(gold_per_row)}

# Default L2 radius (normalized image coords) within which a predicted point is
# credited to a gold point, when gold carries a point rather than a box.
DEFAULT_POINT_DIST = 0.05
# COCO's IoU sweep, used for the averaged-F1 summary on the box task.
_COCO_IOUS = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def _pair_score(pred: dict, gold: dict, task: str, point_dist: float) -> float | None:
    """Geometric agreement of a same-label (pred, gold) pair, or ``None`` if the
    pair can't be compared. Higher is better; used both to gate matches and to
    order greedy matching."""
    if task == "box":
        if "box" not in pred or "box" not in gold:
            return None
        return iou(pred["box"], gold["box"])
    # point task
    if "point" not in pred:
        return None
    if "box" in gold:
        return 1.0 if point_in_box(pred["point"], gold["box"]) else 0.0
    if "point" in gold:
        px, py = pred["point"]
        gx, gy = gold["point"]
        dist = ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5
        return (1.0 - dist / point_dist) if dist <= point_dist else 0.0
    return None


def _count_matches(
    preds: list[dict], golds: list[dict], *, task: str, iou_threshold: float, point_dist: float,
) -> int:
    """Greedy one-to-one matches between *preds* and *golds* at *iou_threshold*."""
    candidates: list[tuple[float, int, int]] = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(golds):
            if p.get("label") != g.get("label"):
                continue
            s = _pair_score(p, g, task, point_dist)
            if s is None or s <= 0.0:
                continue
            if task == "box" and s < iou_threshold:
                continue
            candidates.append((s, pi, gi))
    candidates.sort(key=lambda c: c[0], reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    for _, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
    return len(used_p)


def score_objects(
    pred_per_image: list[list[dict]],
    gold_per_image: list[list[dict]],
    *,
    task: str = "box",
    iou_threshold: float = 0.5,
    point_dist: float = DEFAULT_POINT_DIST,
    labels: list[str] | None = None,
) -> dict[str, float]:
    """Micro-aggregated P/R/F1 over the corpus, plus per-label breakdown.

    For the box task also returns ``f1_iou_avg`` — F1 averaged over IoU
    0.50:0.95 (the COCO sweep), a single number that rewards tight boxes.
    """
    if len(pred_per_image) != len(gold_per_image):
        raise ValueError(
            f"pred/gold length mismatch: {len(pred_per_image)} vs {len(gold_per_image)}"
        )

    tp = fp = fn = 0
    per_label: dict[str, dict[str, int]] = {}
    for preds, golds in zip(pred_per_image, gold_per_image):
        m = _count_matches(preds, golds, task=task, iou_threshold=iou_threshold, point_dist=point_dist)
        tp += m
        fp += len(preds) - m
        fn += len(golds) - m
        if labels:
            for lbl in labels:
                pl = [p for p in preds if p.get("label") == lbl]
                gl = [g for g in golds if g.get("label") == lbl]
                ml = _count_matches(pl, gl, task=task, iou_threshold=iou_threshold, point_dist=point_dist)
                d = per_label.setdefault(lbl, {"tp": 0, "fp": 0, "fn": 0})
                d["tp"] += ml
                d["fp"] += len(pl) - ml
                d["fn"] += len(gl) - ml

    precision, recall, f1 = _prf(tp, fp, fn)
    out: dict[str, Any] = {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
    }

    if task == "box":
        f1s = []
        for thr in _COCO_IOUS:
            t = f_ = n_ = 0
            for preds, golds in zip(pred_per_image, gold_per_image):
                mm = _count_matches(preds, golds, task=task, iou_threshold=thr, point_dist=point_dist)
                t += mm
                f_ += len(preds) - mm
                n_ += len(golds) - mm
            f1s.append(_prf(t, f_, n_)[2])
        out["f1_iou_avg"] = sum(f1s) / len(f1s) if f1s else 0.0

    for lbl, d in per_label.items():
        p, r, fm = _prf(d["tp"], d["fp"], d["fn"])
        out[f"precision_{lbl}"] = p
        out[f"recall_{lbl}"] = r
        out[f"f1_{lbl}"] = fm
    return out


def _gold_objects(row: dict) -> list[dict]:
    """Pull a row's gold objects (already normalized to [0,1])."""
    return list(row.get("objects", []) or [])


def evaluate_split(
    model,
    processor,
    dataset,
    *,
    task: str = "box",
    coord_bins: int = 1000,
    iou_threshold: float = 0.5,
    labels: list[str] | None = None,
    system_prompt: str | None = None,
    max_new_tokens: int = 512,
    batch_size: int = 8,
    max_samples: int | None = None,
    progress: bool = False,
) -> dict[str, float]:
    """Generate over *dataset*, parse, and score grounding P/R/F1.

    Each row supplies its own ``prompt`` (the instruction) and gold ``objects``.
    ``max_samples`` caps how many rows are generated on (eval-time generation is
    the expensive part); ``None`` evaluates the whole split.
    """
    # Lazy import: generation needs torch + the model wrapper, but the pure
    # scoring functions above stay import-light for unit tests.
    from vlm_finetuning.serve import generate_grounding, generate_texts

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    images = [row["image"] for row in dataset]
    prompts = [row["prompt"] for row in dataset]

    if task == "text":
        gold = [str(row.get("response", "")) for row in dataset]
        preds = generate_texts(
            model, processor, images, prompts,
            system_prompt=system_prompt, max_new_tokens=max_new_tokens,
            batch_size=batch_size, progress=progress,
        )
        return score_text(preds, gold)

    gold_per_image = [_gold_objects(row) for row in dataset]
    pred_per_image = generate_grounding(
        model, processor, images, prompts,
        system_prompt=system_prompt,
        coord_bins=coord_bins,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        progress=progress,
    )
    return score_objects(
        pred_per_image, gold_per_image,
        task=task, iou_threshold=iou_threshold, labels=labels,
    )


def build_eval_callback(
    eval_dataset,
    *,
    processor,
    labels: list[str],
    task: str = "box",
    coord_bins: int = 1000,
    system_prompt: str | None = None,
    prefix: str = "eval",
    iou_threshold: float = 0.5,
    max_new_tokens: int = 512,
    batch_size: int = 8,
    max_samples: int | None = None,
):
    """TrainerCallback that populates ``metrics[<prefix>_*]`` from
    :func:`evaluate_split` (generation-based grounding F1)."""
    from transformers import TrainerCallback

    class _EvalCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, model=None, metrics=None, **_):
            if model is None or metrics is None:
                return
            import torch

            was_training = model.training
            model.eval()
            with torch.inference_mode():
                results = evaluate_split(
                    model, processor, eval_dataset,
                    task=task, coord_bins=coord_bins, iou_threshold=iou_threshold,
                    labels=labels, system_prompt=system_prompt,
                    max_new_tokens=max_new_tokens, batch_size=batch_size,
                    max_samples=max_samples,
                )
            if was_training:
                model.train()

            prefixed = {f"{prefix}_{k}": v for k, v in results.items()}
            metrics.update(prefixed)

            # HF Trainer logs `metrics` before this callback fires, so the keys
            # we just added would be invisible mid-training (best-model
            # selection still sees them via the mutated dict). Push the scalars
            # out-of-band so the operator sees the F1 slope; W&B merges by step.
            scalar_extras = {k: v for k, v in prefixed.items() if isinstance(v, (int, float))}
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(scalar_extras, step=state.global_step)
            except ImportError:
                pass
            if task == "text":
                headline_keys = [f"{prefix}_exact_match", f"{prefix}_token_f1"]
            else:
                headline_keys = [f"{prefix}_precision", f"{prefix}_recall", f"{prefix}_f1"]
                if f"{prefix}_f1_iou_avg" in scalar_extras:
                    headline_keys.append(f"{prefix}_f1_iou_avg")
            headline = {k: scalar_extras[k] for k in headline_keys if k in scalar_extras}
            if headline:
                print(
                    f"[eval @ step {state.global_step}] "
                    + " ".join(f"{k}={v:.4f}" for k, v in headline.items())
                )

    return _EvalCallback()
