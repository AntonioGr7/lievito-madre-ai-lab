"""COCO mAP evaluation + the TrainerCallback that runs it.

Eval is the real COCO metric — ``torchmetrics.detection.MeanAveragePrecision``
(which wraps pycocotools): mAP averaged over IoU 0.50:0.95, plus mAP@.50, mAP@.75,
size-stratified mAP, and per-class AP. We run inference over the eval split,
decode predictions with the model's image processor (boxes back in original
pixel coordinates), and accumulate the metric.

We evaluate at a **score threshold of 0.0** so every query's scored box reaches
the metric — mAP integrates over the precision/recall curve, so suppressing
low-confidence boxes up front would only depress recall and bias the number.

``build_eval_callback`` wraps :func:`compute_map` in a ``TrainerCallback`` firing
on ``on_evaluate``, so ``metric_for_best_model="map"`` and ``EarlyStoppingCallback``
work normally. (The actual eval is done here, in the callback — the same pattern
the other lab projects use for metrics the Trainer can't compute from logits.)
"""
from __future__ import annotations

from typing import Any

from object_detection.dataset import coco_to_xyxy

# torchmetrics emits -1 for a metric it has no data for (e.g. no large objects
# in the split); surface that verbatim rather than faking a 0.
_MAP_KEYS = ["map", "map_50", "map_75", "map_small", "map_medium", "map_large",
             "mar_1", "mar_10", "mar_100"]


def _f(v) -> float:
    """torchmetrics scalar (tensor or number) → float."""
    try:
        return float(v.item())
    except AttributeError:
        return float(v)


def summarize_map(result: dict, id2label: dict[int, str] | None = None) -> dict[str, float]:
    """Flatten a ``MeanAveragePrecision.compute()`` dict into ``{metric: float}``.

    Keeps the standard COCO summary keys and, when per-class metrics are present,
    expands ``map_per_class`` into ``map_<ClassName>`` using *id2label*. Pure /
    side-effect-free so it can be unit-tested without a model.
    """
    out: dict[str, float] = {k: _f(result[k]) for k in _MAP_KEYS if k in result}

    per_class = result.get("map_per_class")
    classes = result.get("classes")
    if id2label is not None and per_class is not None and classes is not None:
        per_list = per_class.tolist() if hasattr(per_class, "tolist") else per_class
        cls_list = classes.tolist() if hasattr(classes, "tolist") else classes
        # torchmetrics returns a scalar (not a list) when exactly one class is present.
        if not isinstance(per_list, list):
            per_list = [per_list]
        if not isinstance(cls_list, list):
            cls_list = [cls_list]
        for cls_id, ap in zip(cls_list, per_list):
            name = id2label.get(int(cls_id), str(cls_id))
            out[f"map_{name}"] = _f(ap)
    return out


def _gold_targets(row: dict):
    """Build a torchmetrics target dict (xyxy pixel boxes + labels) for *row*."""
    import torch

    bboxes = row["objects"]["bbox"]
    cats = row["objects"]["category_id"]
    if bboxes:
        boxes = torch.tensor([coco_to_xyxy(b) for b in bboxes], dtype=torch.float32)
        labels = torch.tensor([int(c) for c in cats], dtype=torch.long)
    else:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.long)
    return {"boxes": boxes, "labels": labels}


def compute_map(
    model,
    image_processor,
    dataset,
    *,
    id2label: dict[int, str] | None = None,
    batch_size: int = 8,
    threshold: float = 0.0,
    max_samples: int | None = None,
    progress: bool = False,
) -> dict[str, float]:
    """Run inference over *dataset* and return COCO mAP metrics.

    Boxes are decoded back to each image's original pixel size, so predictions
    and gold are scored in the same coordinate space.
    """
    import torch
    from torchmetrics.detection import MeanAveragePrecision

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    device = next(model.parameters()).device
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=id2label is not None)

    indices = range(0, len(dataset), batch_size)
    if progress:
        try:
            from tqdm.auto import tqdm
            indices = tqdm(indices, desc="mAP eval", unit="batch")
        except Exception:
            pass

    was_training = model.training
    model.eval()
    for start in indices:
        rows = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        images = [r["image"].convert("RGB") for r in rows]
        target_sizes = torch.tensor([[im.height, im.width] for im in images])
        enc = image_processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**enc)
        results = image_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes,
        )
        preds = [
            {"boxes": r["boxes"].cpu().float(),
             "scores": r["scores"].cpu().float(),
             "labels": r["labels"].cpu().long()}
            for r in results
        ]
        targets = [_gold_targets(r) for r in rows]
        metric.update(preds, targets)
    if was_training:
        model.train()

    return summarize_map(metric.compute(), id2label)


def build_eval_callback(
    eval_dataset,
    *,
    image_processor,
    id2label: dict[int, str],
    prefix: str = "eval",
    batch_size: int = 8,
    threshold: float = 0.0,
    max_samples: int | None = None,
):
    """TrainerCallback that populates ``metrics[<prefix>_*]`` from :func:`compute_map`."""
    from transformers import TrainerCallback

    class _EvalCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, model=None, metrics=None, **_):
            if model is None or metrics is None:
                return
            results = compute_map(
                model, image_processor, eval_dataset,
                id2label=id2label, batch_size=batch_size,
                threshold=threshold, max_samples=max_samples,
            )
            prefixed = {f"{prefix}_{k}": v for k, v in results.items()}
            metrics.update(prefixed)

            scalar_extras = {k: v for k, v in prefixed.items() if isinstance(v, (int, float))}
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(scalar_extras, step=state.global_step)
            except ImportError:
                pass
            headline = {
                k: scalar_extras[k]
                for k in (f"{prefix}_map", f"{prefix}_map_50", f"{prefix}_map_75")
                if k in scalar_extras
            }
            if headline:
                print(
                    f"[eval @ step {state.global_step}] "
                    + " ".join(f"{k}={v:.4f}" for k, v in headline.items())
                )

    return _EvalCallback()
