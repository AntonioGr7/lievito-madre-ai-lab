"""Load a DETR-family detector + image processor and configure it for fine-tuning.

Single entry point: :func:`load_detector`. Responsibilities:

  1. Load the image processor (``AutoImageProcessor``), optionally overriding the
     input resolution.
  2. Load the model (``AutoModelForObjectDetection``) with the new dataset's
     ``id2label`` / ``label2id`` and ``ignore_mismatched_sizes=True`` so the
     pretrained classification head is swapped for one sized to the new classes
     while every other weight (backbone, encoder, decoder) is kept.
  3. Optionally freeze the backbone for fast, low-memory fine-tuning.

Backbone-agnostic: works for D-FINE, RT-DETR / RT-DETRv2, Deformable DETR,
Conditional DETR, DETR, YOLOS — anything ``AutoModelForObjectDetection`` loads.

:func:`build_param_groups` implements the canonical DETR-family trick of a
**lower learning rate for the pretrained backbone** than for the freshly-adapted
detection head — the single biggest lever on fine-tuning stability.
"""
from __future__ import annotations

from typing import Any


def resolve_torch_dtype(precision: str):
    """Map a precision string → a torch dtype (used by the predictor for load)."""
    import torch

    if precision == "fp32":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def load_detector(
    model_name: str,
    *,
    id2label: dict[int, str],
    image_size: int | None = None,
    freeze_backbone: bool = False,
    attn_implementation: str | None = None,
    trust_remote_code: bool = False,
):
    """Load a detector + processor configured for fine-tuning on ``id2label``.

    The model is loaded in fp32 (DETR-family Hungarian matching is sensitive to
    low-precision logits); mixed precision is applied by the Trainer's autocast,
    not at load. Returns ``(model, image_processor)``.
    """
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    label2id = {v: k for k, v in id2label.items()}

    proc_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if image_size is not None:
        # DETR-family processors take a square {"height","width"}; this also
        # bounds memory (image-token / feature-map size scales with it).
        proc_kwargs["size"] = {"height": int(image_size), "width": int(image_size)}
    image_processor = AutoImageProcessor.from_pretrained(model_name, **proc_kwargs)

    model_kwargs: dict[str, Any] = {
        "id2label": id2label,
        "label2id": label2id,
        "num_labels": len(id2label),
        # Swap the pretrained class head for one sized to the new vocabulary,
        # keeping backbone + encoder + decoder weights.
        "ignore_mismatched_sizes": True,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForObjectDetection.from_pretrained(model_name, **model_kwargs)

    if freeze_backbone:
        frozen = _freeze_backbone(model)
        if frozen:
            print(f"      [load_detector] froze backbone ({frozen} tensors)")

    return model, image_processor


def _freeze_backbone(model) -> int:
    """Freeze every parameter under the model's backbone. Returns tensor count."""
    n = 0
    for name, p in model.named_parameters():
        if "backbone" in name and p.requires_grad:
            p.requires_grad_(False)
            n += 1
    return n


def _is_no_decay(name: str) -> bool:
    """Biases and every kind of norm (Layer/Batch/Group) get no weight decay —
    the standard rule; decaying them hurts and is never helped."""
    low = name.lower()
    return name.endswith(".bias") or "norm" in low or ".bn" in low or low.endswith("bn")


def build_param_groups(
    model,
    *,
    base_lr: float,
    weight_decay: float,
    backbone_lr_mult: float = 0.1,
) -> list[dict]:
    """Four AdamW parameter groups: {backbone, head} × {decay, no-decay}.

    The backbone trains at ``base_lr * backbone_lr_mult`` (typically 0.1×) and
    the rest at ``base_lr`` — the canonical DETR/RT-DETR/D-FINE schedule. Norms
    and biases are excluded from weight decay in every group. Frozen params are
    skipped, so this composes with ``freeze_backbone``.
    """
    groups: list[dict] = []
    for is_backbone in (True, False):
        for is_decay in (True, False):
            params = [
                p for n, p in model.named_parameters()
                if p.requires_grad
                and (("backbone" in n) == is_backbone)
                and ((not _is_no_decay(n)) == is_decay)
            ]
            if not params:
                continue
            groups.append({
                "params": params,
                "lr": base_lr * (backbone_lr_mult if is_backbone else 1.0),
                "weight_decay": weight_decay if is_decay else 0.0,
            })
    return groups
