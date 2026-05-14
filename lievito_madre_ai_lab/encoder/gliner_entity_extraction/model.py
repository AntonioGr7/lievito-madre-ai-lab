"""Load a GLiNER checkpoint and configure it for fine-tuning.

Single entry point: `load_gliner`. Responsibilities:

  1. Load the base model via `GLiNER.from_pretrained`.
  2. Stamp the entity vocabulary (train / holdout) on `model.config` so
     save/load round-trips preserve it.
  3. Stamp loss + negative-sampling knobs so `gliner.training.Trainer`
     and its collator pick them up at training time.
  4. Stamp `label_aliases` so the train/eval/serve paths can translate
     between canonical labels (in the data) and natural-language prompts
     (what GLiNER encodes).
  5. Optionally wrap the underlying encoder (`model.model`) with a PEFT
     LoRA adapter, leaving the span scoring head fully trainable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PeftConfig:
    enabled: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: Any = "auto"   # "auto" | list[str]


# Manual fallback target modules for backbones not in peft's auto-mapping.
# DeBERTa-v3 (used by gliner_multi-v2.5 / gliner-multitask) keys on these
# submodule names inside the disentangled-attention block.
_DEBERTA_V3_LORA_TARGETS = ["query_proj", "key_proj", "value_proj", "dense"]


def _resolve_lora_targets(model, requested) -> list[str]:
    """Resolve `peft.target_modules`: 'auto' → backbone-appropriate list."""
    if requested != "auto":
        return list(requested)

    # Try peft's mapping first.
    try:
        from peft.utils.other import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

        model_type = getattr(getattr(model, "config", None), "model_type", None)
        if model_type and model_type in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
            return list(TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_type])
    except Exception:
        pass

    return list(_DEBERTA_V3_LORA_TARGETS)


def load_gliner(
    model_name: str,
    *,
    train_types: list[str],
    holdout_types: list[str] | None = None,
    max_span_width: int | None = None,
    loss_cfg: dict | None = None,
    sampling_cfg: dict | None = None,
    label_aliases: dict[str, str] | None = None,
    peft_cfg: PeftConfig | None = None,
):
    """Load a GLiNER model and configure it for fine-tuning.

    All optional kwargs default to GLiNER's own defaults so passing only
    the required two (`model_name`, `train_types`) still produces a
    trainable model.
    """
    from gliner import GLiNER

    model = GLiNER.from_pretrained(model_name)

    # Vocabulary --------------------------------------------------------
    model.config.train_types = list(train_types)
    model.config.holdout_types = list(holdout_types or [])
    model.config.label_aliases = dict(label_aliases or {})

    if max_span_width is not None:
        model.config.max_width = int(max_span_width)

    # Loss --------------------------------------------------------------
    if loss_cfg:
        if "funcs" in loss_cfg:
            model.config.loss_funcs = list(loss_cfg["funcs"])
        if "focal_alpha" in loss_cfg:
            model.config.focal_alpha = float(loss_cfg["focal_alpha"])
        if "focal_gamma" in loss_cfg:
            model.config.focal_gamma = float(loss_cfg["focal_gamma"])

    # Negative-label sampling -------------------------------------------
    if sampling_cfg:
        if "max_types" in sampling_cfg:
            model.config.max_types = int(sampling_cfg["max_types"])
        if "max_neg_type_ratio" in sampling_cfg:
            model.config.max_neg_type_ratio = float(sampling_cfg["max_neg_type_ratio"])
        if "random_drop" in sampling_cfg:
            model.config.random_drop = float(sampling_cfg["random_drop"])

    # PEFT / LoRA -------------------------------------------------------
    if peft_cfg and peft_cfg.enabled:
        from peft import LoraConfig, get_peft_model

        targets = _resolve_lora_targets(model.model, peft_cfg.target_modules)
        lora = LoraConfig(
            r=peft_cfg.r,
            lora_alpha=peft_cfg.alpha,
            lora_dropout=peft_cfg.dropout,
            target_modules=targets,
            bias="none",
            task_type=None,   # GLiNER is custom; no HF task_type fits
        )
        model.model = get_peft_model(model.model, lora)
        model.config.peft_enabled = True
        model.config.peft_target_modules = targets

    return model
