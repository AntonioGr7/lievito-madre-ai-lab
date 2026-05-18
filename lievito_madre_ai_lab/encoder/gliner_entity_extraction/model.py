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

import types
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


def _find_hf_encoder(model):
    """Locate the HuggingFace ``PreTrainedModel`` buried inside GLiNER.

    GLiNER wraps the encoder several levels deep::

        model (outer GLiNER wrapper)
          .model                -> UniEncoderSpanModel / sibling
            .token_rep_layer    -> Encoder / BiEncoder
              .bert_layer       -> Transformer
                .model          -> HF PreTrainedModel  (GC lives here)

    PEFT, when enabled in :func:`load_gliner`, wraps ``model.model`` with a
    ``PeftModel`` whose attribute access forwards to the wrapped module, so
    the same path still resolves.
    """
    cur = getattr(model, "model", None)
    for attr in ("token_rep_layer", "bert_layer", "model"):
        if cur is None:
            break
        cur = getattr(cur, attr, None)
    if cur is not None and hasattr(cur, "gradient_checkpointing_enable"):
        return cur
    # Fallback: if a future GLiNER release moves things, accept any nn.Module
    # along the chain that exposes the method.
    for candidate in (getattr(model, "model", None),):
        if candidate is not None and hasattr(candidate, "gradient_checkpointing_enable"):
            return candidate
    raise AttributeError(
        "Could not locate a PreTrainedModel under the GLiNER wrapper for "
        "gradient checkpointing. Looked at "
        "model.model.token_rep_layer.bert_layer.model and model.model."
    )


def _retarget_span_rep_max_width(model, new_max_width: int) -> None:
    """Propagate ``max_width`` into every submodule that cached it at init.

    The span representation layer stamps ``self.max_width`` in its
    ``__init__`` for two distinct uses depending on the variant:

    - *reshape-only* (``SpanMarker``, ``SpanMarkerV1``): used solely as the
      reshape dim in ``out.view(B, L, max_width, D)``. Safe to retarget — no
      pretrained parameters are tied to the old value.
    - *parameter-bound* (``SpanMarkerV0``, ``SpanConv*``): also indexes
      ``nn.Parameter`` tensors (``query_seg``, ``conv_weigth``). Changing the
      width here would invalidate the pretrained weights, so we raise
      instead of silently re-initialising.

    Without this, setting ``model.config.max_width`` propagates to the data
    processor (which produces ``L * max_width`` candidate spans) but not to
    the layer's cached attribute, and the reshape blows up at the first
    forward call.
    """
    affected: list[str] = []
    for module in model.modules():
        old = getattr(module, "max_width", None)
        if not isinstance(old, int) or old == new_max_width:
            continue
        bound_params = [
            (name, tuple(p.shape))
            for name, p in module.named_parameters(recurse=False)
            if any(d == old for d in p.shape)
        ]
        if bound_params:
            raise ValueError(
                f"Cannot change max_span_width from {old} to {new_max_width}: "
                f"{type(module).__name__} has pretrained parameters whose "
                f"shape is tied to the old width: {bound_params}. Either keep "
                f"max_span_width={old}, or pick a model whose span_rep layer "
                f"is reshape-only (SpanMarker / SpanMarkerV1)."
            )
        module.max_width = new_max_width
        affected.append(type(module).__name__)
    if affected:
        print(
            f"      [load_gliner] propagated max_width={new_max_width} to "
            f"{len(affected)} submodule(s): {affected}"
        )


def _attach_gradient_checkpointing(model) -> None:
    """Make the GLiNER wrapper respond to HF Trainer's GC hook.

    HF Trainer calls ``self.model.gradient_checkpointing_enable(...)`` on the
    outer GLiNER wrapper, which is a plain ``nn.Module``. We attach thin
    delegates that forward to the underlying HF encoder, disable ``use_cache``
    (incompatible with GC), and call ``enable_input_require_grads`` so that
    when LoRA is in play autograd still reaches the adapters through the
    frozen embedding table.
    """
    def _enable(self, gradient_checkpointing_kwargs=None):
        inner = _find_hf_encoder(self)
        if gradient_checkpointing_kwargs is None:
            inner.gradient_checkpointing_enable()
        else:
            inner.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        if hasattr(inner, "config"):
            inner.config.use_cache = False
        if hasattr(inner, "enable_input_require_grads"):
            inner.enable_input_require_grads()

    def _disable(self):
        inner = _find_hf_encoder(self)
        if hasattr(inner, "gradient_checkpointing_disable"):
            inner.gradient_checkpointing_disable()
        if hasattr(inner, "config"):
            inner.config.use_cache = True

    model.gradient_checkpointing_enable = types.MethodType(_enable, model)
    model.gradient_checkpointing_disable = types.MethodType(_disable, model)


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
        new_w = int(max_span_width)
        model.config.max_width = new_w
        _retarget_span_rep_max_width(model, new_w)

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

    _attach_gradient_checkpointing(model)
    return model
