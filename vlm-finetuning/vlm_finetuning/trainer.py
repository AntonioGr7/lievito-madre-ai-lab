"""Trainer factory wired to HuggingFace ``Trainer`` for VLM SFT.

Three responsibilities:

1. Resolve ``cfg.precision`` → (fp16, bf16, tf32) booleans for TrainingArguments.
2. Build a ``Trainer`` with our completion-only collator and ``remove_unused_columns=False``
   (so the ``image`` / ``objects`` / ``prompt`` columns survive to the collator).
3. Wire the generation-based eval callback from ``evaluate.build_eval_callback``
   so HF Trainer's eval loop reports a real ``eval_f1`` (grounding F1 measured on
   *decoded* predictions) instead of only ``eval_loss``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from transformers import TrainingArguments

from vlm_finetuning.dataset import DEFAULT_COORD_BINS, VLMDataCollator
from vlm_finetuning.shared.config import TrainConfig


@dataclass
class GroundingCfg:
    """Knobs the collator and the eval callback share — the coordinate scheme,
    the prompt scaffolding, and how generation-based eval is run."""
    task: str = "box"                       # "box" | "point" | "text"
    coord_bins: int = DEFAULT_COORD_BINS
    system_prompt: str | None = None
    empty_text: str = "No objects detected."
    max_length: int | None = None           # truncate over-long packed sequences
    eval_max_new_tokens: int = 512
    eval_iou_threshold: float = 0.5
    eval_max_samples: int | None = None      # cap eval-time generation cost


def _resolve_precision(precision: str) -> tuple[bool, bool, bool]:
    """Map ``precision`` string → (fp16, bf16, tf32) booleans for HF Trainer."""
    cuda = torch.cuda.is_available()

    can_do_tf32 = False
    if cuda:
        major, _ = torch.cuda.get_device_capability(0)
        can_do_tf32 = major >= 8

    if precision == "auto":
        if cuda and can_do_tf32:
            return False, True, True   # BF16 + TF32 for Ampere+
        if cuda:
            return True, False, False  # FP16 for Turing/T4
        return False, False, False

    if precision == "bf16":
        return False, True, can_do_tf32
    if precision == "fp16":
        return True, False, can_do_tf32
    if precision == "fp32":
        return False, False, False

    raise ValueError(f"unknown precision: {precision!r}")


def build_training_args(
    cfg: TrainConfig,
    *,
    num_training_steps: int | None = None,
) -> TrainingArguments:
    warmup_steps = 0
    if cfg.warmup_ratio > 0:
        if num_training_steps is None:
            raise ValueError(
                "build_training_args requires num_training_steps when cfg.warmup_ratio > 0"
            )
        warmup_steps = math.ceil(num_training_steps * cfg.warmup_ratio)

    fp16, bf16, tf32 = _resolve_precision(cfg.precision)

    return TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=warmup_steps,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim=cfg.optim,
        seed=cfg.seed,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        eval_accumulation_steps=cfg.eval_accumulation_steps,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        save_total_limit=cfg.save_total_limit,
        fp16=fp16, bf16=bf16, tf32=tf32,
        gradient_checkpointing=cfg.gradient_checkpointing,
        # Non-reentrant checkpointing is required for PEFT (the reentrant path
        # can drop adapter gradients) and is the modern default anyway.
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.gradient_checkpointing else None,
        torch_compile=cfg.torch_compile,
        dataloader_num_workers=cfg.dataloader_num_workers,
        logging_steps=cfg.logging_steps,
        report_to=cfg.report_to,
        # Keep the raw image/objects/prompt columns — the collator needs them.
        remove_unused_columns=False,
        # The Trainer can't infer label names from a multimodal batch dict.
        label_names=["labels"],
    )


def build_trainer(
    model,
    processor,
    datasets,
    training_args: TrainingArguments,
    *,
    grounding_cfg: GroundingCfg,
    labels: list[str],
    early_stopping_patience: int | None = None,
):
    """Build a HuggingFace ``Trainer`` configured for VLM grounding SFT."""
    from transformers import EarlyStoppingCallback, Trainer

    from vlm_finetuning.evaluate import build_eval_callback

    eval_split = "validation" if "validation" in datasets else None

    collator = VLMDataCollator(
        processor,
        task=grounding_cfg.task,
        coord_bins=grounding_cfg.coord_bins,
        max_length=grounding_cfg.max_length,
        system_prompt=grounding_cfg.system_prompt,
        empty_text=grounding_cfg.empty_text,
    )

    callbacks = []
    if eval_split is not None:
        callbacks.append(
            build_eval_callback(
                datasets[eval_split],
                processor=processor,
                labels=labels,
                task=grounding_cfg.task,
                coord_bins=grounding_cfg.coord_bins,
                system_prompt=grounding_cfg.system_prompt,
                iou_threshold=grounding_cfg.eval_iou_threshold,
                max_new_tokens=grounding_cfg.eval_max_new_tokens,
                batch_size=training_args.per_device_eval_batch_size,
                max_samples=grounding_cfg.eval_max_samples,
            )
        )
    if early_stopping_patience and early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets[eval_split] if eval_split else None,
        data_collator=collator,
        processing_class=processor,
        callbacks=callbacks,
    )
