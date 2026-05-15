from __future__ import annotations

import math

import torch
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from lievito_madre_ai_lab.shared.config import TrainConfig


def _resolve_precision(precision: str) -> tuple[bool, bool, bool]:
    """Resolve `precision` config string to (fp16, bf16, tf32) booleans.

    auto: bf16 on cap>=8.0 CUDA, else fp16 on CUDA, else fp32 on CPU.
    Explicit values map 1:1; tf32 is enabled whenever bf16/fp16 are on CUDA.
    """
    cuda = torch.cuda.is_available()
    cap_major = torch.cuda.get_device_capability(0)[0] if cuda else 0
    # TF32 needs Ampere+ (cap>=8). Turing (T4, cap=7.5) rejects tf32=True.
    can_do_tf32 = cuda and cap_major >= 8

    if precision == "auto":
        if can_do_tf32:
            return False, True, True   # bf16
        if cuda:
            return True, False, False  # fp16
        return False, False, False     # fp32

    if precision == "bf16":
        return False, True, can_do_tf32
    if precision == "fp16":
        return True, False, can_do_tf32
    if precision == "fp32":
        return False, False, False
    raise ValueError(f"unknown precision: {precision!r}")


def _argmax_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Trainer caches the output of this hook instead of the raw logits.
    # Returning argmax shrinks per-batch eval memory by ~num_labels× and
    # prevents OOM in evaluation_loop's nested_concat on long sequences.
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def build_training_args(
    cfg: TrainConfig,
    *,
    num_training_steps: int | None = None,
) -> TrainingArguments:
    # warmup_ratio is deprecated in transformers; convert to warmup_steps here.
    warmup_steps = 0
    if cfg.warmup_ratio > 0:
        if num_training_steps is None:
            raise ValueError(
                "build_training_args requires num_training_steps when cfg.warmup_ratio > 0. "
                "Compute it with shared.config.compute_total_training_steps()."
            )
        warmup_steps = math.ceil(num_training_steps * cfg.warmup_ratio)

    _fp16, _bf16, _tf32 = _resolve_precision(cfg.precision)

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
        fp16=_fp16,
        bf16=_bf16,
        tf32=_tf32,
        gradient_checkpointing=cfg.gradient_checkpointing,
        torch_compile=cfg.torch_compile,
        train_sampling_strategy=cfg.train_sampling_strategy,
        dataloader_num_workers=cfg.dataloader_num_workers,
        logging_steps=cfg.logging_steps,
        report_to=cfg.report_to,
    )


def build_trainer(
    model: PreTrainedModel,
    datasets: DatasetDict,
    tokenizer: AutoTokenizer,
    training_args: TrainingArguments,
    compute_metrics,
    *,
    pad_to_multiple_of: int | None = 8,
    early_stopping_patience: int | None = None,
) -> Trainer:
    eval_split = "validation" if "validation" in datasets else "test"

    callbacks = []
    if early_stopping_patience and early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets[eval_split],
        processing_class=tokenizer,
        # Pads sequences AND label sequences to batch-local max length, rounded
        # up to pad_to_multiple_of so fp16/bf16 tensor cores stay engaged.
        data_collator=DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of=pad_to_multiple_of
        ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=_argmax_logits,
        callbacks=callbacks or None,
    )
