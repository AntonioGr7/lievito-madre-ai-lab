"""Trainer factory wired to `gliner.training.Trainer`.

Three responsibilities:

1. Resolve `cfg.precision` → (fp16, bf16, tf32) booleans for TrainingArguments.
2. Build an optimizer with two parameter groups (encoder vs head) so the
   head can train at a higher LR than the pre-trained encoder.
3. Wire the eval callback from `evaluate.build_eval_callback` so HF Trainer's
   eval loop reports a real `eval_f1` instead of just `eval_loss`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
# gliner.training.TrainingArguments is a subclass of transformers'
# TrainingArguments with extra fields (focal_loss_alpha, others_lr, …) that
# gliner.training.Trainer.compute_loss reads. Using transformers' base class
# crashes inside compute_loss with AttributeError.
from gliner.training import TrainingArguments

from lievito_madre_ai_lab.shared.config import TrainConfig


@dataclass
class GLiNERTrainCfg:
    """GLiNER knobs the *trainer* needs (loss-fn name + sampling go onto
    model.config via load_gliner; here we carry the trainer-side parameters
    that map onto gliner.training.TrainingArguments fields)."""
    max_span_width: int | None = None
    head_lr_multiplier: float = 5.0
    focal_loss_alpha: float = -1.0   # -1 = focal disabled (falls back to BCE)
    focal_loss_gamma: float = 0.0
    loss_reduction: str = "sum"


def _resolve_precision(precision: str) -> tuple[bool, bool, bool]:
    """Map `precision` string → (fp16, bf16, tf32) booleans for HF Trainer."""
    cuda = torch.cuda.is_available()
    
    # Check for Ampere (8.0) or newer (Hopper/Blackwell/etc)
    can_do_tf32 = False
    if cuda:
        major, _ = torch.cuda.get_device_capability(0)
        can_do_tf32 = (major >= 8)

    if precision == "auto":
        if cuda and can_do_tf32:
            return False, True, True  # BF16 + TF32 for Ampere
        if cuda:
            return True, False, False # FP16 for Turing/Pascal
        return False, False, False

    if precision == "bf16":
        # BF16 also requires Ampere+
        return False, True, can_do_tf32
    
    if precision == "fp16":
        # Even if on CUDA, only enable TF32 if the chip supports it
        return True, False, can_do_tf32 
        
    if precision == "fp32":
        return False, False, False
        
    raise ValueError(f"unknown precision: {precision!r}")


def build_training_args(
    cfg: TrainConfig,
    *,
    num_training_steps: int | None = None,
    gliner_cfg: GLiNERTrainCfg | None = None,
) -> TrainingArguments:
    warmup_steps = 0
    if cfg.warmup_ratio > 0:
        if num_training_steps is None:
            raise ValueError(
                "build_training_args requires num_training_steps when cfg.warmup_ratio > 0"
            )
        warmup_steps = math.ceil(num_training_steps * cfg.warmup_ratio)

    fp16, bf16, tf32 = _resolve_precision(cfg.precision)

    gc = gliner_cfg or GLiNERTrainCfg()
    others_lr = cfg.learning_rate * gc.head_lr_multiplier

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
        torch_compile=cfg.torch_compile,
        dataloader_num_workers=cfg.dataloader_num_workers,
        logging_steps=cfg.logging_steps,
        report_to=cfg.report_to,
        remove_unused_columns=False,
        # GLiNER-specific fields (subclass of HF TrainingArguments)
        others_lr=others_lr,
        others_weight_decay=cfg.weight_decay,
        focal_loss_alpha=gc.focal_loss_alpha,
        focal_loss_gamma=gc.focal_loss_gamma,
        loss_reduction=gc.loss_reduction,
    )


def build_trainer(
    model,
    datasets,
    training_args: TrainingArguments,
    *,
    gliner_cfg: GLiNERTrainCfg,
    train_types: list[str],
    early_stopping_patience: int | None = None,
    eval_threshold: float = 0.5,
):
    """Build a `gliner.training.Trainer` configured for this project."""
    # gliner_cfg fields end up on training_args (others_lr, focal_loss_alpha,
    # …) earlier via build_training_args. Kept on the signature so callers can
    # still pass it; reserved as a hook for future trainer-only knobs.
    _ = gliner_cfg
    from transformers import EarlyStoppingCallback
    # gliner 0.2.x: Trainer is exposed at gliner.training; if a release
    # moves it, fall back to the module path.
    try:
        from gliner.training import Trainer as GLiNERTrainer
    except ImportError:
        from gliner.training.trainer import Trainer as GLiNERTrainer
    from gliner.data_processing.collator import SpanDataCollator, TokenDataCollator
    from gliner.data_processing.processor import (
        BiEncoderSpanProcessor,
        BiEncoderTokenProcessor,
        UniEncoderSpanProcessor,
        UniEncoderTokenProcessor,
    )

    from lievito_madre_ai_lab.encoder.gliner_entity_extraction.evaluate import (
        build_eval_callback,
    )
    from lievito_madre_ai_lab.encoder.gliner_entity_extraction.dataset import (
        to_gliner_native,
    )

    eval_split = "validation" if "validation" in datasets else "test"

    # Char-offset rows → GLiNER native (tokenized_text + ner). The original
    # text/spans columns are preserved (remove_unused_columns=False) so the
    # eval callback can still consume them.
    splitter = model.data_processor.words_splitter
    native_datasets = type(datasets)({
        name: split.map(
            lambda r: to_gliner_native(r, splitter),
            desc=f"Converting {name} -> GLiNER native",
        )
        for name, split in datasets.items()
    })

    processor = model.data_processor
    if isinstance(processor, (UniEncoderSpanProcessor, BiEncoderSpanProcessor)):
        collator_cls = SpanDataCollator
    elif isinstance(processor, (UniEncoderTokenProcessor, BiEncoderTokenProcessor)):
        collator_cls = TokenDataCollator
    else:
        raise TypeError(
            f"unexpected data_processor type {type(processor).__name__}"
        )
    data_collator = collator_cls(model.config, data_processor=processor)

    # Discriminative LR is handled inside gliner.training.Trainer via the
    # args.others_lr / args.others_weight_decay fields we set in
    # build_training_args; no custom optimizer needed here.

    callbacks = [
        build_eval_callback(
            datasets[eval_split],
            train_types=train_types,
            threshold=eval_threshold,
            batch_size=training_args.per_device_eval_batch_size,
        )
    ]
    if early_stopping_patience and early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    return GLiNERTrainer(
        model=model,
        args=training_args,
        train_dataset=native_datasets["train"],
        eval_dataset=native_datasets[eval_split],
        data_collator=data_collator,
        callbacks=callbacks,
    )
