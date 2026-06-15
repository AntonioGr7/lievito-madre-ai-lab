"""Trainer factory for DETR-family detector fine-tuning.

Wires up the SOTA training recipe:

1. ``precision`` → (fp16, bf16, tf32) for ``TrainingArguments`` autocast.
2. A discriminative-LR ``AdamW`` (backbone at a fraction of the head LR — see
   :func:`object_detection.model.build_param_groups`).
3. Separate augmenting (train) and clean (eval) collators on one Trainer.
4. Optional **weight EMA**: a shadow copy updated each optimizer step, swapped in
   for evaluation *and* checkpointing, so the metric you select on and the model
   you ship are both the EMA weights — typically worth +0.5-1.5 AP for free.
5. The COCO-mAP eval callback from ``evaluate.build_eval_callback``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import TrainingArguments

from object_detection.dataset import build_transforms, make_collate_fn
from object_detection.model import build_param_groups
from object_detection.shared.config import TrainConfig


@dataclass
class DetectionCfg:
    """Detector-specific knobs from the YAML ``detection:`` block."""
    image_size: int | None = None
    freeze_backbone: bool = False
    backbone_lr_mult: float = 0.1
    # augmentation
    aug_enabled: bool = True
    hflip: float = 0.5
    brightness_contrast: float = 0.5
    hue_sat: float = 0.3
    # EMA
    ema_enabled: bool = True
    ema_decay: float = 0.9997
    # eval
    eval_threshold: float = 0.0      # keep all queries for mAP
    eval_max_samples: int | None = None
    # serve default (persisted to preprocessing.json)
    score_threshold: float = 0.3
    trust_remote_code: bool = False


def _resolve_precision(precision: str) -> tuple[bool, bool, bool]:
    """Map ``precision`` string → (fp16, bf16, tf32) booleans for HF Trainer."""
    cuda = torch.cuda.is_available()
    can_do_tf32 = False
    if cuda:
        major, _ = torch.cuda.get_device_capability(0)
        can_do_tf32 = major >= 8

    if precision == "auto":
        if cuda and can_do_tf32:
            return False, True, True
        if cuda:
            return True, False, False
        return False, False, False
    if precision == "bf16":
        return False, True, can_do_tf32
    if precision == "fp16":
        return True, False, can_do_tf32
    if precision == "fp32":
        return False, False, False
    raise ValueError(f"unknown precision: {precision!r}")


def build_training_args(cfg: TrainConfig, *, num_training_steps: int | None = None) -> TrainingArguments:
    import math

    warmup_steps = 0
    if cfg.warmup_ratio > 0:
        if num_training_steps is None:
            raise ValueError("build_training_args requires num_training_steps when warmup_ratio > 0")
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
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.gradient_checkpointing else None,
        torch_compile=cfg.torch_compile,
        dataloader_num_workers=cfg.dataloader_num_workers,
        logging_steps=cfg.logging_steps,
        report_to=cfg.report_to,
        remove_unused_columns=False,   # keep image/objects for the collator
        label_names=["labels"],
    )


class ModelEma:
    """Exponential moving average of model weights.

    Tracks every floating-point entry of ``state_dict`` (params + float buffers
    such as BatchNorm stats). ``copy_to``/``restore`` swap the shadow weights in
    and out in place, so the *same* model object can be evaluated/saved as EMA
    and then resume training from its live weights.
    """

    def __init__(self, model, decay: float = 0.9997) -> None:
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }
        self._backup: dict = {}

    @torch.no_grad()
    def update(self, model) -> None:
        for k, v in model.state_dict().items():
            s = self.shadow.get(k)
            if s is not None:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model) -> None:
        msd = model.state_dict()
        self._backup = {k: msd[k].detach().clone() for k in self.shadow}
        for k, s in self.shadow.items():
            msd[k].copy_(s.to(msd[k].dtype))

    @torch.no_grad()
    def restore(self, model) -> None:
        msd = model.state_dict()
        for k, v in self._backup.items():
            msd[k].copy_(v)
        self._backup = {}


def _ema_update_callback(ema: ModelEma):
    from transformers import TrainerCallback

    class _EmaCallback(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **_):
            if model is not None:
                ema.update(model)

    return _EmaCallback()


def _make_detection_trainer_cls():
    """Build the Trainer subclass lazily so importing this module doesn't import
    the (heavy) Trainer until a trainer is actually constructed."""
    from transformers import Trainer

    class _DetectionTrainer(Trainer):
        """Trainer that (a) uses distinct train/eval collators and (b) swaps EMA
        weights in around evaluation and checkpointing."""

        def __init__(self, *args, ema=None, train_collator=None, eval_collator=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.ema = ema
            self._train_collator = train_collator
            self._eval_collator = eval_collator

        def get_train_dataloader(self):
            if self._train_collator is not None:
                self.data_collator = self._train_collator
            return super().get_train_dataloader()

        def get_eval_dataloader(self, eval_dataset=None):
            if self._eval_collator is not None:
                self.data_collator = self._eval_collator
            return super().get_eval_dataloader(eval_dataset)

        def evaluate(self, *args, **kwargs):
            if self.ema is not None:
                self.ema.copy_to(self.model)
            try:
                return super().evaluate(*args, **kwargs)
            finally:
                if self.ema is not None:
                    self.ema.restore(self.model)

        def _save_checkpoint(self, *args, **kwargs):
            # Persist EMA weights into the checkpoint so load_best_model_at_end
            # restores EMA — keeping "what we selected on" == "what we ship".
            if self.ema is not None:
                self.ema.copy_to(self.model)
            try:
                return super()._save_checkpoint(*args, **kwargs)
            finally:
                if self.ema is not None:
                    self.ema.restore(self.model)

    return _DetectionTrainer


def build_trainer(
    model,
    image_processor,
    datasets,
    training_args: TrainingArguments,
    *,
    det_cfg: DetectionCfg,
    cfg: TrainConfig,
    id2label: dict[int, str],
    early_stopping_patience: int | None = None,
):
    """Build the detection Trainer: discriminative-LR AdamW, EMA, mAP callback."""
    from transformers import EarlyStoppingCallback

    from object_detection.evaluate import build_eval_callback

    eval_split = "validation" if "validation" in datasets else None

    train_transforms = build_transforms(
        train=True, hflip=det_cfg.hflip,
        brightness_contrast=det_cfg.brightness_contrast, hue_sat=det_cfg.hue_sat,
    ) if det_cfg.aug_enabled else None
    train_collator = make_collate_fn(image_processor, train_transforms)
    eval_collator = make_collate_fn(image_processor, None)

    # Discriminative-LR optimizer; let the Trainer build the scheduler from it.
    optimizer = torch.optim.AdamW(
        build_param_groups(
            model, base_lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
            backbone_lr_mult=det_cfg.backbone_lr_mult,
        ),
        lr=cfg.learning_rate,
    )

    ema = ModelEma(model, decay=det_cfg.ema_decay) if det_cfg.ema_enabled else None

    callbacks = []
    if eval_split is not None:
        callbacks.append(
            build_eval_callback(
                datasets[eval_split],
                image_processor=image_processor,
                id2label=id2label,
                batch_size=training_args.per_device_eval_batch_size,
                threshold=det_cfg.eval_threshold,
                max_samples=det_cfg.eval_max_samples,
            )
        )
    if ema is not None:
        callbacks.append(_ema_update_callback(ema))
    if early_stopping_patience and early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer_cls = _make_detection_trainer_cls()
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets[eval_split] if eval_split else None,
        data_collator=train_collator,
        processing_class=image_processor,
        optimizers=(optimizer, None),
        callbacks=callbacks,
        ema=ema,
        train_collator=train_collator,
        eval_collator=eval_collator,
    )
    return trainer, ema
