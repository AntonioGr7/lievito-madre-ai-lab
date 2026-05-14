from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    """Hyperparameters and paths for an encoder fine-tuning run.

    Load from YAML:
        cfg = load_config("configs/encoder/text_classification/emotion_bert.yaml")

    All fields map 1-to-1 to TrainingArguments / DataCollator / from_pretrained
    so the trainer can be built without any extra translation logic.
    """

    # --- data & model -------------------------------------------------
    processed_dir: str = "data/processed/emotion"
    model_name: str = "answerdotai/ModernBERT-base"
    output_dir: str = "outputs/run"
    experiment_id: str | None = None  # if set, outputs go to {output_dir}/{experiment_id}
    attn_implementation: str | None = "sdpa"  # "sdpa" | "flash_attention_2" | "eager" | None

    # --- core hyperparameters -----------------------------------------
    num_train_epochs: int = 3
    max_steps: int = -1  # if > 0, overrides num_train_epochs (HF Trainer convention)
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_torch_fused"
    seed: int = 42

    # --- checkpoint & evaluation strategy -----------------------------
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    eval_steps: int | None = None   # required when eval_strategy == "steps"
    save_steps: int | None = None   # required when save_strategy == "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    greater_is_better: bool = True
    save_total_limit: int = 2
    early_stopping_patience: int | None = None  # adds EarlyStoppingCallback when > 0
    # Offload accumulated eval tensors GPU→CPU every N steps. Needed for
    # token classification on small GPUs: Trainer otherwise keeps every batch's
    # predictions on-device until eval ends.
    eval_accumulation_steps: int | None = None

    # --- compute / throughput ----------------------------------------
    precision: str = "auto"  # "auto" | "bf16" | "fp16" | "fp32"
    gradient_checkpointing: bool = False
    torch_compile: bool = False
    train_sampling_strategy: str = "random"  # "random" | "sequential" | "group_by_length"
    dataloader_num_workers: int = 2
    pad_to_multiple_of: int | None = 8  # 8 keeps tensor cores fed under fp16/bf16

    # --- logging ------------------------------------------------------
    logging_steps: int = 50
    report_to: str = "none"

    # --- Weights & Biases ---------------------------------------------
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: list = field(default_factory=list)
    wandb_notes: str | None = None

    def __post_init__(self) -> None:
        # YAML may parse scientific notation (e.g. 2e-5) as a string.
        self.learning_rate = float(self.learning_rate)
        self.weight_decay = float(self.weight_decay)
        self.warmup_ratio = float(self.warmup_ratio)
        self.max_grad_norm = float(self.max_grad_norm)
        if self.precision not in {"auto", "bf16", "fp16", "fp32"}:
            raise ValueError(
                f"precision must be one of 'auto', 'bf16', 'fp16', 'fp32'; "
                f"got {self.precision!r}"
            )
        if self.experiment_id:
            self.output_dir = str(Path(self.output_dir) / self.experiment_id)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
    """Load a TrainConfig from a YAML file with optional dict overrides."""
    data = yaml.safe_load(Path(path).read_text())
    if overrides:
        data.update(overrides)
    return TrainConfig(**data)


def compute_total_training_steps(
    num_train_examples: int,
    cfg: TrainConfig,
    world_size: int = 1,
) -> int:
    """Total optimizer steps the Trainer will run — matches HF's internal formula.

    Used at the trainer boundary to convert ``cfg.warmup_ratio`` (a fraction) into
    ``warmup_steps`` (an absolute count), since ``warmup_ratio`` is deprecated in
    transformers and slated for removal.
    """
    if cfg.max_steps and cfg.max_steps > 0:
        return cfg.max_steps
    total_batch = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * world_size
    steps_per_epoch = max(num_train_examples // total_batch, 1)
    return steps_per_epoch * cfg.num_train_epochs
