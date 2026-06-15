from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    """Hyperparameters and paths for a fine-tuning run.

    Load from YAML:
        cfg = load_config("examples/grounding/configs/cppe5_qwen25vl_3b.yaml")

    All fields map 1-to-1 to TrainingArguments / from_pretrained so the trainer
    can be built without any extra translation logic. Task-specific knobs
    (LoRA, quantization, the coordinate scheme, generation-eval settings) live
    in the YAML's ``vlm:`` block and are parsed separately by the train script
    — exactly so this dataclass stays a thin shadow of HF's TrainingArguments.
    """

    # --- data & model -------------------------------------------------
    processed_dir: str = "data/processed/cppe5"
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    output_dir: str = "outputs/run"
    experiment_id: str | None = None  # if set, outputs go to {output_dir}/{experiment_id}
    attn_implementation: str | None = "sdpa"  # "sdpa" | "flash_attention_2" | "eager" | None

    # --- core hyperparameters -----------------------------------------
    num_train_epochs: int = 3
    max_steps: int = -1  # if > 0, overrides num_train_epochs (HF Trainer convention)
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
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
    eval_accumulation_steps: int | None = None

    # --- compute / throughput ----------------------------------------
    precision: str = "auto"  # "auto" | "bf16" | "fp16" | "fp32"
    gradient_checkpointing: bool = True
    torch_compile: bool = False
    dataloader_num_workers: int = 4

    # --- logging ------------------------------------------------------
    logging_steps: int = 10
    report_to: str = "none"

    # --- Weights & Biases ---------------------------------------------
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: list = field(default_factory=list)
    wandb_notes: str | None = None

    def __post_init__(self) -> None:
        # YAML may parse scientific notation (e.g. 1e-4) as a string.
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
