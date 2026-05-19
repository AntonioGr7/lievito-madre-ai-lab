"""Trainer factory wired to `sentence_transformers.SentenceTransformerTrainer`.

Responsibilities:

1. Resolve `cfg.precision` → (fp16, bf16, tf32) booleans. Mirrors the same
   routine in the encoder pipelines on purpose: keep the trainers
   structurally parallel so a fix to one is easy to mirror to the others.
2. Build `SentenceTransformerTrainingArguments` from the shared
   `TrainConfig` + bi-encoder-only `BiEncoderTrainCfg`, including the
   `prompts` per-column dict for instruction-tuned backbones.
3. Instantiate the matching loss from a string, optionally wrap it in
   `MatryoshkaLoss` for multi-dimensional training.
4. Assemble the trainer with an optional evaluator + early stopping.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from datasets import DatasetDict
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.losses import (
    CachedMultipleNegativesRankingLoss,
    DistillKLDivLoss,
    Matryoshka2dLoss,
    MatryoshkaLoss,
    MultipleNegativesRankingLoss,
)
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)
from transformers import EarlyStoppingCallback

from lievito_madre_ai_lab.shared.config import TrainConfig

# Loss registry. Keys are the strings accepted in the YAML `bi_encoder.loss.name`
# field. Each value is the loss class taking `(model)` (or `(model, **kwargs)`).
LOSS_REGISTRY = {
    "MultipleNegativesRankingLoss": MultipleNegativesRankingLoss,
    "CachedMultipleNegativesRankingLoss": CachedMultipleNegativesRankingLoss,
    "DistillKLDivLoss": DistillKLDivLoss,
}

# Mapping from a plain loss name → its GradCache-enabled variant. Used by
# `bi_encoder.gradient_caching.enabled: true` to swap the loss without the
# user having to rename it in the YAML. Losses absent from this mapping
# don't have a cached variant (e.g. DistillKLDivLoss — KL is structurally
# incompatible with the GradCache pattern).
CACHED_LOSS_MAP = {
    "MultipleNegativesRankingLoss": "CachedMultipleNegativesRankingLoss",
    # Already cached — toggling caching is a no-op.
    "CachedMultipleNegativesRankingLoss": "CachedMultipleNegativesRankingLoss",
}


@dataclass
class MatryoshkaCfg:
    """Train one model that produces useful embeddings at multiple dimensions
    — and optionally at multiple encoder depths too.

    ``mode="1d"`` wraps the base loss in ``MatryoshkaLoss``: the same forward
    pass is scored at every dim in ``dims``. Truncating to dim D at inference
    (slicing the first D components) keeps quality near the natively-trained
    level.

    ``mode="2d"`` wraps in ``Matryoshka2dLoss`` which composes Matryoshka
    (over dim) with AdaptiveLayer (over encoder depth). At inference you can
    trade *both* axes: drop the top transformer layers AND truncate the
    embedding dim, with a single artifact. The extra knobs control the
    adaptive-layer half (``n_layers_per_step`` = train every k-th layer,
    weights split the loss between the last layer and earlier layers,
    ``kl_div_weight`` adds a KL term that aligns earlier layers' output
    distributions with the last one's).
    """
    enabled: bool = False
    mode: str = "1d"  # "1d" | "2d"
    dims: list[int] = field(default_factory=lambda: [768, 512, 256, 128, 64])
    weights: list[float] | None = None  # None = equal weighting across dims

    # --- 2D-only fields (AdaptiveLayer half) --------------------------
    n_layers_per_step: int = 1     # 1 = train every layer; higher = coarser
    last_layer_weight: float = 1.0
    prior_layers_weight: float = 1.0
    kl_div_weight: float = 1.0
    kl_temperature: float = 0.3

    def __post_init__(self) -> None:
        if self.mode not in {"1d", "2d"}:
            raise ValueError(f"matryoshka.mode must be '1d' or '2d'; got {self.mode!r}")


@dataclass
class GradientCachingCfg:
    """GradCache: train contrastive losses with effectively unbounded batches.

    Plain MNRL requires every (anchor, positive) pair in a batch to fit in
    GPU memory simultaneously — fine for tiny models, but on a single 24 GB
    GPU with ModernBERT-base / BGE-large you cap out around batch 32-64.
    That's *much* smaller than the 256-1024 effective batches that produce
    SOTA-quality embeddings (in-batch negatives scale with the batch).

    GradCache splits the batch into ``mini_batch_size`` chunks, runs the
    forward pass per-chunk, caches the embeddings + their gradients, then
    backpropagates through each chunk independently. Memory cost is bounded
    by ``mini_batch_size``, not by ``per_device_train_batch_size``.

    Trade-off: ~30-50% slower per step than plain MNRL (the model is run
    over each chunk twice), but enables 4-32× larger effective batches —
    a net quality win for almost any memory-bound setup.

    Only applies to contrastive losses with a cached variant
    (``CACHED_LOSS_MAP``). For DistillKLDivLoss this is a no-op + warning.
    """
    enabled: bool = False
    mini_batch_size: int = 32  # tune to the largest that fits without OOM


@dataclass
class BiEncoderTrainCfg:
    """Bi-encoder knobs that don't fit into the shared `TrainConfig`."""
    loss_name: str = "MultipleNegativesRankingLoss"
    loss_kwargs: dict = field(default_factory=dict)
    batch_sampler: str = "NO_DUPLICATES"
    matryoshka: MatryoshkaCfg = field(default_factory=MatryoshkaCfg)
    gradient_caching: GradientCachingCfg = field(default_factory=GradientCachingCfg)
    # Per-column prompt prefixes applied during training. E.g.,
    # {"anchor": "search_query: ", "positive": "search_document: "}.
    # The trainer forwards this dict to SentenceTransformerTrainingArguments.
    column_prompts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.loss_name not in LOSS_REGISTRY:
            raise ValueError(
                f"unknown loss_name {self.loss_name!r}. "
                f"Available: {sorted(LOSS_REGISTRY)}"
            )
        if self.matryoshka.enabled and not self.matryoshka.dims:
            raise ValueError(
                "matryoshka.enabled is true but matryoshka.dims is empty."
            )
        if self.matryoshka.weights is not None:
            if len(self.matryoshka.weights) != len(self.matryoshka.dims):
                raise ValueError(
                    f"matryoshka.weights length ({len(self.matryoshka.weights)}) "
                    f"must match dims length ({len(self.matryoshka.dims)})."
                )


def _resolve_precision(precision: str) -> tuple[bool, bool, bool]:
    """Resolve `precision` → (fp16, bf16, tf32) booleans for TrainingArguments."""
    cuda = torch.cuda.is_available()
    cap_major = torch.cuda.get_device_capability(0)[0] if cuda else 0
    can_do_tf32 = cuda and cap_major >= 8

    if precision == "auto":
        if can_do_tf32:
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


def _resolve_batch_sampler(name: str) -> BatchSamplers:
    """Translate the YAML string to the BatchSamplers enum."""
    try:
        return BatchSamplers[name]
    except KeyError as e:
        valid = [m.name for m in BatchSamplers]
        raise ValueError(
            f"unknown batch_sampler {name!r}. Available: {valid}"
        ) from e


def build_training_args(
    cfg: TrainConfig,
    *,
    num_training_steps: int | None = None,
    bi_encoder_cfg: BiEncoderTrainCfg | None = None,
) -> SentenceTransformerTrainingArguments:
    warmup_steps = 0
    if cfg.warmup_ratio > 0:
        if num_training_steps is None:
            raise ValueError(
                "build_training_args requires num_training_steps when cfg.warmup_ratio > 0."
            )
        warmup_steps = math.ceil(num_training_steps * cfg.warmup_ratio)

    fp16, bf16, tf32 = _resolve_precision(cfg.precision)
    bec = bi_encoder_cfg or BiEncoderTrainCfg()

    return SentenceTransformerTrainingArguments(
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
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        save_total_limit=cfg.save_total_limit,
        fp16=fp16,
        bf16=bf16,
        tf32=tf32,
        gradient_checkpointing=cfg.gradient_checkpointing,
        torch_compile=cfg.torch_compile,
        dataloader_num_workers=cfg.dataloader_num_workers,
        logging_steps=cfg.logging_steps,
        report_to=cfg.report_to,
        # Sentence-Transformers-specific: pick a batch sampler that's safe
        # under in-batch-negatives losses (NO_DUPLICATES) so two rows with
        # the same anchor never end up in the same batch.
        batch_sampler=_resolve_batch_sampler(bec.batch_sampler),
        # Per-column instruction prefixes (E5 / Nomic / BGE-M3). Empty dict
        # is a valid no-op for non-instruction models.
        prompts=bec.column_prompts or None,
    )


def build_loss(model: SentenceTransformer, bec: BiEncoderTrainCfg):
    """Instantiate the loss object selected in the YAML, optionally wrapped
    in Matryoshka for multi-dimensional training. When
    ``gradient_caching.enabled`` is on, transparently swaps the loss for
    its GradCache-enabled variant and injects ``mini_batch_size``.
    """
    loss_name = bec.loss_name
    loss_kwargs = dict(bec.loss_kwargs)

    if bec.gradient_caching.enabled:
        cached_name = CACHED_LOSS_MAP.get(loss_name)
        if cached_name is None:
            print(
                f"      [warn] gradient_caching.enabled=true but "
                f"{loss_name!r} has no cached variant — caching ignored."
            )
        else:
            if cached_name != loss_name:
                print(
                    f"      [gradcache] {loss_name} → {cached_name} "
                    f"(mini_batch_size={bec.gradient_caching.mini_batch_size})"
                )
            loss_name = cached_name
            loss_kwargs["mini_batch_size"] = bec.gradient_caching.mini_batch_size

    loss_cls = LOSS_REGISTRY[loss_name]
    base = loss_cls(model, **loss_kwargs)

    if bec.matryoshka.enabled:
        if bec.matryoshka.mode == "2d":
            # Adaptive depth × Matryoshka dim. Earlier layers learn to emit
            # useful embeddings too, so the user can drop the top N layers
            # at inference and still keep meaningful (truncated) vectors.
            return Matryoshka2dLoss(
                model=model,
                loss=base,
                matryoshka_dims=bec.matryoshka.dims,
                matryoshka_weights=bec.matryoshka.weights,
                n_layers_per_step=bec.matryoshka.n_layers_per_step,
                last_layer_weight=bec.matryoshka.last_layer_weight,
                prior_layers_weight=bec.matryoshka.prior_layers_weight,
                kl_div_weight=bec.matryoshka.kl_div_weight,
                kl_temperature=bec.matryoshka.kl_temperature,
            )
        # 1D — dim only.
        return MatryoshkaLoss(
            model=model,
            loss=base,
            matryoshka_dims=bec.matryoshka.dims,
            matryoshka_weights=bec.matryoshka.weights,
        )
    return base


def build_trainer(
    model: SentenceTransformer,
    datasets: DatasetDict,
    training_args: SentenceTransformerTrainingArguments,
    loss,
    *,
    evaluator: SentenceEvaluator | None = None,
    early_stopping_patience: int | None = None,
) -> SentenceTransformerTrainer:
    eval_split = "validation" if "validation" in datasets else "test"
    eval_dataset = datasets.get(eval_split)

    callbacks = []
    if early_stopping_patience and early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        )

    return SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
        callbacks=callbacks or None,
    )
