#!/usr/bin/env python
"""Fine-tune a GLiNER model on a char-offset dataset.

Usage
-----
python scripts/gliner_entity_extraction/train_gliner.py \\
    --config configs/encoder/gliner_entity_extraction/pii_gliner.yaml

# Resume from the latest checkpoint after a crash
python scripts/gliner_entity_extraction/train_gliner.py --config ... --resume

# Smoke test on a small slice
python scripts/gliner_entity_extraction/train_gliner.py --config ... \\
    --max-train-samples 200 --max-eval-samples 100 --max-test-samples 100
"""
import argparse
import dataclasses
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from lievito_madre_ai_lab.encoder.gliner_entity_extraction.dataset import (
    load_processed,
    to_native_dataset,
)
from lievito_madre_ai_lab.encoder.gliner_entity_extraction.evaluate import evaluate_split
from lievito_madre_ai_lab.encoder.gliner_entity_extraction.model import (
    PeftConfig,
    load_gliner,
)
from lievito_madre_ai_lab.encoder.gliner_entity_extraction.trainer import (
    GLiNERTrainCfg,
    build_trainer,
    build_training_args,
)
from lievito_madre_ai_lab.shared.config import (
    TrainConfig,
    compute_total_training_steps,
)
from lievito_madre_ai_lab.shared.preprocessing import (
    load_preprocessing_meta,
    save_preprocessing_meta,
)


def _load_config_with_gliner_block(path: str) -> tuple[TrainConfig, dict]:
    """Split the GLiNER-specific block out so TrainConfig doesn't reject it."""
    data = yaml.safe_load(Path(path).read_text())
    gliner_cfg = data.pop("gliner", {}) or {}
    return TrainConfig(**data), gliner_cfg


def _build_gliner_objects(gliner_cfg: dict) -> tuple[GLiNERTrainCfg, PeftConfig, dict, dict, dict]:
    """Pull the structured sub-dicts out of the raw YAML `gliner:` block."""
    loss = gliner_cfg.get("loss", {}) or {}
    sampling = gliner_cfg.get("sampling", {}) or {}
    aliases = gliner_cfg.get("label_aliases", {}) or {}

    peft_dict = gliner_cfg.get("peft", {}) or {}
    peft_cfg = PeftConfig(
        enabled=bool(peft_dict.get("enabled", False)),
        r=int(peft_dict.get("r", 16)),
        alpha=int(peft_dict.get("alpha", 32)),
        dropout=float(peft_dict.get("dropout", 0.1)),
        target_modules=peft_dict.get("target_modules", "auto"),
    )

    # Focal loss is opt-in via loss.funcs containing "focal". When enabled,
    # forward the alpha/gamma onto GLiNERTrainCfg → TrainingArguments.
    focal_on = "focal" in (loss.get("funcs") or [])

    # Sliding-window chunking of long training docs (word-token level). The
    # YAML can pin chunk_max_words; if absent we let the trainer fall back to
    # model.config.max_len so we never exceed the model's truncation cap.
    chunk_cfg = gliner_cfg.get("chunking", {}) or {}
    chunk_max_words = chunk_cfg.get("max_words")
    chunk_stride = int(chunk_cfg.get("stride", 64))

    train_cfg = GLiNERTrainCfg(
        max_span_width=gliner_cfg.get("max_span_width"),
        head_lr_multiplier=float(gliner_cfg.get("head_lr_multiplier", 5.0)),
        focal_loss_alpha=float(loss.get("focal_alpha", 0.75)) if focal_on else -1.0,
        focal_loss_gamma=float(loss.get("focal_gamma", 2.0)) if focal_on else 0.0,
        chunk_max_words=int(chunk_max_words) if chunk_max_words is not None else None,
        chunk_stride=chunk_stride,
    )
    return train_cfg, peft_cfg, loss, sampling, aliases


def setup_wandb(cfg: TrainConfig) -> None:
    if cfg.wandb_project is None:
        return
    import wandb
    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        tags=cfg.wandb_tags or None,
        notes=cfg.wandb_notes,
        config=dataclasses.asdict(cfg),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True)
    p.add_argument("--resume", nargs="?", const=True, default=None, metavar="CHECKPOINT_DIR")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--max-test-samples", type=int, default=None)
    return p.parse_args()


def _truncate(datasets, args: argparse.Namespace) -> None:
    limits = {
        "train": args.max_train_samples,
        "validation": args.max_eval_samples,
        "test": args.max_test_samples,
    }
    for split, n in limits.items():
        if n is not None and split in datasets:
            datasets[split] = datasets[split].select(range(min(n, len(datasets[split]))))
            print(f"      [smoke] {split}: truncated to {len(datasets[split])}")


def main() -> None:
    args = parse_args()
    cfg, gliner_raw = _load_config_with_gliner_block(args.config)
    setup_wandb(cfg)

    print(f"[1/5] Loading processed dataset from {cfg.processed_dir} …")
    datasets, train_types, holdout_types = load_processed(cfg.processed_dir)
    _truncate(datasets, args)
    print(datasets)
    print(f"      train_types   ({len(train_types)}):   {train_types}")
    print(f"      holdout_types ({len(holdout_types)}): {holdout_types}")

    train_cfg, peft_cfg, loss_cfg, sampling_cfg, label_aliases = _build_gliner_objects(gliner_raw)

    print(f"[2/5] Loading {cfg.model_name} (peft={peft_cfg.enabled}, "
          f"max_span_width={train_cfg.max_span_width}) …")
    model = load_gliner(
        cfg.model_name,
        train_types=train_types,
        holdout_types=holdout_types,
        max_span_width=train_cfg.max_span_width,
        loss_cfg=loss_cfg,
        sampling_cfg=sampling_cfg,
        label_aliases=label_aliases,
        peft_cfg=peft_cfg,
    )

    print("[3/5] Building trainer …")
    total_steps = compute_total_training_steps(len(datasets["train"]), cfg)
    training_args = build_training_args(cfg, num_training_steps=total_steps, gliner_cfg=train_cfg)
    trainer = build_trainer(
        model,
        datasets,
        training_args,
        gliner_cfg=train_cfg,
        train_types=train_types,
        early_stopping_patience=cfg.early_stopping_patience,
    )

    print("[4/5] Training …")
    trainer.train(resume_from_checkpoint=args.resume)

    final_dir = Path(cfg.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    print("[5/5] Final evaluation on the test split …")
    metrics: dict = {}
    if "test" in datasets:
        # Mirror the same chunking the trainer used so final test f1 is
        # computed on the same units that ``eval_f1`` tracked during training.
        effective_max_words_for_eval = (
            train_cfg.chunk_max_words
            if train_cfg.chunk_max_words is not None
            else int(getattr(model.config, "max_len", 384))
        )
        test_for_eval = to_native_dataset(
            datasets["test"],
            model.data_processor.words_splitter,
            max_words=effective_max_words_for_eval if train_cfg.chunk_stride >= 0 else None,
            stride=train_cfg.chunk_stride,
            desc="Chunking test for final evaluation",
        )

        closed = evaluate_split(
            model, test_for_eval, labels=train_types,
            label_aliases=label_aliases,
            batch_size=cfg.per_device_eval_batch_size,
        )
        for k, v in closed.items():
            metrics[f"test_closed_{k}"] = v
        print(json.dumps(
            {k: v for k, v in metrics.items() if k.startswith("test_closed_")},
            indent=2,
        ))

        if holdout_types:
            zero = evaluate_split(
                model, test_for_eval, labels=holdout_types,
                label_aliases=label_aliases,
                batch_size=cfg.per_device_eval_batch_size,
            )
            for k, v in zero.items():
                metrics[f"test_zeroshot_{k}"] = v
            print(json.dumps(
                {k: v for k, v in metrics.items() if k.startswith("test_zeroshot_")},
                indent=2,
            ))

    (final_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    model.save_pretrained(final_dir, safe_serialization=True)

    # Persist the preprocessing/inference contract next to the saved model so
    # the predictor (and humans) can discover the exact chunking settings.
    # GLiNER's prep is tokenizer-agnostic — so unlike text/token classification
    # the model_name and chunking config are decided here, at train time.
    effective_max_words = (
        train_cfg.chunk_max_words
        if train_cfg.chunk_max_words is not None
        else int(getattr(model.config, "max_len", 384))
    )
    prep_meta = load_preprocessing_meta(cfg.processed_dir) or {}
    save_preprocessing_meta(
        final_dir,
        **prep_meta,            # carry over anything the prep script recorded
        tokenizer=cfg.model_name,
        max_words=effective_max_words,
        stride=train_cfg.chunk_stride if train_cfg.chunk_stride >= 0 else None,
        train_types=train_types,
        holdout_types=holdout_types,
    )
    print(
        f"      preprocessing.json written to {final_dir} "
        f"(max_words={effective_max_words}, "
        f"stride={train_cfg.chunk_stride if train_cfg.chunk_stride >= 0 else 'off'})"
    )
    print(f"Model saved -> {final_dir}")


if __name__ == "__main__":
    main()
