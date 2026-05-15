#!/usr/bin/env python
"""Fine-tune an encoder model for token classification (NER / PII) with HF Trainer.

Usage
-----
python scripts/token_classification/train_token_classification.py \\
    --config configs/encoder/token_classification/pii_mbert.yaml

# Resume from the latest checkpoint after a crash
python scripts/token_classification/train_token_classification.py \\
    --config configs/encoder/token_classification/pii_mbert.yaml --resume

# Resume from a specific checkpoint
python scripts/token_classification/train_token_classification.py \\
    --config configs/encoder/token_classification/pii_mbert.yaml \\
    --resume outputs/pii_mbert/checkpoint-5000

# Smoke test on a small slice (laptop-friendly)
python scripts/token_classification/train_token_classification.py \\
    --config configs/encoder/token_classification/pii_mbert.yaml \\
    --max-train-samples 200 --max-eval-samples 100 --max-test-samples 100
"""

import argparse
import dataclasses
import json
from pathlib import Path

from datasets import ClassLabel, Sequence, load_from_disk
from dotenv import load_dotenv

load_dotenv()

from lievito_madre_ai_lab.encoder.token_classification.evaluate import build_compute_metrics
from lievito_madre_ai_lab.encoder.token_classification.model import load_model_and_tokenizer
from lievito_madre_ai_lab.encoder.token_classification.trainer import build_trainer, build_training_args
from lievito_madre_ai_lab.shared.config import TrainConfig, compute_total_training_steps, load_config


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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Path to a TrainConfig YAML file")
    p.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        metavar="CHECKPOINT_DIR",
        help="Resume training. Omit a path to auto-detect the latest checkpoint in output_dir.",
    )
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="If set, truncate the train split to this many examples (smoke testing).")
    p.add_argument("--max-eval-samples", type=int, default=None,
                   help="If set, truncate the validation split to this many examples.")
    p.add_argument("--max-test-samples", type=int, default=None,
                   help="If set, truncate the test split to this many examples.")
    return p.parse_args()


def infer_labels(datasets) -> tuple[int, list[str] | None]:
    """Return (num_labels, label_names) from the processed dataset's labels feature."""
    feature = datasets["train"].features.get("labels")
    if isinstance(feature, Sequence) and isinstance(feature.feature, ClassLabel):
        return feature.feature.num_classes, feature.feature.names
    # Fallback: scan unique label ids across all splits
    all_ids: set[int] = set()
    for split in datasets.values():
        for row in split["labels"]:
            all_ids.update(v for v in row if v != -100)
    return len(all_ids), None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    setup_wandb(cfg)

    # ------------------------------------------------------------------
    # 1. Load the tokenized dataset produced by a prepare script (see examples/)
    # ------------------------------------------------------------------
    print(f"[1/4] Loading processed dataset from '{cfg.processed_dir}' …")
    datasets = load_from_disk(cfg.processed_dir)

    limits = {"train": args.max_train_samples,
              "validation": args.max_eval_samples,
              "test": args.max_test_samples}
    for split, n in limits.items():
        if n is not None and split in datasets:
            datasets[split] = datasets[split].select(range(min(n, len(datasets[split]))))
            print(f"      [smoke] {split}: truncated to {len(datasets[split])} examples")

    print(datasets)

    num_labels, label_names = infer_labels(datasets)
    print(f"      {num_labels} labels: {label_names or list(range(num_labels))}")

    # ------------------------------------------------------------------
    # 2. Load model + tokenizer
    # ------------------------------------------------------------------
    print(f"[2/4] Loading '{cfg.model_name}' …")
    model, tokenizer = load_model_and_tokenizer(
        cfg.model_name,
        num_labels,
        label_names,
        attn_implementation=cfg.attn_implementation,
    )

    # ------------------------------------------------------------------
    # 3. Build trainer
    # ------------------------------------------------------------------
    print("[3/4] Building trainer …")
    total_steps = compute_total_training_steps(len(datasets["train"]), cfg)
    training_args = build_training_args(cfg, num_training_steps=total_steps)
    compute_metrics = build_compute_metrics(label_names or [str(i) for i in range(num_labels)])
    trainer = build_trainer(
        model,
        datasets,
        tokenizer,
        training_args,
        compute_metrics,
        pad_to_multiple_of=cfg.pad_to_multiple_of,
        early_stopping_patience=cfg.early_stopping_patience,
    )

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    print("[4/4] Training …")
    trainer.train(resume_from_checkpoint=args.resume)

    # ------------------------------------------------------------------
    # 5. Final evaluation on the held-out test split
    # ------------------------------------------------------------------
    final_dir = Path(cfg.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    if "test" in datasets:
        print("Evaluating on test split …")
        metrics = trainer.evaluate(datasets["test"], metric_key_prefix="test")
        print(json.dumps(metrics, indent=2))
        (final_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Model saved → {final_dir}")


if __name__ == "__main__":
    main()
