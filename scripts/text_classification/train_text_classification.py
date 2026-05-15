#!/usr/bin/env python
"""Fine-tune an encoder model for sequence classification with HF Trainer.

Usage
-----
python scripts/train_text_classification.py \
    --config configs/encoder/text_classification/emotion_bert.yaml
"""

import argparse
import dataclasses
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from datasets import ClassLabel, load_from_disk

from lievito_madre_ai_lab.encoder.text_classification.dataset import (
    load_preprocessing_meta,
    save_preprocessing_meta,
)
from lievito_madre_ai_lab.encoder.text_classification.evaluate import build_compute_metrics
from lievito_madre_ai_lab.encoder.text_classification.model import load_model_and_tokenizer
from lievito_madre_ai_lab.encoder.text_classification.trainer import build_trainer, build_training_args
from lievito_madre_ai_lab.shared.config import TrainConfig, compute_total_training_steps, load_config
from lievito_madre_ai_lab.shared.preprocessing import (
    assert_tokenizer_matches,
    warn_if_max_length_exceeds_model_capacity,
)


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
    return p.parse_args()


def infer_labels(datasets) -> tuple[int, list[str] | None]:
    """Return (num_labels, label_names) from the processed dataset."""
    feature = datasets["train"].features.get("labels")
    if isinstance(feature, ClassLabel):
        return feature.num_classes, feature.names
    # plain int — infer from unique values across all splits
    all_labels: set[int] = set()
    for split in datasets.values():
        all_labels.update(split["labels"])
    return len(all_labels), None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    setup_wandb(cfg)

    # ------------------------------------------------------------------
    # 1. Load the tokenized dataset produced by a prepare script (see examples/)
    # ------------------------------------------------------------------
    print(f"[1/4] Loading processed dataset from '{cfg.processed_dir}' …")
    datasets = load_from_disk(cfg.processed_dir)

    # Hard-fail early if the tokenizer used at prep time doesn't match the
    # model picked here — a silent vocabulary mismatch wastes the whole run.
    prep_meta = load_preprocessing_meta(cfg.processed_dir)
    if prep_meta is not None:
        assert_tokenizer_matches(prep_meta, model_name=cfg.model_name)

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
    if prep_meta is not None:
        warn_if_max_length_exceeds_model_capacity(prep_meta, model_config=model.config)

    # ------------------------------------------------------------------
    # 3. Build trainer
    # ------------------------------------------------------------------
    print("[3/4] Building trainer …")
    total_steps = compute_total_training_steps(len(datasets["train"]), cfg)
    training_args = build_training_args(cfg, num_training_steps=total_steps)
    compute_metrics = build_compute_metrics(label_names)
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

    # Forward preprocessing settings (tokenizer, max_length, …) recorded by the
    # prepare script into the model dir, so serve.py can pick the same values
    # without the caller having to remember them.
    if prep_meta is not None:
        save_preprocessing_meta(final_dir, **prep_meta)
        print(f"      preprocessing.json copied to {final_dir} "
              f"(max_length={prep_meta.get('max_length')})")
    else:
        print(f"      [warn] no preprocessing.json in '{cfg.processed_dir}'; "
              f"serve.py will fall back to its hardcoded defaults.")

    print(f"Model saved → {final_dir}")


if __name__ == "__main__":
    main()
