#!/usr/bin/env python
"""Fine-tune a DETR-family object detector on a COCO-format dataset.

Usage
-----
python scripts/train_detector.py --config examples/cppe5/configs/dfine_x.yaml

# Resume after a crash
python scripts/train_detector.py --config ... --resume

# Smoke a tiny slice
python scripts/train_detector.py --config ... \\
    --max-train-samples 8 --max-eval-samples 4 --max-test-samples 4
"""
import argparse
import dataclasses
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from object_detection.dataset import load_processed
from object_detection.evaluate import compute_map
from object_detection.model import load_detector
from object_detection.trainer import (
    DetectionCfg,
    build_trainer,
    build_training_args,
)
from object_detection.shared.config import (
    TrainConfig,
    compute_total_training_steps,
)
from object_detection.shared.preprocessing import (
    load_preprocessing_meta,
    save_preprocessing_meta,
)


def _load_config_with_detection_block(path: str) -> tuple[TrainConfig, dict]:
    data = yaml.safe_load(Path(path).read_text())
    det_cfg = data.pop("detection", {}) or {}
    return TrainConfig(**data), det_cfg


def _build_detection_cfg(det: dict) -> DetectionCfg:
    aug = det.get("augmentation", {}) or {}
    ema = det.get("ema", {}) or {}
    ev = det.get("eval", {}) or {}
    return DetectionCfg(
        image_size=det.get("image_size"),
        freeze_backbone=bool(det.get("freeze_backbone", False)),
        backbone_lr_mult=float(det.get("backbone_lr_mult", 0.1)),
        aug_enabled=bool(aug.get("enabled", True)),
        hflip=float(aug.get("hflip", 0.5)),
        brightness_contrast=float(aug.get("brightness_contrast", 0.5)),
        hue_sat=float(aug.get("hue_sat", 0.3)),
        ema_enabled=bool(ema.get("enabled", True)),
        ema_decay=float(ema.get("decay", 0.9997)),
        eval_threshold=float(ev.get("threshold", 0.0)),
        eval_max_samples=ev.get("max_samples"),
        score_threshold=float(det.get("score_threshold", 0.3)),
        trust_remote_code=bool(det.get("trust_remote_code", False)),
    )


def setup_wandb(cfg: TrainConfig) -> None:
    if cfg.wandb_project is None:
        return
    import wandb
    wandb.init(
        project=cfg.wandb_project, name=cfg.wandb_run_name,
        tags=cfg.wandb_tags or None, notes=cfg.wandb_notes,
        config=dataclasses.asdict(cfg),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--resume", nargs="?", const=True, default=None, metavar="CHECKPOINT_DIR")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--max-test-samples", type=int, default=None)
    return p.parse_args()


def _truncate(datasets, args) -> None:
    for split, n in (("train", args.max_train_samples),
                     ("validation", args.max_eval_samples),
                     ("test", args.max_test_samples)):
        if n is not None and split in datasets:
            datasets[split] = datasets[split].select(range(min(n, len(datasets[split]))))
            print(f"      [smoke] {split}: truncated to {len(datasets[split])}")


def main() -> None:
    args = parse_args()
    cfg, det_raw = _load_config_with_detection_block(args.config)
    setup_wandb(cfg)
    det_cfg = _build_detection_cfg(det_raw)

    print(f"[1/5] Loading processed dataset from {cfg.processed_dir} …")
    datasets, id2label = load_processed(cfg.processed_dir)
    _truncate(datasets, args)
    print(datasets)
    print(f"      {len(id2label)} classes: {list(id2label.values())}")

    print(f"[2/5] Loading {cfg.model_name} (freeze_backbone={det_cfg.freeze_backbone}, "
          f"image_size={det_cfg.image_size}) …")
    model, image_processor = load_detector(
        cfg.model_name,
        id2label=id2label,
        image_size=det_cfg.image_size,
        freeze_backbone=det_cfg.freeze_backbone,
        attn_implementation=cfg.attn_implementation,
        trust_remote_code=det_cfg.trust_remote_code,
    )

    print("[3/5] Building trainer …")
    total_steps = compute_total_training_steps(len(datasets["train"]), cfg)
    training_args = build_training_args(cfg, num_training_steps=total_steps)
    trainer, ema = build_trainer(
        model, image_processor, datasets, training_args,
        det_cfg=det_cfg, cfg=cfg, id2label=id2label,
        early_stopping_patience=cfg.early_stopping_patience,
    )

    print("[4/5] Training …")
    trainer.train(resume_from_checkpoint=args.resume)

    final_dir = Path(cfg.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Ship the EMA weights (if enabled): swap them into the model before saving,
    # and keep them in place for the final test eval so the saved numbers match
    # the saved weights. Save BEFORE eval so a crash still leaves a loadable model.
    if ema is not None:
        ema.copy_to(trainer.model)
    trainer.save_model(str(final_dir))
    image_processor.save_pretrained(str(final_dir))

    prep_meta = load_preprocessing_meta(cfg.processed_dir) or {}
    meta_out = {
        **prep_meta,
        "processor": cfg.model_name,
        "task": "object-detection",
        "image_size": det_cfg.image_size,
        "threshold": det_cfg.score_threshold,
        "ema": det_cfg.ema_enabled,
        "labels": [id2label[i] for i in range(len(id2label))],
    }
    save_preprocessing_meta(final_dir, **meta_out)
    print(f"      preprocessing.json written to {final_dir}")
    print(f"Model saved -> {final_dir}")

    print("[5/5] Final mAP evaluation on the test split (model already saved) …")
    metrics: dict = {}
    if "test" in datasets and len(datasets["test"]):
        test_metrics = compute_map(
            trainer.model, image_processor, datasets["test"],
            id2label=id2label,
            batch_size=cfg.per_device_eval_batch_size,
            threshold=det_cfg.eval_threshold,
            progress=True,
        )
        metrics = {f"test_{k}": v for k, v in test_metrics.items()}
        print(json.dumps(metrics, indent=2))

    (final_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
