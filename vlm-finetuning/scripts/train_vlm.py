#!/usr/bin/env python
"""Fine-tune a vision-language model on a grounding dataset (LoRA / QLoRA / full).

Usage
-----
python scripts/train_vlm.py \\
    --config examples/grounding/configs/cppe5_qwen25vl_3b.yaml

# Resume from the latest checkpoint after a crash
python scripts/train_vlm.py --config ... --resume

# Smoke test on a small slice
python scripts/train_vlm.py --config ... \\
    --max-train-samples 8 --max-eval-samples 4 --max-test-samples 4
"""
import argparse
import dataclasses
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from vlm_finetuning.dataset import load_processed
from vlm_finetuning.evaluate import evaluate_split
from vlm_finetuning.model import LoraSpec, QuantSpec, load_vlm
from vlm_finetuning.trainer import (
    GroundingCfg,
    build_trainer,
    build_training_args,
)
from vlm_finetuning.shared.config import (
    TrainConfig,
    compute_total_training_steps,
)
from vlm_finetuning.shared.preprocessing import (
    load_preprocessing_meta,
    save_preprocessing_meta,
)


def _load_config_with_vlm_block(path: str) -> tuple[TrainConfig, dict]:
    """Split the VLM-specific block out so TrainConfig doesn't reject it."""
    data = yaml.safe_load(Path(path).read_text())
    vlm_cfg = data.pop("vlm", {}) or {}
    return TrainConfig(**data), vlm_cfg


def _build_vlm_objects(vlm_cfg: dict) -> tuple[GroundingCfg, LoraSpec, QuantSpec, dict]:
    """Pull the structured sub-dicts out of the raw YAML ``vlm:`` block."""
    lora_d = vlm_cfg.get("lora", {}) or {}
    lora = LoraSpec(
        enabled=bool(lora_d.get("enabled", True)),
        r=int(lora_d.get("r", 16)),
        alpha=int(lora_d.get("alpha", 32)),
        dropout=float(lora_d.get("dropout", 0.05)),
        target_modules=lora_d.get("target_modules", "auto"),
        modules_to_save=list(lora_d.get("modules_to_save", []) or []),
    )

    quant_d = vlm_cfg.get("quant", {}) or {}
    quant = QuantSpec(
        load_in_4bit=bool(quant_d.get("load_in_4bit", False)),
        bnb_4bit_quant_type=str(quant_d.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(quant_d.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=str(quant_d.get("bnb_4bit_compute_dtype", "bfloat16")),
    )

    eval_d = vlm_cfg.get("eval", {}) or {}
    task = str(vlm_cfg.get("task", "box"))
    if task not in {"box", "point", "text"}:
        raise ValueError(f"vlm.task must be 'box', 'point' or 'text'; got {task!r}")
    grounding = GroundingCfg(
        task=task,
        coord_bins=int(vlm_cfg.get("coord_bins", 1000)),
        system_prompt=vlm_cfg.get("system_prompt"),
        empty_text=str(vlm_cfg.get("empty_text", "No objects detected.")),
        max_length=vlm_cfg.get("max_length"),
        eval_max_new_tokens=int(eval_d.get("max_new_tokens", 512)),
        eval_iou_threshold=float(eval_d.get("iou_threshold", 0.5)),
        eval_max_samples=eval_d.get("max_samples"),
    )
    return grounding, lora, quant, vlm_cfg


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
    cfg, vlm_raw = _load_config_with_vlm_block(args.config)
    setup_wandb(cfg)

    print(f"[1/5] Loading processed dataset from {cfg.processed_dir} …")
    datasets, labels = load_processed(cfg.processed_dir)
    _truncate(datasets, args)
    print(datasets)
    print(f"      labels ({len(labels)}): {labels}")

    grounding_cfg, lora, quant, vlm_raw = _build_vlm_objects(vlm_raw)

    print(
        f"[2/5] Loading {cfg.model_name} "
        f"(lora={lora.enabled}, 4bit={quant.load_in_4bit}, "
        f"freeze_vision={vlm_raw.get('freeze_vision_tower', True)}) …"
    )
    model, processor = load_vlm(
        cfg.model_name,
        precision=cfg.precision,
        attn_implementation=cfg.attn_implementation,
        lora_cfg=lora,
        quant_cfg=quant,
        freeze_vision_tower=bool(vlm_raw.get("freeze_vision_tower", True)),
        gradient_checkpointing=cfg.gradient_checkpointing,
        add_coord_special_tokens=bool(vlm_raw.get("add_coord_special_tokens", False)),
        image_min_pixels=vlm_raw.get("image_min_pixels"),
        image_max_pixels=vlm_raw.get("image_max_pixels"),
        trust_remote_code=bool(vlm_raw.get("trust_remote_code", False)),
    )

    print("[3/5] Building trainer …")
    total_steps = compute_total_training_steps(len(datasets["train"]), cfg)
    training_args = build_training_args(cfg, num_training_steps=total_steps)
    trainer = build_trainer(
        model, processor, datasets, training_args,
        grounding_cfg=grounding_cfg,
        labels=labels,
        early_stopping_patience=cfg.early_stopping_patience,
    )

    print("[4/5] Training …")
    trainer.train(resume_from_checkpoint=args.resume)

    final_dir = Path(cfg.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Persist the model + processor + preprocessing contract BEFORE the test
    # eval. If we evaluated first, a crash mid-generation would leave `final/`
    # empty and the only recoverable artefact would be the last checkpoint-N/.
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    default_prompt = vlm_raw.get("default_prompt") or (
        datasets["train"][0]["prompt"] if len(datasets["train"]) else None
    )
    prep_meta = load_preprocessing_meta(cfg.processed_dir) or {}
    meta_out = {
        **prep_meta,                       # carry over anything prep recorded
        "processor": cfg.model_name,
        "task": grounding_cfg.task,
        "coord_bins": grounding_cfg.coord_bins,
        "system_prompt": grounding_cfg.system_prompt,
        "default_prompt": default_prompt,
        "max_new_tokens": grounding_cfg.eval_max_new_tokens,
        "iou_threshold": grounding_cfg.eval_iou_threshold,
        "labels": labels,
    }
    save_preprocessing_meta(final_dir, **meta_out)
    print(f"      preprocessing.json written to {final_dir}")
    print(f"Model saved -> {final_dir}")

    print("[5/5] Final evaluation on the test split (model already saved) …")
    metrics: dict = {"task": grounding_cfg.task, "iou_threshold": grounding_cfg.eval_iou_threshold}
    if "test" in datasets and len(datasets["test"]):
        model.eval()
        test_metrics = evaluate_split(
            model, processor, datasets["test"],
            task=grounding_cfg.task,
            coord_bins=grounding_cfg.coord_bins,
            iou_threshold=grounding_cfg.eval_iou_threshold,
            labels=labels,
            system_prompt=grounding_cfg.system_prompt,
            max_new_tokens=grounding_cfg.eval_max_new_tokens,
            batch_size=cfg.per_device_eval_batch_size,
            progress=True,
        )
        for k, v in test_metrics.items():
            metrics[f"test_{k}"] = v
        print(json.dumps(
            {k: v for k, v in metrics.items()
             if k.startswith("test_") and not k.startswith("test_tp")},
            indent=2,
        ))

    (final_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
