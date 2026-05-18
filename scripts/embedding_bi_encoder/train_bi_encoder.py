#!/usr/bin/env python
"""Fine-tune a SentenceTransformer bi-encoder with the HF-style trainer.

Usage
-----
python scripts/embedding_bi_encoder/train_bi_encoder.py \
    --config configs/embedding/bi_encoder/<your>.yaml
"""

import argparse
import dataclasses
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from datasets import load_from_disk

from lievito_madre_ai_lab.embedding.bi_encoder.dataset import (
    load_preprocessing_meta,
    save_preprocessing_meta,
    validate_dataset,
)
from lievito_madre_ai_lab.embedding.bi_encoder.evaluate import build_evaluator
from lievito_madre_ai_lab.embedding.bi_encoder.model import load_sentence_transformer
from lievito_madre_ai_lab.embedding.bi_encoder.trainer import (
    BiEncoderTrainCfg,
    MatryoshkaCfg,
    build_loss,
    build_trainer,
    build_training_args,
)
from lievito_madre_ai_lab.shared.config import (
    TrainConfig,
    compute_total_training_steps,
)
from lievito_madre_ai_lab.shared.preprocessing import assert_tokenizer_matches


def _load_config_with_bi_encoder_block(path: str) -> tuple[TrainConfig, BiEncoderTrainCfg]:
    """Pop the `bi_encoder:` block off the YAML so TrainConfig doesn't reject it."""
    data = yaml.safe_load(Path(path).read_text())
    raw = data.pop("bi_encoder", {}) or {}

    loss = raw.get("loss", {}) or {}
    mat = raw.get("matryoshka", {}) or {}
    prompts = raw.get("prompts", {}) or {}

    matryoshka_cfg = MatryoshkaCfg(
        enabled=bool(mat.get("enabled", False)),
        mode=str(mat.get("mode", "1d")),
        dims=list(mat.get("dims", [768, 512, 256, 128, 64])),
        weights=list(mat["weights"]) if mat.get("weights") is not None else None,
        n_layers_per_step=int(mat.get("n_layers_per_step", 1)),
        last_layer_weight=float(mat.get("last_layer_weight", 1.0)),
        prior_layers_weight=float(mat.get("prior_layers_weight", 1.0)),
        kl_div_weight=float(mat.get("kl_div_weight", 1.0)),
        kl_temperature=float(mat.get("kl_temperature", 0.3)),
    )

    bec = BiEncoderTrainCfg(
        loss_name=loss.get("name", "MultipleNegativesRankingLoss"),
        loss_kwargs=loss.get("kwargs", {}) or {},
        batch_sampler=raw.get("batch_sampler", "NO_DUPLICATES"),
        matryoshka=matryoshka_cfg,
        column_prompts=prompts.get("columns", {}) or {},
    )
    # `inference_prompts` is carried through to model.load — stored on the
    # SentenceTransformer object and persisted by save_pretrained.
    inference_prompts = prompts.get("inference") or _derive_inference_prompts(bec.column_prompts)
    return TrainConfig(**data), bec, inference_prompts


def _derive_inference_prompts(column_prompts: dict[str, str]) -> dict[str, str]:
    """Default {query, document} from column prompts when `inference:` is absent.

    Convention: anchor → 'query', positive → 'document'. Asymmetric setups
    (rare) should set `prompts.inference` explicitly.
    """
    if not column_prompts:
        return {}
    out: dict[str, str] = {}
    if (q := column_prompts.get("anchor")) is not None:
        out["query"] = q
    if (d := column_prompts.get("positive")) is not None:
        out["document"] = d
    return out


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
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Path to a TrainConfig YAML file")
    p.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        metavar="CHECKPOINT_DIR",
        help="Resume training. Omit a path to auto-detect the latest checkpoint.",
    )
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
    cfg, bec, inference_prompts = _load_config_with_bi_encoder_block(args.config)
    setup_wandb(cfg)

    # ------------------------------------------------------------------
    # 1. Load the prepared dataset.
    # ------------------------------------------------------------------
    print(f"[1/5] Loading dataset from '{cfg.processed_dir}' …")
    datasets = load_from_disk(cfg.processed_dir)
    _truncate(datasets, args)

    shape = validate_dataset(datasets)
    print(f"      shape={shape}  columns={datasets['train'].column_names}")
    print(datasets)

    prep_meta = load_preprocessing_meta(cfg.processed_dir)
    if prep_meta is not None and prep_meta.get("tokenizer"):
        assert_tokenizer_matches(prep_meta, model_name=cfg.model_name)

    # Loss vs shape sanity check — fail before any GPU work.
    if bec.loss_name == "DistillKLDivLoss" and shape != "distill":
        raise ValueError(
            f"DistillKLDivLoss requires a 'distill' dataset (multi-negative + "
            f"teacher score 'label' column); got shape {shape!r}. Run "
            f"scripts/embedding_bi_encoder/score_with_cross_encoder.py first."
        )
    if shape == "distill" and bec.loss_name not in {"DistillKLDivLoss"}:
        print(
            f"      [warn] dataset shape is 'distill' but loss is {bec.loss_name!r}. "
            f"The 'label' column will be ignored by this loss."
        )

    # ------------------------------------------------------------------
    # 2. Load the SentenceTransformer backbone.
    # ------------------------------------------------------------------
    print(f"[2/5] Loading '{cfg.model_name}' …")
    max_seq_length = (prep_meta or {}).get("max_seq_length")
    model = load_sentence_transformer(
        cfg.model_name,
        max_seq_length=max_seq_length,
        attn_implementation=cfg.attn_implementation,
        inference_prompts=inference_prompts,
    )
    if inference_prompts:
        print(f"      inference prompts → {inference_prompts}")
    if bec.matryoshka.enabled:
        native = model.get_sentence_embedding_dimension()
        if native not in bec.matryoshka.dims:
            print(
                f"      [warn] matryoshka.dims does not include the model's "
                f"native dim ({native}). Add it as the first entry to keep "
                f"full-quality embeddings available at inference."
            )

    # ------------------------------------------------------------------
    # 3. Build the loss + evaluator.
    # ------------------------------------------------------------------
    print(f"[3/5] Building loss={bec.loss_name}"
          f"{' (+ Matryoshka)' if bec.matryoshka.enabled else ''} and evaluator …")
    loss = build_loss(model, bec)

    eval_split = "validation" if "validation" in datasets else "test"
    evaluator = None
    metric_key = None
    if eval_split in datasets:
        evaluator, eval_shape, metric_key = build_evaluator(
            datasets[eval_split],
            name=eval_split,
            batch_size=cfg.per_device_eval_batch_size,
        )
        if eval_shape != shape:
            print(
                f"      [warn] eval split shape ({eval_shape}) differs from "
                f"train shape ({shape}). Evaluator picked from the eval split."
            )
        print(f"      evaluator metric → {metric_key}")
        if cfg.metric_for_best_model == "f1":
            print(
                f"      [info] metric_for_best_model='f1' replaced with "
                f"{metric_key!r} (auto-detected from evaluator shape)."
            )
            cfg.metric_for_best_model = metric_key

    # ------------------------------------------------------------------
    # 4. Build trainer.
    # ------------------------------------------------------------------
    print("[4/5] Building trainer …")
    total_steps = compute_total_training_steps(len(datasets["train"]), cfg)
    training_args = build_training_args(
        cfg, num_training_steps=total_steps, bi_encoder_cfg=bec
    )
    trainer = build_trainer(
        model,
        datasets,
        training_args,
        loss,
        evaluator=evaluator,
        early_stopping_patience=cfg.early_stopping_patience,
    )

    # ------------------------------------------------------------------
    # 5. Train, then save BEFORE running test-set eval — a crash inside
    # the test loop would otherwise leave `final/` empty.
    # ------------------------------------------------------------------
    print("[5/5] Training …")
    trainer.train(resume_from_checkpoint=args.resume)

    final_dir = Path(cfg.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))

    sidecar_fields: dict = {
        "tokenizer": cfg.model_name,
        "shape": shape,
        "columns": list(datasets["train"].column_names),
        "loss": bec.loss_name,
    }
    if max_seq_length is not None:
        sidecar_fields["max_seq_length"] = int(max_seq_length)
    if bec.matryoshka.enabled:
        sidecar_fields["matryoshka_mode"] = bec.matryoshka.mode
        sidecar_fields["matryoshka_dims"] = bec.matryoshka.dims
        # Smallest dim is the conservative default for storage-bound serve.
        # Caller can still override at inference.
        sidecar_fields["default_truncate_dim"] = min(bec.matryoshka.dims)
        if bec.matryoshka.mode == "2d":
            sidecar_fields["matryoshka_n_layers_per_step"] = bec.matryoshka.n_layers_per_step
    if inference_prompts:
        sidecar_fields["inference_prompts"] = inference_prompts
    if prep_meta is not None:
        for k, v in prep_meta.items():
            sidecar_fields.setdefault(k, v)
    save_preprocessing_meta(final_dir, **sidecar_fields)
    print(f"Model saved → {final_dir}")

    if "test" in datasets:
        print("Evaluating on test split …")
        test_evaluator, _, _ = build_evaluator(
            datasets["test"],
            name="test",
            batch_size=cfg.per_device_eval_batch_size,
        )
        metrics = test_evaluator(model, output_path=str(final_dir))
        if not isinstance(metrics, dict):
            metrics = {"primary": float(metrics)}
        print(json.dumps(metrics, indent=2, default=str))
        (final_dir / "test_metrics.json").write_text(
            json.dumps(metrics, indent=2, default=str)
        )


if __name__ == "__main__":
    main()
