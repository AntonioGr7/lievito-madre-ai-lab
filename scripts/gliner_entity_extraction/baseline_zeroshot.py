#!/usr/bin/env python
"""Zero-shot baseline: how good is an *off-the-shelf* GLiNER on this task?

Answers the question every fine-tune should have to justify itself against:
"does training actually beat just prompting a pretrained model?" It evaluates a
pretrained GLiNER checkpoint on the **same** processed test split, with the
**same** scoring (strict char-offset entity-set F1), threshold tuning, and
chunking the fine-tune used — then prints a side-by-side delta against the
fine-tune's ``test_metrics.json``.

Usage
-----
# Baseline = the base model the fine-tune started from (the most direct
# "did training help?" comparison — this is the default when --model is omitted):
python scripts/gliner_entity_extraction/baseline_zeroshot.py \\
    --config examples/gliner_entity_extraction/pii/configs/a100_nemotron.yaml

# Baseline = a dedicated off-the-shelf PII model:
python scripts/gliner_entity_extraction/baseline_zeroshot.py \\
    --config examples/gliner_entity_extraction/pii/configs/a100_nemotron.yaml \\
    --model knowledgator/gliner-pii-base-v1.0

# Pin a fixed threshold instead of tuning on validation:
python scripts/gliner_entity_extraction/baseline_zeroshot.py --config ... --threshold 0.5
"""
import argparse
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from lievito_madre_ai_lab.finetuning.encoder.gliner_entity_extraction.dataset import (
    load_processed,
    to_native_dataset,
)
from lievito_madre_ai_lab.finetuning.encoder.gliner_entity_extraction.evaluate import (
    evaluate_split,
    tune_threshold,
)
from lievito_madre_ai_lab.shared.config import TrainConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True,
                   help="The training config — supplies processed_dir, label_aliases, "
                        "chunking, eval batch size, and (by default) the baseline model.")
    p.add_argument("--model", default=None,
                   help="Baseline checkpoint (hub id or path). Default: the config's "
                        "model_name, i.e. the base model the fine-tune started from.")
    p.add_argument("--finetuned-dir", default=None,
                   help="Fine-tuned model dir holding test_metrics.json for the delta. "
                        "Default: <output_dir>/final.")
    p.add_argument("--threshold", type=float, default=None,
                   help="Fixed decision threshold. If omitted, tune on validation "
                        "(same protocol the fine-tune uses).")
    p.add_argument("--threshold-objective", choices=["f1", "f2"], default=None,
                   help="What threshold tuning maximises: f1 (balanced) or f2 "
                        "(recall-weighted, for PII). Default: the config's "
                        "gliner.threshold_objective, else f1.")
    p.add_argument("--no-chunking", action="store_true",
                   help="Disable sliding-window chunking even if the config enables it.")
    p.add_argument("--device", default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--max-test-samples", type=int, default=None)
    p.add_argument("--out", default=None,
                   help="Where to write baseline metrics JSON. "
                        "Default: <finetuned-dir>/baseline_metrics.json.")
    return p.parse_args()


def _load_config(path: str) -> tuple[TrainConfig, dict, int, dict, str]:
    """Return (cfg, label_aliases, chunk_stride, {'max_words': ...}, threshold_objective)."""
    data = yaml.safe_load(Path(path).read_text())
    gliner = data.pop("gliner", {}) or {}
    cfg = TrainConfig(**data)
    chunk_cfg = gliner.get("chunking", {}) or {}
    chunk = {
        "stride": int(chunk_cfg.get("stride", 64)),
        "max_words": chunk_cfg.get("max_words"),
    }
    objective = str(gliner.get("threshold_objective", "f1"))
    return cfg, gliner.get("label_aliases", {}) or {}, chunk["stride"], chunk, objective


def _resolve_device(name: str | None):
    import torch
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_baseline(model_name: str, device):
    import torch.nn as nn
    from gliner import GLiNER

    model = GLiNER.from_pretrained(model_name)
    model.eval()
    if hasattr(model, "model") and isinstance(model.model, nn.Module):
        model.model.to(device)
    else:
        model.to(device)
    return model


def _print_comparison(model_name: str, baseline: dict, finetuned: dict | None) -> None:
    def fmt(d: dict, key: str) -> str:
        v = d.get(key)
        return f"{v:.4f}" if isinstance(v, (int, float)) else "  -   "

    print("\n" + "=" * 64)
    print(f"Zero-shot baseline ({model_name}) vs fine-tune")
    print("=" * 64)
    header = f"{'metric':<22}{'baseline':>12}{'finetuned':>12}{'Δ':>10}"
    print(header)
    print("-" * len(header))
    rows = [
        ("closed precision", "closed", "precision", "test_closed_precision"),
        ("closed recall",    "closed", "recall",    "test_closed_recall"),
        ("closed F1",        "closed", "f1",        "test_closed_f1"),
        ("zero-shot precision", "zeroshot", "precision", "test_zeroshot_precision"),
        ("zero-shot recall",    "zeroshot", "recall",    "test_zeroshot_recall"),
        ("zero-shot F1",        "zeroshot", "f1",        "test_zeroshot_f1"),
    ]
    for label, section, mkey, ftkey in rows:
        b = baseline.get(section, {}).get(mkey)
        if b is None:
            continue
        ft = finetuned.get(ftkey) if finetuned else None
        delta = (f"{ft - b:+.4f}" if isinstance(ft, (int, float)) else "   -   ")
        ft_s = f"{ft:.4f}" if isinstance(ft, (int, float)) else "  -   "
        print(f"{label:<22}{b:>12.4f}{ft_s:>12}{delta:>10}")
    if finetuned is None:
        print("\n(no fine-tuned test_metrics.json found — baseline only)")
    print("=" * 64)


def main() -> None:
    import torch

    args = parse_args()
    cfg, label_aliases, chunk_stride, chunk, cfg_objective = _load_config(args.config)
    if args.no_chunking:
        chunk_stride = -1
    objective = args.threshold_objective or cfg_objective

    model_name = args.model or cfg.model_name
    finetuned_dir = Path(args.finetuned_dir) if args.finetuned_dir else Path(cfg.output_dir) / "final"

    print(f"[1/4] Loading processed dataset from {cfg.processed_dir} …")
    datasets, train_types, holdout_types = load_processed(cfg.processed_dir)
    if args.max_eval_samples and "validation" in datasets:
        datasets["validation"] = datasets["validation"].select(
            range(min(args.max_eval_samples, len(datasets["validation"])))
        )
    if args.max_test_samples and "test" in datasets:
        datasets["test"] = datasets["test"].select(
            range(min(args.max_test_samples, len(datasets["test"])))
        )
    print(f"      train_types ({len(train_types)}), holdout_types ({len(holdout_types)})")

    device = _resolve_device(args.device)
    print(f"[2/4] Loading baseline {model_name} on {device} (zero-shot, no training) …")
    model = _load_baseline(model_name, device)

    effective_max_words = (
        chunk["max_words"] if chunk["max_words"] is not None
        else int(getattr(model.config, "max_len", 384))
    )
    native_max_words = effective_max_words if chunk_stride >= 0 else None
    splitter = model.data_processor.words_splitter

    with torch.inference_mode():
        # Threshold: tune on validation (matching the fine-tune protocol) unless pinned.
        threshold = args.threshold
        if threshold is None and "validation" in datasets and len(datasets["validation"]):
            val_native = to_native_dataset(
                datasets["validation"], splitter,
                max_words=native_max_words, stride=chunk_stride,
                desc="Chunking validation for threshold tuning",
            )
            threshold, best_m, _ = tune_threshold(
                model, val_native, labels=train_types,
                label_aliases=label_aliases,
                batch_size=cfg.per_device_eval_batch_size,
                objective=objective,
                progress=True,
            )
            print(
                f"      tuned baseline threshold = {threshold} (objective={objective}; "
                f"val f1={best_m['f1']:.4f} f2={best_m['f2']:.4f})"
            )
        elif threshold is None:
            threshold = 0.5
            print("      no validation split — using threshold 0.5")
        else:
            print(f"      using fixed threshold = {threshold}")

        print("[3/4] Evaluating baseline on the test split …")
        test_native = to_native_dataset(
            datasets["test"], splitter,
            max_words=native_max_words, stride=chunk_stride,
            desc="Chunking test for baseline eval",
        )
        closed = evaluate_split(
            model, test_native, labels=train_types,
            label_aliases=label_aliases, threshold=threshold,
            batch_size=cfg.per_device_eval_batch_size,
            progress=True,
        )
        zeroshot = {}
        if holdout_types:
            zeroshot = evaluate_split(
                model, test_native, labels=holdout_types,
                label_aliases=label_aliases, threshold=threshold,
                batch_size=cfg.per_device_eval_batch_size,
                progress=True,
            )

    baseline = {
        "baseline_model": model_name,
        "threshold": threshold,
        "threshold_objective": objective,
        "chunking": {"stride": chunk_stride if chunk_stride >= 0 else None,
                     "max_words": native_max_words},
        "closed": closed,
        "zeroshot": zeroshot,
    }

    finetuned = None
    ft_metrics_path = finetuned_dir / "test_metrics.json"
    if ft_metrics_path.exists():
        finetuned = json.loads(ft_metrics_path.read_text())

    _print_comparison(model_name, baseline, finetuned)

    out = Path(args.out) if args.out else finetuned_dir / "baseline_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(baseline, indent=2))
    print(f"[4/4] Baseline metrics -> {out}")


if __name__ == "__main__":
    main()
