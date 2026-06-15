#!/usr/bin/env python
"""Zero-shot baseline: how good is an *off-the-shelf* VLM on this task?

Answers the question every fine-tune should justify itself against: "does
training actually beat just prompting the pretrained model?" It evaluates a
pretrained VLM checkpoint on the **same** processed test split with the **same**
generation + grounding scoring the fine-tune uses, then prints a side-by-side
delta against the fine-tune's ``test_metrics.json``.

Usage
-----
# Baseline = the base model the fine-tune started from (default):
python scripts/baseline_zeroshot.py \\
    --config examples/grounding/configs/cppe5_qwen25vl_3b.yaml

# Baseline = a different off-the-shelf VLM:
python scripts/baseline_zeroshot.py --config ... --model Qwen/Qwen2.5-VL-7B-Instruct
"""
import argparse
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from vlm_finetuning.dataset import load_processed
from vlm_finetuning.evaluate import evaluate_split
from vlm_finetuning.shared.config import TrainConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True,
                   help="Training config — supplies processed_dir, the vlm: block, "
                        "eval batch size, and (by default) the baseline model.")
    p.add_argument("--model", default=None,
                   help="Baseline checkpoint. Default: the config's model_name.")
    p.add_argument("--finetuned-dir", default=None,
                   help="Fine-tuned model dir holding test_metrics.json for the delta. "
                        "Default: <output_dir>/final.")
    p.add_argument("--device", default=None)
    p.add_argument("--max-test-samples", type=int, default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


# Headline metrics per task (baseline key -> display label). Counts (tp/fp/fn/n)
# and the baseline_model string are skipped.
_HEADLINE = {
    "box": [("precision", "precision"), ("recall", "recall"),
            ("f1", "F1"), ("f1_iou_avg", "F1@[.5:.95]")],
    "point": [("precision", "precision"), ("recall", "recall"), ("f1", "F1")],
    "text": [("exact_match", "exact match"), ("token_f1", "token F1")],
}


def _print_comparison(model_name: str, task: str, baseline: dict, finetuned: dict | None) -> None:
    print("\n" + "=" * 64)
    print(f"Zero-shot baseline ({model_name}) vs fine-tune")
    print("=" * 64)
    header = f"{'metric':<16}{'baseline':>12}{'finetuned':>12}{'Δ':>10}"
    print(header)
    print("-" * len(header))
    for bkey, label in _HEADLINE.get(task, _HEADLINE["box"]):
        b = baseline.get(bkey)
        if b is None:
            continue
        ft = finetuned.get(f"test_{bkey}") if finetuned else None
        ft_s = f"{ft:.4f}" if isinstance(ft, (int, float)) else "  -   "
        delta = f"{ft - b:+.4f}" if isinstance(ft, (int, float)) else "   -   "
        print(f"{label:<16}{b:>12.4f}{ft_s:>12}{delta:>10}")
    if finetuned is None:
        print("\n(no fine-tuned test_metrics.json found — baseline only)")
    print("=" * 64)


def main() -> None:
    args = parse_args()
    data = yaml.safe_load(Path(args.config).read_text())
    vlm = data.pop("vlm", {}) or {}
    cfg = TrainConfig(**data)

    model_name = args.model or cfg.model_name
    finetuned_dir = Path(args.finetuned_dir) if args.finetuned_dir else Path(cfg.output_dir) / "final"
    task = str(vlm.get("task", "box"))
    coord_bins = int(vlm.get("coord_bins", 1000))
    eval_d = vlm.get("eval", {}) or {}

    print(f"[1/3] Loading processed dataset from {cfg.processed_dir} …")
    datasets, labels = load_processed(cfg.processed_dir)
    if args.max_test_samples and "test" in datasets:
        datasets["test"] = datasets["test"].select(
            range(min(args.max_test_samples, len(datasets["test"])))
        )

    print(f"[2/3] Loading baseline {model_name} (zero-shot, no training) …")
    from transformers import AutoProcessor

    from vlm_finetuning.model import _load_auto_model, resolve_torch_dtype

    model = _load_auto_model(
        model_name,
        torch_dtype=resolve_torch_dtype(cfg.precision),
        attn_implementation=cfg.attn_implementation,
        trust_remote_code=bool(vlm.get("trust_remote_code", False)),
    )
    model.eval()
    if args.device:
        model.to(args.device)
    elif __import__("torch").cuda.is_available():
        model.to("cuda")
    processor = AutoProcessor.from_pretrained(model_name)

    print("[3/3] Evaluating baseline on the test split …")
    baseline = evaluate_split(
        model, processor, datasets["test"],
        task=task, coord_bins=coord_bins,
        iou_threshold=float(eval_d.get("iou_threshold", 0.5)),
        labels=labels, system_prompt=vlm.get("system_prompt"),
        max_new_tokens=int(eval_d.get("max_new_tokens", 512)),
        batch_size=cfg.per_device_eval_batch_size,
        progress=True,
    )
    baseline["baseline_model"] = model_name

    finetuned = None
    ft_metrics_path = finetuned_dir / "test_metrics.json"
    if ft_metrics_path.exists():
        finetuned = json.loads(ft_metrics_path.read_text())

    _print_comparison(model_name, task, baseline, finetuned)

    out = Path(args.out) if args.out else finetuned_dir / "baseline_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(baseline, indent=2))
    print(f"Baseline metrics -> {out}")


if __name__ == "__main__":
    main()
