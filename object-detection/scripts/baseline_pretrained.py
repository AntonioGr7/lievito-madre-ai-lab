#!/usr/bin/env python
"""Pretrained baseline: how good is the off-the-shelf detector before fine-tuning?

The honest bar every fine-tune must clear. Evaluates a pretrained checkpoint on
the **same** processed test split with the **same** COCO-mAP scoring, then prints
a delta against the fine-tune's ``test_metrics.json``.

Note on label spaces: an off-the-shelf COCO detector predicts COCO's 80 classes,
not your dataset's — so a raw mAP against your labels is usually near zero and is
mostly a sanity check that the pipeline runs. The real comparison is fine-tune vs
fine-tune (e.g. different backbones/configs); point ``--model`` at another saved
run for that.

Usage
-----
python scripts/baseline_pretrained.py --config examples/cppe5/configs/dfine_x.yaml
python scripts/baseline_pretrained.py --config ... --model outputs/other_run/final
"""
import argparse
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from object_detection.dataset import load_processed
from object_detection.evaluate import compute_map
from object_detection.shared.config import TrainConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--model", default=None, help="Baseline checkpoint. Default: the config's model_name.")
    p.add_argument("--finetuned-dir", default=None, help="Default: <output_dir>/final.")
    p.add_argument("--device", default=None)
    p.add_argument("--max-test-samples", type=int, default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def _print_comparison(model_name: str, baseline: dict, finetuned: dict | None) -> None:
    print("\n" + "=" * 60)
    print(f"Pretrained baseline ({model_name}) vs fine-tune")
    print("=" * 60)
    header = f"{'metric':<14}{'baseline':>12}{'finetuned':>12}{'Δ':>10}"
    print(header)
    print("-" * len(header))
    for key, label in [("map", "mAP"), ("map_50", "mAP@.50"), ("map_75", "mAP@.75")]:
        b = baseline.get(key)
        if b is None:
            continue
        ft = finetuned.get(f"test_{key}") if finetuned else None
        ft_s = f"{ft:.4f}" if isinstance(ft, (int, float)) else "  -   "
        delta = f"{ft - b:+.4f}" if isinstance(ft, (int, float)) else "   -   "
        print(f"{label:<14}{b:>12.4f}{ft_s:>12}{delta:>10}")
    if finetuned is None:
        print("\n(no fine-tuned test_metrics.json found — baseline only)")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    data = yaml.safe_load(Path(args.config).read_text())
    det = data.pop("detection", {}) or {}
    cfg = TrainConfig(**data)

    model_name = args.model or cfg.model_name
    finetuned_dir = Path(args.finetuned_dir) if args.finetuned_dir else Path(cfg.output_dir) / "final"

    print(f"[1/3] Loading processed dataset from {cfg.processed_dir} …")
    datasets, id2label = load_processed(cfg.processed_dir)
    if args.max_test_samples and "test" in datasets:
        datasets["test"] = datasets["test"].select(range(min(args.max_test_samples, len(datasets["test"]))))

    print(f"[2/3] Loading baseline {model_name} (no fine-tuning) …")
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForObjectDetection.from_pretrained(model_name)
    model.eval()
    if args.device:
        model.to(args.device)
    elif __import__("torch").cuda.is_available():
        model.to("cuda")

    print("[3/3] Evaluating baseline on the test split …")
    baseline = compute_map(
        model, image_processor, datasets["test"],
        id2label=id2label,
        batch_size=cfg.per_device_eval_batch_size,
        threshold=float(det.get("eval", {}).get("threshold", 0.0)),
        progress=True,
    )
    baseline["baseline_model"] = model_name

    finetuned = None
    ft_path = finetuned_dir / "test_metrics.json"
    if ft_path.exists():
        finetuned = json.loads(ft_path.read_text())

    _print_comparison(model_name, baseline, finetuned)

    out = Path(args.out) if args.out else finetuned_dir / "baseline_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(baseline, indent=2))
    print(f"Baseline metrics -> {out}")


if __name__ == "__main__":
    main()
