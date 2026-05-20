#!/usr/bin/env python
"""Evaluate a trained bi-encoder on the RTEB-finance English OPEN subset.

Runs the three publicly-available RTEB-finance English retrieval tasks
(FinanceBench, HC3Finance, FinQA) via the `mteb` Python library. The four
private tasks (`_EnglishFinance1..4`) are held out and can only be scored
by submitting a HuggingFace-hosted model to the RTEB maintainers
(see https://huggingface.co/spaces/embedding-benchmark/RTEB).

Requires `pip install mteb`. The model directory must be a saved
SentenceTransformer (e.g. `outputs/<run>/<exp>/final/`). Local paths are
loaded directly — no HF push needed for the open tasks.

Usage
-----
python examples/embedding_bi_encoder/eval_rteb_finance.py \\
    --model-dir outputs/bi_encoder_fiqa_ettin68m/exp_01/final

# Optional — restrict to a single task while iterating:
python examples/embedding_bi_encoder/eval_rteb_finance.py \\
    --model-dir <path> --tasks FinanceBench
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer

# RTEB-finance English OPEN subset, per the RTEB repo README
# (https://github.com/embedding-benchmark/rteb).
RTEB_FINANCE_EN_OPEN = ["FinanceBench", "HC3Finance", "FinQA"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model-dir",
        required=True,
        help="Path to a saved SentenceTransformer directory (e.g. .../final).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to write per-task results. Defaults to <model-dir>/rteb_finance_en.",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=RTEB_FINANCE_EN_OPEN,
        help=f"Override task list. Default: {RTEB_FINANCE_EN_OPEN}",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import mteb
    except ImportError as e:
        raise SystemExit(
            "mteb is not installed. Run: pip install mteb"
        ) from e

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        raise SystemExit(f"model_dir does not exist: {model_dir}")

    output_dir = (
        Path(args.output_dir) if args.output_dir else model_dir / "rteb_finance_en"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading SentenceTransformer from {model_dir} …")
    model = SentenceTransformer(str(model_dir))

    print(f"[2/3] Resolving mteb tasks: {args.tasks}")
    tasks = mteb.get_tasks(tasks=args.tasks)
    if not tasks:
        raise SystemExit(
            f"mteb.get_tasks returned no tasks for {args.tasks}. Your installed "
            f"mteb may predate the RTEB merge — upgrade with `pip install -U mteb`, "
            f"or use the dedicated runner: `python -m rteb --models <id> "
            f"--tasks {','.join(args.tasks)}` from https://github.com/embedding-benchmark/rteb."
        )
    resolved = [getattr(t, "metadata", t).name for t in tasks]
    print(f"      resolved: {resolved}")

    print(f"[3/3] Running evaluation → {output_dir}")
    evaluation = mteb.MTEB(tasks=tasks)
    evaluation.run(model, output_folder=str(output_dir))

    print(
        f"Done. Per-task JSON results under {output_dir}\n"
        f"Private tasks (_EnglishFinance1..4) require submission — see "
        f"https://huggingface.co/spaces/embedding-benchmark/RTEB"
    )


if __name__ == "__main__":
    main()
