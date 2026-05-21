#!/usr/bin/env python
"""Add cross-encoder teacher scores to a multi-negative dataset.

Reads a DatasetDict with columns ``(anchor, positive, neg_1, …, neg_N)``
(produced by `mine_hard_negatives.py`), scores every (anchor, candidate)
pair with a cross-encoder, and appends a ``label`` column carrying the list
of teacher scores per row.

The output is a `distill`-shaped dataset trainable with `DistillKLDivLoss`:
the bi-encoder student learns to match the cross-encoder's score
distribution over candidates (listwise KL).

Usage
-----
python scripts/embedding_bi_encoder/score_with_cross_encoder.py \
    --input-dataset data/processed/my-pairs-mined \
    --output-dir    data/processed/my-pairs-distill \
    --cross-encoder BAAI/bge-reranker-v2-m3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict, load_from_disk
from sentence_transformers import CrossEncoder

from lievito_madre_ai_lab.finetuning.embedding.bi_encoder.distill import (
    ScoringConfig,
    add_cross_encoder_scores,
)
from lievito_madre_ai_lab.shared.preprocessing import (
    load_preprocessing_meta,
    save_preprocessing_meta,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input-dataset", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--cross-encoder", required=True,
        help="Stronger model than the bi-encoder student. Typical picks: "
             "BAAI/bge-reranker-v2-m3, mixedbread-ai/mxbai-rerank-large-v2, "
             "Alibaba-NLP/gte-multilingual-reranker-base.",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--activation-fn", default=None,
        help="Optional activation on cross-encoder logits. None keeps raw logits "
             "(recommended — DistillKLDivLoss softmaxes internally). "
             "'sigmoid' / 'softmax' override.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[1/3] Loading dataset from '{args.input_dataset}' …")
    datasets: DatasetDict = load_from_disk(args.input_dataset)
    print(datasets)

    print(f"[2/3] Loading cross-encoder '{args.cross_encoder}' …")
    cross_encoder = CrossEncoder(args.cross_encoder)
    cfg = ScoringConfig(
        batch_size=args.batch_size,
        activation_fn=args.activation_fn,
    )

    out = DatasetDict()
    for split_name, split in datasets.items():
        print(f"      [{split_name}] scoring {len(split)} rows …")
        out[split_name] = add_cross_encoder_scores(
            split, cross_encoder=cross_encoder, cfg=cfg,
        )

    print(f"[3/3] Saving distill dataset → {args.output_dir} …")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(out_dir))

    prep = load_preprocessing_meta(args.input_dataset) or {}
    prep.setdefault("source", args.input_dataset)
    prep["distillation"] = {
        "cross_encoder": args.cross_encoder,
        "activation_fn": args.activation_fn,
    }
    save_preprocessing_meta(out_dir, **prep)


if __name__ == "__main__":
    main()
