#!/usr/bin/env python
"""Mine hard negatives for a bi-encoder dataset.

Reads a (anchor, positive) DatasetDict from disk, mines ``num_negatives``
hard negatives per anchor using one of three strategies, optionally filters
them with a cross-encoder, and saves a multi-negative DatasetDict on disk.

Strategies
----------
- dense    : one dense bi-encoder retriever (default).
- bm25     : BM25 lexical retrieval. Use --strategy bm25 (no --retriever needed).
- ensemble : Union of one or more --retriever values plus optional BM25, deduped
             then cross-encoder filtered. Pass --bm25 to add BM25 to the ensemble.

Examples
--------
# 1. Dense mining with BGE-large + reranker filter
python scripts/embedding_bi_encoder/mine_hard_negatives.py \
    --input-dataset data/processed/my-pairs --output-dir data/processed/my-mined \
    --strategy dense \
    --retriever BAAI/bge-large-en-v1.5 \
    --cross-encoder BAAI/bge-reranker-v2-m3 \
    --num-negatives 5 --relative-margin 0.4

# 2. BM25 only — useful when domain has strong lexical signal
python scripts/embedding_bi_encoder/mine_hard_negatives.py \
    --input-dataset data/processed/my-pairs --output-dir data/processed/my-mined-bm25 \
    --strategy bm25 \
    --cross-encoder BAAI/bge-reranker-v2-m3

# 3. Ensemble — two dense retrievers + BM25, all unioned + filtered
python scripts/embedding_bi_encoder/mine_hard_negatives.py \
    --input-dataset data/processed/my-pairs --output-dir data/processed/my-mined-ens \
    --strategy ensemble \
    --retriever BAAI/bge-large-en-v1.5 \
    --retriever intfloat/e5-large-v2 \
    --bm25 \
    --cross-encoder BAAI/bge-reranker-v2-m3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict, load_from_disk
from sentence_transformers import CrossEncoder, SentenceTransformer

from lievito_madre_ai_lab.embedding.bi_encoder.mining import (
    MiningConfig,
    mine_ensemble_for_dataset,
    mine_hard_negatives_for_dataset,
    mine_with_bm25_for_dataset,
)
from lievito_madre_ai_lab.shared.preprocessing import (
    load_preprocessing_meta,
    save_preprocessing_meta,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input-dataset", required=True, help="Path to load_from_disk()")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--strategy", choices=["dense", "bm25", "ensemble"], default="dense",
        help="Mining strategy. See module docstring for the trade-offs.",
    )
    p.add_argument(
        "--retriever", action="append", default=[],
        help="SentenceTransformer used to retrieve candidates. Repeat for ensemble. "
             "Ignored for --strategy bm25.",
    )
    p.add_argument(
        "--bm25", action="store_true",
        help="In ensemble mode, also mine with BM25 alongside the dense retrievers.",
    )
    p.add_argument(
        "--cross-encoder", default=None,
        help="Reranker used to filter false negatives. Strongly recommended.",
    )
    p.add_argument("--num-negatives", type=int, default=5)
    p.add_argument(
        "--relative-margin", type=float, default=0.4,
        help="Drop negatives with score > (1 - relative_margin) * positive_score. "
             "0.4 implements the recipe's '60%% of positive' rule.",
    )
    p.add_argument("--margin", type=float, default=None,
                   help="Absolute alternative to --relative-margin.")
    p.add_argument("--max-score", type=float, default=None)
    p.add_argument("--min-score", type=float, default=None)
    p.add_argument("--range-min", type=int, default=10,
                   help="Skip the top-K candidates (avoid near-duplicates of the positive).")
    p.add_argument("--range-max", type=int, default=100,
                   help="Sample from within the top-K candidates.")
    p.add_argument(
        "--sampling-strategy", choices=["top", "random"], default="top",
    )
    p.add_argument("--use-faiss", action="store_true",
                   help="Use FAISS for ANN retrieval — much faster on large corpora.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--anchor-column", default="anchor")
    p.add_argument("--positive-column", default="positive")
    p.add_argument(
        "--corpus-file", default=None,
        help="Path to a newline-delimited text file (one document per line) to use as the negative pool. "
             "When set, replaces the default 'union of positives' corpus.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_corpus(path: str | None) -> list[str] | None:
    if path is None:
        return None
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def _validate_args(args: argparse.Namespace) -> None:
    if args.strategy == "dense" and len(args.retriever) != 1:
        raise SystemExit("--strategy dense needs exactly one --retriever.")
    if args.strategy == "ensemble" and not args.retriever and not args.bm25:
        raise SystemExit("--strategy ensemble needs at least one --retriever or --bm25.")
    if args.margin is not None and args.relative_margin is not None:
        # An explicit --margin disables relative_margin (ST allows only one).
        args.relative_margin = None


def main() -> None:
    args = parse_args()
    _validate_args(args)

    print(f"[1/4] Loading dataset from '{args.input_dataset}' …")
    datasets: DatasetDict = load_from_disk(args.input_dataset)
    print(datasets)

    print(f"[2/4] Loading retrievers={args.retriever or '—'}  "
          f"bm25={args.bm25}  cross_encoder={args.cross_encoder or '—'} …")
    retrievers = [SentenceTransformer(r) for r in args.retriever]
    cross_encoder = CrossEncoder(args.cross_encoder) if args.cross_encoder else None

    corpus = _load_corpus(args.corpus_file)

    cfg = MiningConfig(
        num_negatives=args.num_negatives,
        relative_margin=args.relative_margin,
        margin=args.margin,
        max_score=args.max_score,
        min_score=args.min_score,
        range_min=args.range_min,
        range_max=args.range_max,
        sampling_strategy=args.sampling_strategy,
        output_format="n-tuple",
        use_faiss=args.use_faiss,
        batch_size=args.batch_size,
        anchor_column=args.anchor_column,
        positive_column=args.positive_column,
    )

    print(f"[3/4] Mining (strategy={args.strategy}, num_negatives={cfg.num_negatives}, "
          f"relative_margin={cfg.relative_margin}, range=[{cfg.range_min}, {cfg.range_max}]) …")
    mined = DatasetDict()
    for split_name, split in datasets.items():
        pre = len(split)
        if args.strategy == "dense":
            out = mine_hard_negatives_for_dataset(
                split, retriever=retrievers[0], cross_encoder=cross_encoder,
                cfg=cfg, corpus=corpus,
            )
        elif args.strategy == "bm25":
            out = mine_with_bm25_for_dataset(
                split, cross_encoder=cross_encoder, cfg=cfg, corpus=corpus,
            )
        else:  # ensemble
            out = mine_ensemble_for_dataset(
                split, retrievers=retrievers, use_bm25=args.bm25,
                cross_encoder=cross_encoder, cfg=cfg, corpus=corpus,
            )
        mined[split_name] = out
        post = len(out)
        print(f"      [{split_name}] {pre} pairs → {post} rows ({out.column_names})")

    print(f"[4/4] Saving mined dataset → {args.output_dir} …")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mined.save_to_disk(str(out_dir))

    prep = load_preprocessing_meta(args.input_dataset) or {}
    prep.setdefault("source", args.input_dataset)
    prep["mining"] = {
        "strategy": args.strategy,
        "retrievers": args.retriever,
        "bm25": args.bm25,
        "cross_encoder": args.cross_encoder,
        "num_negatives": cfg.num_negatives,
        "relative_margin": cfg.relative_margin,
        "margin": cfg.margin,
        "max_score": cfg.max_score,
        "min_score": cfg.min_score,
        "range_min": cfg.range_min,
        "range_max": cfg.range_max,
        "sampling_strategy": cfg.sampling_strategy,
        "corpus_file": args.corpus_file,
    }
    save_preprocessing_meta(out_dir, **prep)


if __name__ == "__main__":
    main()
