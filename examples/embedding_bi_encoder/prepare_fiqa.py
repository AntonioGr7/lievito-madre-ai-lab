#!/usr/bin/env python
"""Build (anchor, positive) pairs from FiQA-2018 for bi-encoder training.

Pulls `BeIR/fiqa` (corpus + queries) and `BeIR/fiqa-qrels` (relevance
judgments) from HuggingFace, joins them on query-id / corpus-id, and writes
a DatasetDict with train / validation / test splits in the bi-encoder
`pair` shape.

FiQA is the classic finance retrieval dataset from BeIR. It is NOT part of
RTEB-finance (the three open RTEB-finance English tasks are FinanceBench,
HC3Finance, and FinQA), so training on it does not leak into RTEB-finance
evaluation. The framework's smoke tests show the same `pair` shape; the
trainer's evaluator will report `eval_validation_cosine_mrr@10` as the
selection metric.

Output schema (pair shape):
    anchor   : query text
    positive : title + "\\n" + text   (BeIR convention; title omitted when empty)

Usage
-----
python examples/embedding_bi_encoder/prepare_fiqa.py
# writes:
#   data/processed/fiqa-bi-encoder/{train,validation,test}/...
#   data/processed/fiqa-bi-encoder/preprocessing.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

from lievito_madre_ai_lab.finetuning.embedding.bi_encoder.dataset import (
    save_preprocessing_meta,
    validate_dataset,
)

CORPUS_REPO = "BeIR/fiqa"
QRELS_REPO = "BeIR/fiqa-qrels"


def _format_passage(title: str | None, text: str) -> str:
    """Prepend title with a newline when present (standard BeIR convention)."""
    title = (title or "").strip()
    text = (text or "").strip()
    if title:
        return f"{title}\n{text}"
    return text


def _join_split(
    qrels: Dataset, queries: dict[str, str], corpus: dict[str, str]
) -> Dataset:
    """Materialize (anchor, positive) rows from a qrels split.

    BeIR qrels carry only positive judgments (score >= 1), so every kept row
    becomes one (query, positive-doc) training pair. Queries with multiple
    annotated positives expand to multiple rows — `NO_DUPLICATES` in the
    batch sampler stops two of them landing in the same MNRL batch.
    """
    anchors: list[str] = []
    positives: list[str] = []
    skipped = 0
    for row in qrels:
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        score = int(row["score"])
        if score < 1:
            continue
        if qid not in queries or cid not in corpus:
            skipped += 1
            continue
        anchors.append(queries[qid])
        positives.append(corpus[cid])
    if skipped:
        print(f"      [warn] {skipped} qrels rows skipped (missing query / corpus id)")
    return Dataset.from_dict({"anchor": anchors, "positive": positives})


def build_datasets() -> DatasetDict:
    print(f"      loading {CORPUS_REPO} corpus + queries …")
    corpus_ds = load_dataset(CORPUS_REPO, "corpus", split="corpus")
    queries_ds = load_dataset(CORPUS_REPO, "queries", split="queries")
    corpus = {
        str(r["_id"]): _format_passage(r.get("title"), r["text"]) for r in corpus_ds
    }
    queries = {str(r["_id"]): r["text"] for r in queries_ds}
    print(f"      corpus={len(corpus):,}  queries={len(queries):,}")

    print(f"      loading {QRELS_REPO} qrels …")
    qrels = load_dataset(QRELS_REPO)

    out: dict[str, Dataset] = {}
    for split_name, hf_split in (
        ("train", "train"),
        ("validation", "validation"),
        ("test", "test"),
    ):
        if hf_split not in qrels:
            print(f"      [warn] qrels has no '{hf_split}' split — skipping")
            continue
        ds = _join_split(qrels[hf_split], queries, corpus)
        print(f"      {split_name}: {len(ds):,} pairs")
        out[split_name] = ds

    if "train" not in out:
        raise RuntimeError("qrels did not yield a 'train' split — refusing to save.")
    return DatasetDict(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", default="data/processed/fiqa-bi-encoder")
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Recorded in preprocessing.json and used by the trainer as the "
             "model truncation cap. 512 fits ~99% of FiQA passages.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("[1/3] Building FiQA-2018 (anchor, positive) pairs …")
    datasets = build_datasets()
    shape = validate_dataset(datasets)
    print(f"      shape={shape}")
    print(datasets)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[2/3] Saving to {out_dir} …")
    datasets.save_to_disk(str(out_dir))

    print("[3/3] Writing preprocessing.json …")
    save_preprocessing_meta(
        out_dir,
        source="BeIR/fiqa + BeIR/fiqa-qrels",
        max_seq_length=int(args.max_seq_length),
    )
    print(f"Done. Point a YAML's processed_dir at {out_dir} to train on it.")


if __name__ == "__main__":
    main()
