#!/usr/bin/env python
"""Download a (sampled) slice of TeraflopAI/SEC-EDGAR as a raw JSONL corpus.

SEC-EDGAR is a finance-domain dump of US SEC filings — raw documents, no
labels. Streaming-friendly, but the full split is >8M rows, so this script
defaults to taking the first N rows for fast pipeline iteration before
committing to a full run.

The output schema (`{id, text}` per line) matches the input format expected
by [scripts/pipelines/generate_bi_encoder_pairs.py](../../scripts/pipelines/generate_bi_encoder_pairs.py),
so the file can be plugged in directly via the YAML's `input_path` or the
`--input` flag.

Examples
--------
# 1. Small sample for pipeline smoke-testing (default: 1000 docs).
python examples/embedding_bi_encoder/financial/download_sec_edgar.py

# 2. Full download — pass 0 (or a negative number) to lift the cap.
python examples/embedding_bi_encoder/financial/download_sec_edgar.py --sample-size 0

# 3. Custom sample size + output path.
python examples/embedding_bi_encoder/financial/download_sec_edgar.py \\
    --sample-size 5000 \\
    --out-path data/raw/sec-edgar-5k.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import islice
from pathlib import Path

from datasets import load_dataset

REPO_ID = "TeraflopAI/SEC-EDGAR"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--repo-id", default=REPO_ID, help="HF dataset repo id.")
    p.add_argument("--split", default="train", help="HF split to stream.")
    p.add_argument(
        "--sample-size", type=int, default=1000,
        help="Number of rows to take from the head of the stream. "
             "Pass 0 (or any value <= 0) to download the full split.",
    )
    p.add_argument(
        "--text-column", default="text",
        help="Field on each row that holds the document text.",
    )
    p.add_argument(
        "--id-column", default="id",
        help="Field that holds the document id. If a row is missing this "
             "field, a fallback id `sec-edgar-{index}` is generated.",
    )
    p.add_argument(
        "--out-path", default=None,
        help="Output JSONL path. Defaults to "
             "`data/raw/sec-edgar-{sample-size}.jsonl` (or "
             "`data/raw/sec-edgar-full.jsonl` when --sample-size <= 0).",
    )
    return p.parse_args()


def _default_out_path(sample_size: int) -> Path:
    if sample_size <= 0:
        return Path("data/raw/sec-edgar-full.jsonl")
    return Path(f"data/raw/sec-edgar-{sample_size}.jsonl")


def main() -> None:
    args = parse_args()

    out_path = Path(args.out_path) if args.out_path else _default_out_path(args.sample_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = args.sample_size if args.sample_size > 0 else None
    cap_str = f"first {cap:,}" if cap is not None else "ALL"
    print(f"[1/2] Streaming {args.repo_id} (split={args.split}, taking {cap_str} rows) …")
    stream = load_dataset(args.repo_id, split=args.split, streaming=True)
    if cap is not None:
        stream = islice(stream, cap)

    print(f"[2/2] Writing JSONL → {out_path}")
    n_written = 0
    n_skipped = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(stream):
            text = row.get(args.text_column)
            if not text or not str(text).strip():
                n_skipped += 1
                continue
            doc_id = row.get(args.id_column) or f"sec-edgar-{i}"
            f.write(json.dumps({"id": str(doc_id), "text": str(text)}, ensure_ascii=False) + "\n")
            n_written += 1
            if n_written % 1000 == 0:
                print(f"      … {n_written:,} rows written")

    print(f"Done. wrote={n_written:,}  skipped_empty={n_skipped:,}  → {out_path}")
    print(
        "Next: point the pipeline at it, e.g.\n"
        f"  python scripts/pipelines/generate_bi_encoder_pairs.py \\\n"
        f"      --config configs/pipelines/bi_encoder_pairs/default.yaml \\\n"
        f"      --input {out_path}"
    )

    # Hard-exit on success. HF datasets streaming leaves a background aiohttp
    # prefetcher running for the next shard; when we stop early (islice), it
    # races interpreter shutdown and crashes with `PyGILState_Release` after
    # the file is already safely written. Skipping finalizers avoids the noise.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
