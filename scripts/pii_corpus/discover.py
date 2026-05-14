#!/usr/bin/env python
"""Inspect a PII dataset on the HF Hub to map its schema.

Streams the dataset (does NOT download the full archive), prints columns +
feature types, a handful of sample rows, and a best-effort auto-detected
label vocabulary — enough to write a correct adapter without trial-and-error.

Run once per source you want to add::

    python scripts/pii_corpus/discover.py nvidia/Nemotron-PII
    python scripts/pii_corpus/discover.py gretelai/gretel-pii-masking-en-v1
    python scripts/pii_corpus/discover.py piimb/privy

then paste the output back so we can implement the matching adapter.

If a dataset is gated, run ``huggingface-cli login`` first.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Any, Iterable

from datasets import load_dataset


# Likely keys an annotation dict uses to carry the entity-type label.
# Order matters — the first match wins so we don't double-count.
LABEL_KEY_CANDIDATES = (
    "label", "entity_type", "entity", "pii_type", "type",
    "tag", "category", "class", "pii_class",
)


def scan_labels(value: Any, counter: Counter) -> None:
    """Walk an arbitrary JSON-ish value and increment ``counter`` for any
    label-shaped string found inside dicts."""
    if isinstance(value, dict):
        for key in LABEL_KEY_CANDIDATES:
            v = value.get(key)
            if isinstance(v, str):
                counter[v] += 1
                break  # only one label per dict
        for v in value.values():
            if isinstance(v, (dict, list)):
                scan_labels(v, counter)
    elif isinstance(value, list):
        for v in value:
            scan_labels(v, counter)


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f" …(+{len(text) - limit} chars)"


def take(stream: Iterable, n: int) -> list:
    out = []
    for i, row in enumerate(stream):
        if i >= n:
            break
        out.append(row)
    return out


def inspect_split(split: str, stream, *, max_rows: int, vocab_scan_rows: int,
                  value_trunc: int) -> None:
    print(f"\n──── split = {split} ────")

    # Features are available on most HF datasets even in streaming mode.
    features = getattr(stream, "features", None)
    if features:
        print("Declared columns:")
        for col, ft in features.items():
            print(f"  - {col}: {ft}")
    else:
        print("Declared columns: (not exposed by stream — will infer from rows)")

    # Stream just enough rows for samples + vocab scan in one pass.
    target = max(max_rows, vocab_scan_rows)
    print(f"\nStreaming up to {target} rows for inspection…", flush=True)
    rows = take(stream, target)
    if not rows:
        print("  (split is empty)")
        return
    print(f"  pulled {len(rows)} rows.")

    if not features:
        print("\nInferred top-level keys:")
        for k, v in rows[0].items():
            print(f"  - {k}: {type(v).__name__}")

    print(f"\nFirst {min(max_rows, len(rows))} row(s):")
    for i, row in enumerate(rows[:max_rows]):
        print(f"\n  [row {i}]")
        for k, v in row.items():
            v_str = json.dumps(v, ensure_ascii=False, default=str)
            print(f"    {k}: {truncate(v_str, value_trunc)}")

    vocab: Counter = Counter()
    for row in rows[:vocab_scan_rows]:
        scan_labels(row, vocab)
    if vocab:
        print(f"\nAuto-detected label vocabulary (top {min(50, len(vocab))} "
              f"from {min(vocab_scan_rows, len(rows))} rows; "
              f"{sum(vocab.values())} hits total):")
        for label, count in vocab.most_common(50):
            print(f"  {count:7d}  {label}")
        if len(vocab) > 50:
            print(f"  … {len(vocab) - 50} more distinct labels not shown")
    else:
        print("\nAuto-detected label vocabulary: (none — inspect rows above; "
              "the dataset may store labels as BIO sequences, masked text, "
              "or under an uncommon key)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("dataset_id", help="HF Hub dataset id, e.g. nvidia/Nemotron-PII")
    ap.add_argument("--config", default=None, help="Optional dataset config name")
    ap.add_argument("--split", default=None,
                    help="Limit inspection to a single split (e.g. 'train')")
    ap.add_argument("--max-rows", type=int, default=2,
                    help="Sample rows to print per split (default: 2)")
    ap.add_argument("--vocab-scan-rows", type=int, default=500,
                    help="Rows to scan when guessing the label vocabulary")
    ap.add_argument("--value-trunc", type=int, default=600,
                    help="Truncate any single field's JSON over this many chars")
    ap.add_argument("--no-stream", action="store_true",
                    help="Fall back to full download (use only if streaming fails)")
    args = ap.parse_args()

    print(f"Loading '{args.dataset_id}' "
          f"({'streaming' if not args.no_stream else 'full download'})…",
          flush=True)
    kwargs: dict = {"streaming": not args.no_stream}
    if args.config:
        kwargs["name"] = args.config

    try:
        raw = load_dataset(args.dataset_id, **kwargs)
    except Exception as e:
        print(f"\n!! load_dataset failed: {e!r}", file=sys.stderr)
        msg = str(e).lower()
        if "gated" in msg or "401" in msg or "403" in msg:
            print("   → looks gated; run `huggingface-cli login` and accept the "
                  "dataset terms on the Hub.", file=sys.stderr)
        sys.exit(1)

    splits = list(raw.keys())
    print(f"Splits: {splits}")

    for split, stream in raw.items():
        if args.split and split != args.split:
            continue
        inspect_split(split, stream,
                      max_rows=args.max_rows,
                      vocab_scan_rows=args.vocab_scan_rows,
                      value_trunc=args.value_trunc)


if __name__ == "__main__":
    main()
