#!/usr/bin/env python
"""Download and preprocess nvidia/Nemotron-PII for HF Trainer (token classification).

Schema vs OpenPII
-----------------
- Text column   : `text`         (OpenPII uses `source_text`)
- Spans column  : `spans`        (OpenPII uses `privacy_mask`)
- Span format   : {"start": int, "end": int, "text": str, "label": str}
                  end is half-open — compatible with tokenize_for_trainer as-is.
- Label names   : snake_case     (first_name, last_name, …)
                  OpenPII uses UPPERCASE (GIVENNAME, SURNAME, …).
                  Labels are collected dynamically so the output vocabulary
                  matches exactly what is in the corpus.
- Splits        : train (50k) + test (50k); no validation.
                  A validation slice is carved from train (--val-size, default 10%).

Full label set (~55 categories):
  first_name, last_name, date_of_birth, age, gender, ssn, phone_number, email,
  street_address, city, state, county, postcode, country, coordinate,
  account_number, bank_routing_number, swift_bic, credit_card_number, cvv,
  company_name, occupation, employee_id, medical_record_number, blood_type,
  user_name, password, pin, ip, mac_address, url, vehicle_identifier,
  license_plate, date, date_time, time, … (see dataset card for full list)

Usage examples
--------------
# Full dataset, mdeberta tokenizer
python examples/token_classification/prepare_nemotron_pii.py \
    --out-dir data/processed/nemotron-pii

# Quick smoke-test on 2 000 rows
python examples/token_classification/prepare_nemotron_pii.py \
    --out-dir data/processed/nemotron-pii-smoke --limit 2000

# US locale only
python examples/token_classification/prepare_nemotron_pii.py \
    --out-dir data/processed/nemotron-pii-us --locale us

# Different tokenizer
python examples/token_classification/prepare_nemotron_pii.py \
    --out-dir data/processed/nemotron-pii --model bert-base-cased
"""

import argparse
import ast
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Value, load_dataset, load_from_disk

from lievito_madre_ai_lab.encoder.token_classification.dataset import (
    collect_entity_types,
    preview_alignment,
    save_preprocessing_meta,
    tokenize_for_trainer,
)

DEFAULT_DATASET_ID = "nvidia/Nemotron-PII"
DEFAULT_MODEL = "microsoft/mdeberta-v3-base"
DEFAULT_MAX_LEN = 512
TEXT_COL = "text"
MASK_COL = "spans"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", default="data/processed/nemotron-pii",
                   help="Where to save the tokenized dataset")
    p.add_argument("--raw-dir", default=None,
                   help="Load from a saved raw snapshot instead of HF Hub")
    p.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="HF model name for the tokenizer")
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument(
        "--stride", type=int, default=128,
        help=(
            "Overlapping tokens between consecutive chunks of long documents. "
            "Default 128. Use 0 for chunking without overlap, or -1 to disable "
            "chunking and fall back to plain truncation (legacy behaviour)."
        ),
    )
    p.add_argument(
        "--val-size", type=float, default=0.1,
        help="Fraction of train to reserve as validation split (default: 0.1)",
    )
    p.add_argument(
        "--locale", choices=["us", "intl"], default=None,
        help="Keep only rows with this locale. Default: both.",
    )
    p.add_argument(
        "--domains", default=None,
        help="Comma-separated domain names to keep (exact match). Default: all.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Stream only N rows per split — useful for smoke-tests.",
    )
    p.add_argument(
        "--no-label-all-tokens",
        dest="label_all_tokens", action="store_false",
        help="Only label the first subword of each entity (legacy). Default labels all.",
    )
    p.set_defaults(label_all_tokens=True)
    return p.parse_args()


def _load_from_hub(
    dataset_id: str, *, limit: int | None,
) -> DatasetDict:
    if limit is None:
        return load_dataset(dataset_id)

    streamed = load_dataset(dataset_id, streaming=True)
    return DatasetDict({
        split: Dataset.from_list(list(ds.take(limit)))
        for split, ds in streamed.items()
    })


def _parse_spans(raw: DatasetDict, mask_col: str) -> DatasetDict:
    """Decode spans from JSON strings to dicts if the column was stored that way.

    Nemotron-PII stores each span as a Python-repr string inside a Sequence
    column. This normalises them to plain dicts so the rest of the pipeline
    (collect_entity_types, tokenize_for_trainer) can use .get("label") etc.
    """
    span_feature = [{
        "start": Value("int64"),
        "end": Value("int64"),
        "text": Value("string"),
        "label": Value("string"),
    }]

    def _parse_one(s: str) -> dict:
        return ast.literal_eval(s)

    def _normalise(span: dict) -> dict:
        text = span.get("text")
        return {
            "start": span.get("start"),
            "end": span.get("end"),
            "text": "" if text is None else str(text),
            "label": span.get("label"),
        }

    def _decode(batch: dict) -> dict:
        decoded = []
        for spans in batch[mask_col]:
            if isinstance(spans, str):
                # Whole-row string: "[{...}, {...}]"
                parsed = _parse_one(spans)
            elif isinstance(spans, list):
                # Sequence of per-span strings or already-parsed dicts
                parsed = [_parse_one(s) if isinstance(s, str) else s for s in spans]
            else:
                parsed = spans or []
            decoded.append([_normalise(span) for span in parsed])
        return {mask_col: decoded}

    result = {}
    for split, ds in raw.items():
        new_features = Features({**ds.features, mask_col: span_feature})
        result[split] = ds.map(
            _decode, batched=True,
            features=new_features,
            desc=f"Parsing spans [{split}]",
        )
    return DatasetDict(result)


def _carve_validation(raw: DatasetDict, val_size: float, seed: int = 42) -> DatasetDict:
    """Split train into train + validation; leave test untouched."""
    split = raw["train"].train_test_split(test_size=val_size, seed=seed)
    return DatasetDict({
        "train":      split["train"],
        "validation": split["test"],
        "test":       raw["test"],
    })


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    if args.raw_dir:
        print(f"[1/4] Loading raw snapshot from {args.raw_dir} …")
        raw = load_from_disk(args.raw_dir)
    else:
        print(f"[1/4] Loading {args.dataset_id} from HF Hub (limit={args.limit}) …")
        raw = _load_from_hub(args.dataset_id, limit=args.limit)

    print(raw)

    # Nemotron stores each span as a JSON string inside the Sequence column.
    # Decode to dicts so the rest of the pipeline can access .get("label") etc.
    raw = _parse_spans(raw, MASK_COL)

    # ------------------------------------------------------------------
    # 2. Filter (locale / domain)
    # ------------------------------------------------------------------
    filters_applied: list[str] = []

    if args.locale:
        raw = DatasetDict({
            split: ds.filter(
                lambda row: row["locale"] == args.locale,
                desc=f"locale={args.locale} [{split}]",
            )
            for split, ds in raw.items()
        })
        filters_applied.append(f"locale={args.locale!r}")

    if args.domains:
        keep = {d.strip() for d in args.domains.split(",")}
        raw = DatasetDict({
            split: ds.filter(
                lambda row: row["domain"] in keep,
                desc=f"domain filter [{split}]",
            )
            for split, ds in raw.items()
        })
        filters_applied.append(f"domains={keep}")

    if filters_applied:
        print(f"      Filters applied: {', '.join(filters_applied)}")
        print(raw)

    # ------------------------------------------------------------------
    # 3. Carve validation split
    # ------------------------------------------------------------------
    if "validation" not in raw:
        raw = _carve_validation(raw, args.val_size)
        print(
            f"      Carved validation from train "
            f"(val_size={args.val_size}): "
            f"train={len(raw['train'])} "
            f"validation={len(raw['validation'])} "
            f"test={len(raw['test'])}"
        )

    # ------------------------------------------------------------------
    # 4. Collect label vocabulary
    # ------------------------------------------------------------------
    # Nemotron uses snake_case labels; collect them dynamically so the
    # output vocabulary matches exactly what is present in the filtered data.
    print("[2/4] Collecting entity-type vocabulary …")
    entity_types = collect_entity_types(raw, mask_col=MASK_COL)
    print(f"      {len(entity_types)} types: {entity_types}")

    # ------------------------------------------------------------------
    # 5. Tokenize + align BIO labels
    # ------------------------------------------------------------------
    stride = None if args.stride < 0 else args.stride
    stride_desc = "off (truncate)" if stride is None else f"stride={stride}"
    print(
        f"[3/4] Tokenising with '{args.model}' "
        f"(max_length={args.max_length}, sliding window: {stride_desc}) …"
    )
    pre_counts = {split: len(ds) for split, ds in raw.items()}
    processed = tokenize_for_trainer(
        raw,
        args.model,
        text_col=TEXT_COL,
        mask_col=MASK_COL,
        max_length=args.max_length,
        label_all_tokens=args.label_all_tokens,
        entity_types=entity_types,
        stride=stride,
    )
    print(processed)
    for split, pre in pre_counts.items():
        post = len(processed[split])
        gain = post - pre
        if gain:
            print(
                f"      [{split}] {pre} docs → {post} chunks "
                f"(+{gain} from long documents)"
            )

    preview_alignment(processed, args.model)

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    processed.save_to_disk(str(out_dir))
    save_preprocessing_meta(
        out_dir,
        source=args.dataset_id,
        tokenizer=args.model,
        text_col=TEXT_COL,
        mask_col=MASK_COL,
        max_length=args.max_length,
        stride=stride,
        label_all_tokens=args.label_all_tokens,
        entity_types=entity_types,
    )
    print(f"[4/4] Processed dataset saved → {out_dir}")
    print(f"      preprocessing.json written (max_length={args.max_length}, "
          f"stride={stride}) so serve.py can rediscover the settings.")


if __name__ == "__main__":
    main()
