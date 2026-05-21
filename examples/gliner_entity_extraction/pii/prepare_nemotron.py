#!/usr/bin/env python
"""Build a char-offset GLiNER DatasetDict from nvidia/Nemotron-PII.

Sibling of `prepare_openpii.py` — same output contract, different corpus:

  data/processed/<task>/
    train/, validation/, test/   <- HF DatasetDict shards
    train_types.json             <- labels the model trains on
    holdout_types.json           <- labels held out for zero-shot probe

Row schema in every split:

  {"text": str, "spans": [{"start": int, "end": int, "label": str}, ...]}

Nemotron-PII quirks (vs OpenPII)
--------------------------------
- Text column is `text` (OpenPII: `source_text`); span column is `spans`
  (OpenPII: `privacy_mask`).
- Each span is stored as a Python-repr string inside a Sequence column — we
  decode with `ast.literal_eval` before doing anything else.
- Per-span keys are `start`/`end`/`text`/`label`; `end` is already half-open.
- Labels are snake_case (~55 categories: first_name, ssn, medical_record_number,
  swift_bic, …). Vocabulary is collected dynamically.
- Splits are train (50k) + test (50k); no validation. We carve a slice off
  train (--val-size, default 10%).

Usage
-----
python examples/gliner_entity_extraction/pii/prepare_nemotron.py \\
    --out-dir data/processed/nemotron-gliner
"""
import argparse
import ast
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Value, load_dataset, load_from_disk

from lievito_madre_ai_lab.finetuning.encoder.gliner_entity_extraction.dataset import (
    collect_entity_types,
    partition_entity_types,
    validate_row,
)
from lievito_madre_ai_lab.shared.preprocessing import save_preprocessing_meta


DEFAULT_HOLDOUT = ["medical_record_number", "swift_bic", "age"]
DEFAULT_DATASET_ID = "nvidia/Nemotron-PII"
TEXT_COL = "text"
MASK_COL = "spans"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    p.add_argument("--raw-dir", default=None,
                   help="Load from a saved raw snapshot instead of HF Hub")
    p.add_argument("--limit", type=int, default=None,
                   help="Stream only N rows per split — useful for smoke-tests.")
    p.add_argument("--holdout-types", nargs="*", default=DEFAULT_HOLDOUT)
    p.add_argument("--val-size", type=float, default=0.1,
                   help="Fraction of train to reserve as validation (default: 0.1)")
    p.add_argument("--locale", choices=["us", "intl"], default=None,
                   help="Keep only rows with this locale. Default: both.")
    p.add_argument("--domains", default=None,
                   help="Comma-separated domain names to keep (exact match). Default: all.")
    return p.parse_args()


def _load_from_hub(dataset_id: str, *, limit: int | None) -> DatasetDict:
    if limit is None:
        return load_dataset(dataset_id)
    streamed = load_dataset(dataset_id, streaming=True)
    return DatasetDict({
        split: Dataset.from_list(list(ds.take(limit)))
        for split, ds in streamed.items()
    })


def _parse_spans(raw: DatasetDict, mask_col: str) -> DatasetDict:
    """Decode spans from Python-repr strings to typed dicts.

    Nemotron-PII stores each span as a string inside a Sequence column. We
    normalise to a Sequence of structs so the rest of the pipeline can do
    `span["label"]` etc. without further parsing.
    """
    span_feature = [{
        "start": Value("int64"),
        "end": Value("int64"),
        "text": Value("string"),
        "label": Value("string"),
    }]

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
                parsed = ast.literal_eval(spans)
            elif isinstance(spans, list):
                parsed = [ast.literal_eval(s) if isinstance(s, str) else s for s in spans]
            else:
                parsed = spans or []
            decoded.append([_normalise(span) for span in parsed])
        return {mask_col: decoded}

    out = {}
    for split, ds in raw.items():
        new_features = Features({**ds.features, mask_col: span_feature})
        out[split] = ds.map(
            _decode, batched=True, features=new_features,
            desc=f"Parsing spans [{split}]",
        )
    return DatasetDict(out)


def _carve_validation(raw: DatasetDict, val_size: float, seed: int = 42) -> DatasetDict:
    """Split train into train + validation; leave test untouched."""
    split = raw["train"].train_test_split(test_size=val_size, seed=seed)
    return DatasetDict({
        "train":      split["train"],
        "validation": split["test"],
        "test":       raw["test"],
    })


def _convert_to_char_offset(
    row: dict, *, allowed_labels: frozenset[str],
) -> dict:
    """Filter the row's spans to `allowed_labels` and emit the char-offset
    contract (drops the `text` field; keeps only start/end/label)."""
    text = row[TEXT_COL]
    spans = []
    for span in row[MASK_COL]:
        label = span.get("label")
        if label not in allowed_labels:
            continue
        s, e = int(span["start"]), int(span["end"])
        # Nemotron-PII's `end` is already half-open against `text`.
        if e <= s or s < 0 or e > len(text):
            continue
        spans.append({"start": s, "end": e, "label": label})
    return {"text": text, "spans": spans}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    # 1. Load the raw corpus -------------------------------------------
    if args.raw_dir:
        print(f"[1/4] Loading raw corpus from {args.raw_dir} …")
        raw = load_from_disk(args.raw_dir)
    else:
        print(f"[1/4] Loading {args.dataset_id} from HF Hub (limit={args.limit}) …")
        raw = _load_from_hub(args.dataset_id, limit=args.limit)
    print(raw)

    raw = _parse_spans(raw, MASK_COL)

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
        keep = {d.strip() for d in args.domains.split(",") if d.strip()}
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

    if "validation" not in raw:
        raw = _carve_validation(raw, args.val_size)
        print(
            f"      Carved validation from train "
            f"(val_size={args.val_size}): "
            f"train={len(raw['train'])} "
            f"validation={len(raw['validation'])} "
            f"test={len(raw['test'])}"
        )

    # 2. Vocabulary partition ------------------------------------------
    print("[2/4] Collecting entity vocabulary …")
    all_types = collect_entity_types(raw, span_col=MASK_COL)
    print(f"      {len(all_types)} labels: {all_types}")
    train_types, holdout_types = partition_entity_types(all_types, args.holdout_types)
    print(f"      train_types ({len(train_types)}):   {train_types}")
    print(f"      holdout_types ({len(holdout_types)}): {holdout_types}")

    # 3. Convert to char-offset format ---------------------------------
    print("[3/4] Converting to char-offset format …")
    train_set = frozenset(train_types)
    full_set = frozenset(all_types)

    processed = {}
    for split_name in ("train", "validation"):
        if split_name not in raw:
            continue
        processed[split_name] = raw[split_name].map(
            lambda row: _convert_to_char_offset(row, allowed_labels=train_set),
            remove_columns=raw[split_name].column_names,
            desc=f"Converting {split_name} -> char-offset",
        )

    if "test" in raw:
        # The test split keeps all labels — closed-set and zero-shot views
        # are produced at eval time by filtering on `labels`.
        processed["test"] = raw["test"].map(
            lambda row: _convert_to_char_offset(row, allowed_labels=full_set),
            remove_columns=raw["test"].column_names,
            desc="Converting test -> char-offset",
        )

    processed = DatasetDict(processed)

    for split_name, split in processed.items():
        if len(split) == 0:
            continue
        errs = validate_row(split[0])
        if errs:
            raise ValueError(
                f"split {split_name!r} row 0 violates the char-offset contract:\n  - "
                + "\n  - ".join(errs)
            )

    # 4. Save -----------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    processed.save_to_disk(str(out_dir))
    (out_dir / "train_types.json").write_text(json.dumps(train_types, indent=2))
    (out_dir / "holdout_types.json").write_text(json.dumps(holdout_types, indent=2))

    save_preprocessing_meta(
        out_dir,
        source=args.dataset_id,
        locale=args.locale,
        domains=args.domains,
        text_col=TEXT_COL,
        mask_col=MASK_COL,
        train_types=train_types,
        holdout_types=holdout_types,
    )

    print(f"[4/4] Saved -> {out_dir}")
    print(f"       train_types   -> {out_dir / 'train_types.json'}")
    print(f"       holdout_types -> {out_dir / 'holdout_types.json'}")
    print(f"       preprocessing.json written (tokenizer-agnostic)")


if __name__ == "__main__":
    main()
