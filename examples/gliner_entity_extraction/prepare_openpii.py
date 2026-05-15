#!/usr/bin/env python
"""Build a char-offset GLiNER DatasetDict from the ai4privacy OpenPII corpus.

This is the worked example of the char-offset contract that
`lievito_madre_ai_lab.encoder.gliner_entity_extraction.dataset` expects.
Copy this file, adapt it to your corpus, and place it anywhere outside
`scripts/` — the train script only cares about the on-disk output.

Each per-corpus prepare script you write should produce the same on-disk layout:

  data/processed/<task>/
    train/, validation/, test/   <- HF DatasetDict shards
    train_types.json             <- labels the model trains on
    holdout_types.json           <- labels held out for zero-shot probe

Row schema in every split:

  {"text": str, "spans": [{"start": int, "end": int, "label": str}, ...]}

Usage
-----
python examples/gliner_entity_extraction/prepare_openpii.py \\
    --out-dir data/processed/pii-gliner
"""
import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from lievito_madre_ai_lab.encoder.gliner_entity_extraction.dataset import (
    collect_entity_types,
    partition_entity_types,
    validate_row,
)


DEFAULT_HOLDOUT = ["PASSPORTNUM", "DRIVERLICENSENUM", "AGE"]
DEFAULT_DATASET_ID = "ai4privacy/open-pii-masking-500k-ai4privacy"


def _load_openpii_from_hub(
    dataset_id: str, *, languages: list[str] | None, limit: int | None,
) -> DatasetDict:
    if limit is None:
        raw = load_dataset(dataset_id)
    else:
        streamed = load_dataset(dataset_id, streaming=True)
        raw = DatasetDict({
            split: Dataset.from_list(list(ds.take(limit)))
            for split, ds in streamed.items()
        })

    if languages:
        allow = set(languages)
        raw = DatasetDict({
            split: ds.filter(lambda r: r["language"] in allow,
                             desc=f"Filtering {split} by language")
            for split, ds in raw.items()
        })
    return raw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    p.add_argument("--raw-dir", default=None)
    p.add_argument("--languages", default="en")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--holdout-types", nargs="*", default=DEFAULT_HOLDOUT)
    p.add_argument("--text-col", default="source_text")
    p.add_argument("--mask-col", default="privacy_mask")
    return p.parse_args()


def _convert_to_char_offset(
    row: dict, *, allowed_labels: frozenset[str], text_col: str, mask_col: str,
) -> dict:
    """Filter the row's privacy_mask to `allowed_labels` and emit the
    char-offset contract. End is normalized to half-open."""
    text = row[text_col]
    spans = []
    for span in row[mask_col]:
        label = span.get("label")
        if label not in allowed_labels:
            continue
        s, e = int(span["start"]), int(span["end"])
        # OpenPII's `end` is already half-open against `source_text`; if a
        # downstream corpus uses inclusive end, fix it here (e += 1).
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
        langs = [l.strip() for l in args.languages.split(",") if l.strip()] or None
        print(f"[1/4] Loading {args.dataset_id} from HF Hub (languages={langs}, limit={args.limit}) …")
        raw = _load_openpii_from_hub(args.dataset_id, languages=langs, limit=args.limit)
        if "test" not in raw and "validation" in raw:
            split = raw["validation"].train_test_split(test_size=0.5, seed=42)
            raw = DatasetDict({
                "train": raw["train"],
                "validation": split["train"],
                "test": split["test"],
            })
            print(f"      Carved test split off validation: "
                  f"validation={len(raw['validation'])} test={len(raw['test'])}")
    print(raw)

    # 2. Vocabulary partition ------------------------------------------
    print("[2/4] Collecting entity vocabulary …")
    all_types = collect_entity_types(raw, span_col=args.mask_col)
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
            lambda row: _convert_to_char_offset(
                row, allowed_labels=train_set,
                text_col=args.text_col, mask_col=args.mask_col,
            ),
            remove_columns=raw[split_name].column_names,
            desc=f"Converting {split_name} -> char-offset",
        )

    if "test" in raw:
        # The test split keeps all labels — closed-set and zero-shot views
        # are produced at eval time by filtering on `labels`.
        processed["test"] = raw["test"].map(
            lambda row: _convert_to_char_offset(
                row, allowed_labels=full_set,
                text_col=args.text_col, mask_col=args.mask_col,
            ),
            remove_columns=raw["test"].column_names,
            desc="Converting test -> char-offset",
        )

    processed = DatasetDict(processed)

    # Surface contract violations at prep time rather than at training step 1.
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
    print(f"[4/4] Saved -> {out_dir}")
    print(f"       train_types   -> {out_dir / 'train_types.json'}")
    print(f"       holdout_types -> {out_dir / 'holdout_types.json'}")


if __name__ == "__main__":
    main()
