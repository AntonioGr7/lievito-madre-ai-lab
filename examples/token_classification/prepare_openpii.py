#!/usr/bin/env python
"""Download and preprocess a token-classification dataset for HF Trainer.

This is the worked example targeting the ai4privacy OpenPII corpus.
Copy this file, adapt it to your corpus, and place it anywhere outside
`scripts/` — the train script only cares about the on-disk output.

Steps
-----
1. Load the raw dataset from a configurable source (HF Hub, local disk, Google Drive)
2. Save the raw Arrow snapshot to --raw-dir  (skipped when source is already local)
3. Tokenize with AutoTokenizer + BIO label alignment from character spans
4. Save the processed Arrow dataset to --out-dir

Usage examples
--------------
# Default: ai4privacy PII dataset from HF Hub, bert-base-multilingual-cased tokenizer
python examples/token_classification/prepare_openpii.py

# English only
python examples/token_classification/prepare_openpii.py --languages en

# Multiple languages
python examples/token_classification/prepare_openpii.py --languages en,fr,de

# Use a local snapshot already on disk
python examples/token_classification/prepare_openpii.py --source local --local-path data/raw/pii

# Different model tokenizer
python examples/token_classification/prepare_openpii.py --model xlm-roberta-base

# Only label the first subword of each entity (legacy HF recipe; default labels all)
python examples/token_classification/prepare_openpii.py --no-label-all-tokens
"""

import argparse
from pathlib import Path

from datasets import DatasetDict

from lievito_madre_ai_lab.encoder.token_classification.dataset import (
    ENTITY_TYPES,
    preview_alignment,
    save_preprocessing_meta,
    tokenize_for_trainer,
)
from lievito_madre_ai_lab.shared.sources import DriveSource, HFSource, LocalSource

DEFAULT_DATASET_ID = "ai4privacy/open-pii-masking-500k-ai4privacy"
DEFAULT_MODEL = "microsoft/mdeberta-v3-base"
DEFAULT_MAX_LEN = 512


def dataset_slug(dataset_id: str) -> str:
    return dataset_id.split("/")[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="HF Hub dataset id")
    p.add_argument("--dataset-config", default=None, help="Dataset config name")
    p.add_argument(
        "--source",
        choices=["hf", "local", "drive"],
        default="hf",
        help="Where to load the raw dataset from (default: hf)",
    )

    # local / drive options
    p.add_argument("--local-path", default=None, help="Path for --source local")
    p.add_argument("--drive-id", default=None, help="Google Drive file/folder ID")
    p.add_argument("--drive-dest", default=None, help="Local destination for Drive download")
    p.add_argument("--drive-folder", action="store_true", help="Download a whole Drive folder")

    # dataset filtering
    p.add_argument(
        "--languages",
        default=None,
        help="Comma-separated language codes to keep, e.g. 'en,fr,de'. Default: all.",
    )

    # tokenizer options
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model name for the tokenizer")
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LEN, help="Tokenizer max_length")
    p.add_argument(
        "--stride", type=int, default=128,
        help=(
            "Overlapping tokens between consecutive chunks of long documents. "
            "Default 128. Use 0 for chunking without overlap, or -1 to disable "
            "chunking and fall back to plain truncation (legacy behaviour)."
        ),
    )
    p.add_argument("--text-col", default="source_text", help="Name of the text column")
    p.add_argument("--mask-col", default="privacy_mask", help="Name of the privacy mask column")
    p.add_argument(
        "--no-label-all-tokens",
        dest="label_all_tokens",
        action="store_false",
        help="Only label the first subword of each entity (legacy). Default labels all subwords with I-.",
    )
    p.set_defaults(label_all_tokens=True)

    # output
    p.add_argument("--raw-dir", default=None, help="Where to save the raw Arrow snapshot")
    p.add_argument("--out-dir", default=None, help="Where to save the tokenized dataset")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    slug = dataset_slug(args.dataset_id)
    raw_dir = Path(args.raw_dir or f"data/raw/{slug}")
    out_dir = Path(args.out_dir or f"data/processed/{slug}")
    local_path = Path(args.local_path or raw_dir)
    drive_dest = args.drive_dest or str(raw_dir)

    # ------------------------------------------------------------------
    # 1. Build the data source
    # ------------------------------------------------------------------
    if args.source == "hf":
        source = HFSource(args.dataset_id, config_name=args.dataset_config)
    elif args.source == "local":
        source = LocalSource(local_path)
    elif args.source == "drive":
        if not args.drive_id:
            raise ValueError("--drive-id is required when --source drive")
        source = DriveSource(
            file_id=args.drive_id,
            dest=drive_dest,
            is_folder=args.drive_folder,
        )
    else:
        raise ValueError(f"Unknown source: {args.source}")

    # ------------------------------------------------------------------
    # 2. Load & persist the raw snapshot
    # ------------------------------------------------------------------
    print(f"[1/3] Loading '{args.dataset_id}' from {args.source.upper()} …")
    raw: DatasetDict = source.load()

    # Optional language filter
    if args.languages:
        langs = {l.strip() for l in args.languages.split(",")}
        raw = DatasetDict({
            split: ds.filter(lambda row: row["language"] in langs, desc=f"Filtering {split}")
            for split, ds in raw.items()
        })
        print(f"      Kept languages: {langs}")

    print(raw)

    if args.source == "hf":
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw.save_to_disk(str(raw_dir))
        print(f"      Raw snapshot saved → {raw_dir}")

    # ------------------------------------------------------------------
    # 3. Tokenize + align BIO labels
    # ------------------------------------------------------------------
    stride = None if args.stride < 0 else args.stride
    stride_desc = "off (truncate)" if stride is None else f"stride={stride}"
    print(
        f"[2/3] Tokenising with '{args.model}' "
        f"(max_length={args.max_length}, sliding window: {stride_desc}) …"
    )
    pre_counts = {split: len(ds) for split, ds in raw.items()}
    processed = tokenize_for_trainer(
        raw,
        args.model,
        text_col=args.text_col,
        mask_col=args.mask_col,
        max_length=args.max_length,
        label_all_tokens=args.label_all_tokens,
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

    # Sanity-check the BIO alignment by eyeballing a few tokenized examples.
    # Cheap, and the kind of thing that catches tokenizer-specific bugs
    # (e.g. SentencePiece "▁" eating the leading space) before training.
    preview_alignment(processed, args.model)

    # ------------------------------------------------------------------
    # 4. Save processed dataset
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    processed.save_to_disk(str(out_dir))
    save_preprocessing_meta(
        out_dir,
        source=args.dataset_id,
        tokenizer=args.model,
        text_col=args.text_col,
        mask_col=args.mask_col,
        max_length=args.max_length,
        stride=stride,
        label_all_tokens=args.label_all_tokens,
        entity_types=ENTITY_TYPES,
    )
    print(f"[3/3] Processed dataset saved → {out_dir}")
    print(f"      preprocessing.json written (max_length={args.max_length}, "
          f"stride={stride}) so serve.py can rediscover the settings.")


if __name__ == "__main__":
    main()
