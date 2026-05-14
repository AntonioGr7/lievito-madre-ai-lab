#!/usr/bin/env python
"""Download and preprocess a HuggingFace text-classification dataset for HF Trainer.

Steps
-----
1. Load the raw dataset from a configurable source (HF Hub, local disk, Google Drive)
2. Save the raw Arrow snapshot to --raw-dir  (skipped when source is already local)
3. Tokenize with AutoTokenizer
4. Save the processed Arrow dataset to --out-dir

Usage examples
--------------
# Default: dair-ai/emotion from HF Hub, answerdotai/ModernBERT-base tokenizer
python scripts/prepare_dataset.py

# Different dataset
python scripts/prepare_dataset.py --dataset-id cardiffnlp/tweet_eval --dataset-config sentiment

# Use a local snapshot already on disk
python scripts/prepare_dataset.py --source local --local-path data/raw/emotion

# Download from a shared Google Drive folder (Arrow format)
python scripts/prepare_dataset.py --source drive --drive-id <FILE_ID> --drive-dest data/raw/emotion --drive-folder

# Change tokenizer and max sequence length
python scripts/prepare_dataset.py --model roberta-base --max-length 64
"""

import argparse
from pathlib import Path

from lievito_madre_ai_lab.encoder.text_classification.dataset import tokenize_for_trainer
from lievito_madre_ai_lab.shared.sources import DriveSource, HFSource, LocalSource

DEFAULT_DATASET_ID = "dair-ai/emotion"
DEFAULT_MODEL = "answerdotai/ModernBERT-base"
DEFAULT_MAX_LEN = 128


def dataset_slug(dataset_id: str) -> str:
    """Turn 'dair-ai/emotion' into 'emotion' for use in directory names."""
    return dataset_id.split("/")[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="HF Hub dataset id")
    p.add_argument("--dataset-config", default=None, help="Dataset config name (e.g. 'sentiment' for tweet_eval)")
    p.add_argument(
        "--source",
        choices=["hf", "local", "drive"],
        default="hf",
        help="Where to load the raw dataset from (default: hf)",
    )

    # local / drive options
    p.add_argument("--local-path", default=None, help="Path for --source local (defaults to data/raw/<slug>)")
    p.add_argument("--drive-id", default=None, help="Google Drive file/folder ID")
    p.add_argument("--drive-dest", default=None, help="Local destination for Drive download")
    p.add_argument("--drive-folder", action="store_true", help="Download a whole Drive folder instead of a single file")

    # tokenizer options
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model name for the tokenizer")
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LEN, help="Tokenizer max_length")
    p.add_argument("--text-col", default="text", help="Name of the text column")
    p.add_argument("--label-col", default="label", help="Name of the label column")

    # output
    p.add_argument("--raw-dir", default=None, help="Where to save the raw Arrow snapshot (defaults to data/raw/<slug>)")
    p.add_argument("--out-dir", default=None, help="Where to save the tokenized dataset (defaults to data/processed/<slug>)")

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
    raw = source.load()
    print(raw)

    if args.source == "hf":
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw.save_to_disk(str(raw_dir))
        print(f"      Raw snapshot saved → {raw_dir}")

    # ------------------------------------------------------------------
    # 3. Tokenize
    # ------------------------------------------------------------------
    print(f"[2/3] Tokenizing with '{args.model}' (max_length={args.max_length}) …")
    processed = tokenize_for_trainer(
        raw,
        args.model,
        text_col=args.text_col,
        label_col=args.label_col,
        max_length=args.max_length,
    )
    print(processed)

    # ------------------------------------------------------------------
    # 4. Save processed dataset
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    processed.save_to_disk(str(out_dir))
    print(f"[3/3] Processed dataset saved → {out_dir}")


if __name__ == "__main__":
    main()
