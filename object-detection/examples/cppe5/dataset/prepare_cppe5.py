#!/usr/bin/env python
"""Build a COCO-format detection DatasetDict from CPPE-5 (medical PPE).

The worked example of the contract `object_detection.dataset` expects. CPPE-5
already ships COCO-style pixel boxes; we only normalize the column layout and
carve a validation split. Same dataset as the `vlm-finetuning` grounding
example, so you can compare a classical detector against VLM grounding head-to-head.

  data/processed/<task>/
    train/, validation/, test/   <- HF DatasetDict shards (with an Image column)
    labels.json                  <- ordered category names (index = category_id)
    preprocessing.json           <- provenance

Row schema:
  {"image": PIL, "image_id": int,
   "objects": {"bbox": [[x,y,w,h],...], "category_id": [...], "area": [...], "iscrowd": [...]}}

Usage
-----
python examples/cppe5/dataset/prepare_cppe5.py --out-dir data/processed/cppe5
"""
import argparse
import json
from pathlib import Path

from datasets import DatasetDict, load_dataset

from object_detection.dataset import validate_row
from object_detection.shared.preprocessing import save_preprocessing_meta

DEFAULT_DATASET_ID = "cppe-5"
_FALLBACK_NAMES = ["Coverall", "Face_Shield", "Gloves", "Goggles", "Mask"]


def _category_names(raw) -> list[str]:
    try:
        names = raw["train"].features["objects"].feature["category"].names
        if names:
            return list(names)
    except Exception:
        pass
    return _FALLBACK_NAMES


def _convert_row(row) -> dict:
    """CPPE-5 row -> the COCO contract (rename `category`->`category_id`, add iscrowd)."""
    obj = row["objects"]
    bboxes = [[float(v) for v in b] for b in obj["bbox"]]
    cats = [int(c) for c in obj["category"]]
    areas = [float(a) for a in obj.get("area", [b[2] * b[3] for b in bboxes])]
    return {
        "image_id": int(row["image_id"]),
        "objects": {
            "bbox": bboxes,
            "category_id": cats,
            "area": areas,
            "iscrowd": [0] * len(bboxes),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="CPPE-5 has no validation split; carve this fraction off train.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    print(f"[1/4] Loading {args.dataset_id} from HF Hub …")
    raw = load_dataset(args.dataset_id, trust_remote_code=args.trust_remote_code)
    label_names = _category_names(raw)
    print(f"      categories: {label_names}")

    if "validation" not in raw and args.val_fraction > 0:
        split = raw["train"].train_test_split(test_size=args.val_fraction, seed=42)
        raw = DatasetDict({"train": split["train"], "validation": split["test"], "test": raw["test"]})
        print(f"      carved validation off train: "
              f"train={len(raw['train'])} validation={len(raw['validation'])} test={len(raw['test'])}")

    if args.limit:
        raw = DatasetDict({s: ds.select(range(min(args.limit, len(ds)))) for s, ds in raw.items()})

    print("[2/4] Converting to the COCO contract …")
    processed = {}
    for split_name, ds in raw.items():
        drop = [c for c in ds.column_names if c != "image"]
        processed[split_name] = ds.map(_convert_row, remove_columns=drop, desc=f"Converting {split_name}")
    processed = DatasetDict(processed)

    for split_name, split in processed.items():
        if len(split) and (errs := validate_row(split[0])):
            raise ValueError(f"split {split_name!r} row 0 violates the contract:\n  - " + "\n  - ".join(errs))

    print("[3/4] Saving …")
    out_dir.mkdir(parents=True, exist_ok=True)
    processed.save_to_disk(str(out_dir))
    (out_dir / "labels.json").write_text(json.dumps(label_names, indent=2))
    save_preprocessing_meta(out_dir, source=args.dataset_id, task="object-detection", labels=label_names)
    print(f"[4/4] Saved -> {out_dir}")
    print(f"       labels -> {label_names}")


if __name__ == "__main__":
    main()
