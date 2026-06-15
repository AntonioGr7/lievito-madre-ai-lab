#!/usr/bin/env python
"""Build a grounding DatasetDict from CPPE-5 (medical PPE detection).

This is the worked example of the grounding contract that
`vlm_finetuning.dataset` expects. CPPE-5 ships COCO-style pixel boxes
(``[x, y, w, h]``) with 5 categories; we convert each box to a normalized
``[x1, y1, x2, y2]`` (and its centre point), so the same processed dataset can
train either the box task or the point task.

Copy this file, adapt it to your corpus, and place it anywhere outside
`scripts/` — the train script only cares about the on-disk output:

  data/processed/<task>/
    train/, validation/, test/   <- HF DatasetDict shards (with an Image column)
    labels.json                  <- ["Coverall", "Face Shield", ...]
    preprocessing.json           <- source + coordinate-scheme provenance

Row schema in every split:

  {"image": PIL.Image,
   "prompt": str,
   "objects": [{"label": str, "box": [x1,y1,x2,y2], "point": [x,y]}, ...]}   # normalized [0,1]

Usage
-----
python examples/grounding/dataset/prepare_cppe5.py --out-dir data/processed/cppe5
"""
import argparse
import json
from pathlib import Path

from datasets import DatasetDict, load_dataset

from vlm_finetuning.dataset import collect_labels, validate_row
from vlm_finetuning.shared.preprocessing import save_preprocessing_meta

DEFAULT_DATASET_ID = "cppe-5"
# Fallback if the dataset doesn't expose category names in its features.
_FALLBACK_NAMES = ["Coverall", "Face_Shield", "Gloves", "Goggles", "Mask"]


def _humanize(name: str) -> str:
    """`Face_Shield` -> `Face Shield` — nicer as a natural-language label."""
    return name.replace("_", " ").replace("-", " ").strip()


def _category_names(raw) -> list[str]:
    try:
        names = raw["train"].features["objects"].feature["category"].names
        if names:
            return list(names)
    except Exception:
        pass
    return _FALLBACK_NAMES


def _build_prompt(label_names: list[str], task: str) -> str:
    cats = ", ".join(label_names)
    what = "bounding box" if task == "box" else "centre point"
    return (
        "Detect every item of personal protective equipment in the image. "
        f"Possible categories: {cats}. "
        f"For each item, output its category label followed by its {what} "
        "as <box> coordinate tokens, one per line."
    )


def _convert_row(row, *, id2label: dict[int, str], prompt: str) -> dict:
    """CPPE-5 COCO-format row -> grounding contract row (normalized coords)."""
    image = row["image"]
    W = row.get("width") or image.width
    H = row.get("height") or image.height
    obj = row["objects"]
    objects = []
    for bbox, cat in zip(obj["bbox"], obj["category"]):
        x, y, w, h = bbox
        x1, y1, x2, y2 = x / W, y / H, (x + w) / W, (y + h) / H
        # Clamp to the image and drop degenerate boxes.
        x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
        x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        objects.append({
            "label": id2label[int(cat)],
            "box": [x1, y1, x2, y2],
            "point": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
        })
    return {"prompt": prompt, "objects": objects}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    p.add_argument("--task", choices=["box", "point"], default="box",
                   help="Only affects the default prompt text written into the rows; "
                        "both box and point coordinates are stored regardless.")
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="CPPE-5 has no validation split; carve this fraction off train.")
    p.add_argument("--limit", type=int, default=None, help="Cap rows per split (debugging).")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    print(f"[1/4] Loading {args.dataset_id} from HF Hub …")
    raw = load_dataset(args.dataset_id, trust_remote_code=args.trust_remote_code)
    names = _category_names(raw)
    id2label = {i: _humanize(n) for i, n in enumerate(names)}
    label_names = [id2label[i] for i in range(len(names))]
    prompt = _build_prompt(label_names, args.task)
    print(f"      categories: {label_names}")

    # CPPE-5 ships train/test; carve a validation split off train.
    if "validation" not in raw and args.val_fraction > 0:
        split = raw["train"].train_test_split(test_size=args.val_fraction, seed=42)
        raw = DatasetDict({"train": split["train"], "validation": split["test"], "test": raw["test"]})
        print(f"      carved validation off train: "
              f"train={len(raw['train'])} validation={len(raw['validation'])} test={len(raw['test'])}")

    if args.limit:
        raw = DatasetDict({
            s: ds.select(range(min(args.limit, len(ds)))) for s, ds in raw.items()
        })

    print("[2/4] Converting to the grounding contract …")
    processed = {}
    for split_name, ds in raw.items():
        keep = "image"
        drop = [c for c in ds.column_names if c != keep]
        processed[split_name] = ds.map(
            lambda row: _convert_row(row, id2label=id2label, prompt=prompt),
            remove_columns=drop,
            desc=f"Converting {split_name}",
        )
    processed = DatasetDict(processed)

    # Surface contract violations at prep time, not at training step 1.
    for split_name, split in processed.items():
        if len(split) and (errs := validate_row(split[0])):
            raise ValueError(
                f"split {split_name!r} row 0 violates the grounding contract:\n  - "
                + "\n  - ".join(errs)
            )

    print("[3/4] Saving …")
    out_dir.mkdir(parents=True, exist_ok=True)
    processed.save_to_disk(str(out_dir))
    labels = collect_labels(processed)
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2))

    save_preprocessing_meta(
        out_dir,
        source=args.dataset_id,
        task=args.task,
        coord_bins=1000,
        labels=labels,
        default_prompt=prompt,
    )
    print(f"[4/4] Saved -> {out_dir}")
    print(f"       labels -> {labels}")


if __name__ == "__main__":
    main()
