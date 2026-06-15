#!/usr/bin/env python
"""Build a free-form (task=text) dataset from CORD-v2 (receipt → JSON).

This is the worked example of the *generic* text-target contract: each row is
just ``{"image", "prompt", "response"}`` where ``response`` is the exact string
the assistant should produce — here, the receipt's structured contents as JSON.
The same train script, collator and predictor that do box/point grounding handle
this unchanged; only the target serialization (none — it's already text) and the
eval metric (exact-match / token-F1 instead of IoU) differ.

Swap in any image→text task the same way: produce rows with a ``response``
string (a tool call, an answer, a caption, …) and set ``vlm.task: text``.

Usage
-----
python examples/json_extraction/dataset/prepare_cord.py --out-dir data/processed/cord
"""
import argparse
import json
from pathlib import Path

from datasets import DatasetDict, load_dataset

from vlm_finetuning.dataset import validate_row
from vlm_finetuning.shared.preprocessing import save_preprocessing_meta

DEFAULT_DATASET_ID = "naver-clova-ix/cord-v2"
PROMPT = "Extract the contents of this receipt as a JSON object."


def _convert_row(row, *, prompt: str) -> dict:
    """CORD-v2 row -> generic text-target row. ``ground_truth`` is a JSON string
    whose ``gt_parse`` holds the structured menu; we target its compact form."""
    gt = json.loads(row["ground_truth"])
    parse = gt.get("gt_parse", gt)
    response = json.dumps(parse, ensure_ascii=False, separators=(",", ":"))
    return {"prompt": prompt, "response": response}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    p.add_argument("--limit", type=int, default=None, help="Cap rows per split (debugging).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    print(f"[1/3] Loading {args.dataset_id} from HF Hub …")
    raw = load_dataset(args.dataset_id)
    if args.limit:
        raw = DatasetDict({s: ds.select(range(min(args.limit, len(ds)))) for s, ds in raw.items()})

    print("[2/3] Converting to the text-target contract …")
    processed = {}
    for split_name, ds in raw.items():
        drop = [c for c in ds.column_names if c != "image"]
        processed[split_name] = ds.map(
            lambda row: _convert_row(row, prompt=PROMPT),
            remove_columns=drop,
            desc=f"Converting {split_name}",
        )
    processed = DatasetDict(processed)

    for split_name, split in processed.items():
        if len(split) and (errs := validate_row(split[0])):
            raise ValueError(
                f"split {split_name!r} row 0 violates the contract:\n  - " + "\n  - ".join(errs)
            )

    print("[3/3] Saving …")
    out_dir.mkdir(parents=True, exist_ok=True)
    processed.save_to_disk(str(out_dir))
    save_preprocessing_meta(
        out_dir, source=args.dataset_id, task="text", default_prompt=PROMPT,
    )
    print(f"       Saved -> {out_dir}")


if __name__ == "__main__":
    main()
