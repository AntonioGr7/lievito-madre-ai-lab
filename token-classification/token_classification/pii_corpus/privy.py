"""Adapter for piimb/privy.

Source schema (from ``discover.py``)::

    full_text   : str
    masked      : str
    spans       : list[{entity_type, entity_value, start_position, end_position}]
    tags        : list[str]   # BIO tag sequence aligned to `tokens`
    tokens      : list[str]
    template_id : int
    metadata    : null

Splits: train / test / dev.

Intermediate output::

    source_text  : str
    privacy_mask : list[{label,value,start,end}]
    language     : "en"
    source       : "privy"
"""
from __future__ import annotations

from datasets import DatasetDict

from .schema import load_raw, stamp_source, validate

DEFAULT_DATASET_ID = "piimb/privy"
SOURCE_ID = "privy"


def _convert(row: dict) -> dict:
    spans = row.get("spans") or []
    mask: list[dict] = []
    for s in spans:
        label = s.get("entity_type")
        if not label:
            continue
        mask.append({
            "label": label,
            "value": s.get("entity_value", ""),
            "start": int(s["start_position"]),
            "end": int(s["end_position"]),
        })
    mask.sort(key=lambda x: x["start"])

    # Drop any overlapping spans (keep the earliest).
    deduped: list[dict] = []
    last_end = -1
    for m in mask:
        if m["start"] >= last_end:
            deduped.append(m)
            last_end = m["end"]

    return {
        "source_text": row["full_text"],
        "privacy_mask": deduped,
        "language": "en",
    }


def load(
    dataset_id: str = DEFAULT_DATASET_ID,
    *,
    limit: int | None = None,
    **_kwargs,
) -> DatasetDict:
    raw = load_raw(dataset_id, limit=limit)
    out: dict = {}
    for split, ds in raw.items():
        ds = ds.map(
            _convert,
            remove_columns=ds.column_names,
            desc=f"Converting privy/{split}",
        )
        out[split] = stamp_source(ds, SOURCE_ID)
    result = DatasetDict(out)
    validate(result, SOURCE_ID)
    return result
