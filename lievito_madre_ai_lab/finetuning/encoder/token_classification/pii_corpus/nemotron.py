"""Adapter for nvidia/Nemotron-PII.

Source schema (from ``discover.py``)::

    text          : str
    spans         : str   # JSON string: [{start,end,text,label}, ...]
    text_tagged   : str   # ignored — same info as `spans` in inline form
    locale        : str   # us / etc — coarser than BCP-47; we keep "en"
    + document metadata

Intermediate output (see :mod:`.schema`)::

    source_text  : str
    privacy_mask : list[{label,value,start,end}]
    language     : str
    source       : "nemotron"
"""
from __future__ import annotations

import ast
import json

from datasets import DatasetDict

from .schema import load_raw, stamp_source, validate

DEFAULT_DATASET_ID = "nvidia/Nemotron-PII"
SOURCE_ID = "nemotron"


def _parse_spans(raw) -> list[dict]:
    """Nemotron publishes ``spans`` as a string. It's usually a Python repr
    (single quotes) but JSON has been observed in places — try both."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str):
        return []
    try:
        return json.loads(raw)
    except (ValueError, json.JSONDecodeError):
        pass
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    return parsed if isinstance(parsed, list) else []


def _convert(row: dict) -> dict:
    spans = _parse_spans(row.get("spans"))

    mask: list[dict] = []
    for s in spans:
        label = s.get("label")
        if not label:
            continue
        mask.append({
            "label": label,
            "value": s.get("text", ""),
            "start": int(s["start"]),
            "end": int(s["end"]),
        })
    mask.sort(key=lambda x: x["start"])

    # Drop pathological overlaps (shouldn't happen in Nemotron but be defensive).
    deduped: list[dict] = []
    last_end = -1
    for m in mask:
        if m["start"] >= last_end:
            deduped.append(m)
            last_end = m["end"]

    return {
        "source_text": row["text"],
        "privacy_mask": deduped,
        # Locale is "us"/"in"/… not BCP-47. The dataset is English-only;
        # collapse to "en" so the combiner can stratify by language.
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
            desc=f"Converting nemotron/{split}",
        )
        out[split] = stamp_source(ds, SOURCE_ID)
    result = DatasetDict(out)
    validate(result, SOURCE_ID)
    return result
