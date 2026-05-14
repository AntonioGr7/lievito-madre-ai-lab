"""Adapter for gretelai/gretel-pii-masking-en-v1.

Source schema (from ``discover.py``)::

    text     : str
    entities : str    # JSON-ish: [{'entity': 'Alec', 'types': ['first_name']}, ...]
    + document metadata

Gretel does NOT publish character offsets — we recover them by
string-matching each entity ``value`` against ``text``, greedily claiming
non-overlapping occurrences. Entities that can't be located (date format
differences, free-form variants, etc.) are dropped silently. The retained
spans are sorted and non-overlapping, matching the intermediate-schema
contract used by the rest of the pipeline.

Intermediate output::

    source_text  : str
    privacy_mask : list[{label,value,start,end}]
    language     : "en"
    source       : "gretel"
"""
from __future__ import annotations

import ast

from datasets import DatasetDict

from .schema import load_raw, stamp_source, validate

DEFAULT_DATASET_ID = "gretelai/gretel-pii-masking-en-v1"
SOURCE_ID = "gretel"


def _parse_entities(raw) -> list[dict]:
    """Gretel stores entities as a string repr of a Python list. Be tolerant
    of plain lists too (in case the dataset gets republished with proper
    Sequence types)."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _claim_span(text: str, value: str, claimed: list[tuple[int, int]]
                ) -> tuple[int, int] | None:
    """Return (start, end) of the first occurrence of ``value`` in ``text`` that
    does not overlap any range in ``claimed``. ``None`` if no usable hit."""
    if not value:
        return None

    # Try the value as given first, then a stripped variant — Gretel
    # occasionally emits entity strings with leading/trailing whitespace
    # (e.g. " Zashil Tripathi") that don't appear verbatim in the text.
    candidates = [value]
    stripped = value.strip()
    if stripped and stripped != value:
        candidates.append(stripped)

    for candidate in candidates:
        offset = 0
        while True:
            idx = text.find(candidate, offset)
            if idx == -1:
                break
            end = idx + len(candidate)
            if not any(idx < ce and cs < end for cs, ce in claimed):
                return idx, end
            offset = idx + 1
    return None


def _convert(row: dict) -> dict:
    text = row["text"]
    entities = _parse_entities(row.get("entities"))

    mask: list[dict] = []
    claimed: list[tuple[int, int]] = []

    for ent in entities:
        value = ent.get("entity")
        types = ent.get("types") or []
        if not value or not types:
            continue
        # Keep the FIRST declared type — Gretel rarely lists multiples, and
        # creating duplicate spans would violate the non-overlap contract.
        label = types[0]

        hit = _claim_span(text, value, claimed)
        if hit is None:
            continue
        start, end = hit
        mask.append({"label": label, "value": text[start:end],
                     "start": start, "end": end})
        claimed.append((start, end))

    mask.sort(key=lambda x: x["start"])
    return {
        "source_text": text,
        "privacy_mask": mask,
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
            desc=f"Converting gretel/{split}",
        )
        out[split] = stamp_source(ds, SOURCE_ID)
    result = DatasetDict(out)
    validate(result, SOURCE_ID)
    return result
