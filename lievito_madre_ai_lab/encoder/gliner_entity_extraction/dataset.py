"""Char-offset dataset contract for GLiNER fine-tuning.

Each row in train / validation / test splits must have the shape::

    {
      "text": str,                                       # raw text, untouched
      "spans": [                                         # may be empty
          {"start": int, "end": int, "label": str},      # half-open [start, end)
          ...
      ],
      # any extra columns are silently kept (ignored by the trainer)
    }

Prepare scripts (see `examples/`) call `validate_row` before saving to surface schema
bugs at prep time. `load_processed` reads the saved DatasetDict back, plus the
two type files written alongside it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_row(row: dict[str, Any]) -> list[str]:
    """Return a list of error strings (empty == OK) for *row*.

    Checks the row matches the char-offset contract. Extra columns are
    ignored; only `text` and `spans` are required.
    """
    errors: list[str] = []
    if "text" not in row:
        errors.append("row missing required key 'text'")
    elif not isinstance(row["text"], str):
        errors.append(f"row['text'] must be str, got {type(row['text']).__name__}")

    if "spans" not in row:
        errors.append("row missing required key 'spans'")
        return errors

    if not isinstance(row["spans"], list):
        errors.append(f"row['spans'] must be list, got {type(row['spans']).__name__}")
        return errors

    text_len = len(row["text"]) if isinstance(row.get("text"), str) else 0
    for i, span in enumerate(row["spans"]):
        if not isinstance(span, dict):
            errors.append(f"spans[{i}] must be dict, got {type(span).__name__}")
            continue
        for k in ("start", "end", "label"):
            if k not in span:
                errors.append(f"spans[{i}] missing key {k!r}")
        if any(k not in span for k in ("start", "end", "label")):
            continue
        s, e, lbl = span["start"], span["end"], span["label"]
        if not isinstance(s, int) or not isinstance(e, int):
            errors.append(f"spans[{i}] start/end must be int")
            continue
        if not isinstance(lbl, str) or not lbl:
            errors.append(f"spans[{i}] label must be non-empty str")
        if s < 0:
            errors.append(f"spans[{i}] requires start >= 0; got start={s}")
        if e <= s:
            errors.append(f"spans[{i}] requires end > start; got start={s} end={e}")
        if text_len and e > text_len:
            errors.append(
                f"spans[{i}] requires end <= len(text)={text_len}; got end={e}"
            )
    return errors


def partition_entity_types(
    all_types: list[str],
    holdout: list[str],
) -> tuple[list[str], list[str]]:
    """Split *all_types* into (train_types, holdout_types).

    The two lists are disjoint and their union equals *all_types*. Raises
    when a holdout label is not in *all_types* (typo guard) or when the
    holdout set leaves no training labels.
    """
    all_set = set(all_types)
    hold_set = set(holdout)

    missing = hold_set - all_set
    if missing:
        raise ValueError(
            f"holdout labels not in the entity vocabulary: {sorted(missing)}. "
            f"Known labels: {sorted(all_set)}"
        )

    train = [t for t in all_types if t not in hold_set]
    held = [t for t in all_types if t in hold_set]

    if not train:
        raise ValueError(
            "holdout set leaves an empty train set — pick a smaller holdout"
        )
    return train, held


def collect_entity_types(raw, span_col: str = "spans") -> list[str]:
    """Scan every split of *raw* and return the sorted set of distinct labels.

    Works on both the char-offset format (`spans` column with `label` keys)
    and any other column whose entries are lists of `{"label": str, ...}`
    dicts — pass `span_col` to override.
    """
    seen: set[str] = set()
    for split in raw.values():
        for row in split:
            for span in row[span_col]:
                lbl = span.get("label") if isinstance(span, dict) else None
                if lbl:
                    seen.add(lbl)
    return sorted(seen)


def to_gliner_native(row: dict, words_splitter) -> dict:
    """Convert a char-offset row → GLiNER's native training format.

    Input row::

        {"text": str, "spans": [{"start": int, "end": int, "label": str}, ...]}

    Output row (added columns; existing `text`/`spans` are preserved so the
    eval callback can still consume the char-offset form)::

        {
          ...input...,
          "tokenized_text": list[str],
          "ner": list[[start_tok, end_tok, label_str]],   # inclusive token indices
        }

    `words_splitter` is GLiNER's own splitter (`model.data_processor.words_splitter`),
    so there is no tokenization drift between training and inference.
    """
    text = row["text"]

    # WordsSplitter returns `(token_text, char_start, char_end)` triples in
    # gliner 0.2.x. Tolerate the simpler `(token_text, char_start, char_end)`
    # shape and the bare `(start, end)` shape just in case.
    words = list(words_splitter(text))
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for w in words:
        if isinstance(w, str):
            tokens.append(w)
            offsets.append((-1, -1))   # offsets unknown; spans below will skip
        elif len(w) == 3:
            tokens.append(w[0])
            offsets.append((int(w[1]), int(w[2])))
        elif len(w) == 2:
            tokens.append(str(w[0]))
            offsets.append((int(w[1]), -1))
        else:
            tokens.append(str(w[0]))
            offsets.append((-1, -1))

    ner: list[list] = []
    for span in row["spans"]:
        s_char, e_char = int(span["start"]), int(span["end"])
        start_tok: int | None = None
        end_tok: int | None = None
        for i, (ts, te) in enumerate(offsets):
            if ts < 0 or te < 0:
                # offset unknown; can't map this token
                continue
            # Overlap rule: token range and span range share at least one char.
            if te <= s_char or ts >= e_char:
                continue
            if start_tok is None:
                start_tok = i
            end_tok = i
        if start_tok is not None and end_tok is not None:
            ner.append([start_tok, end_tok, span["label"]])

    return {"tokenized_text": tokens, "ner": ner}


def load_processed(processed_dir: str | Path):
    """Load a processed char-offset DatasetDict + the two type files.

    Returns ``(datasets, train_types, holdout_types)``. Validates the first
    non-empty row of each split against `validate_row` and raises a useful
    error if the contract is violated.
    """
    from datasets import load_from_disk

    processed_dir = Path(processed_dir)
    datasets = load_from_disk(str(processed_dir))

    train_types_path = processed_dir / "train_types.json"
    if not train_types_path.exists():
        raise FileNotFoundError(
            f"missing {train_types_path}. Did the prepare script run to "
            f"completion?"
        )
    train_types: list[str] = json.loads(train_types_path.read_text())

    holdout_types_path = processed_dir / "holdout_types.json"
    holdout_types: list[str] = (
        json.loads(holdout_types_path.read_text())
        if holdout_types_path.exists() else []
    )

    for split_name, split in datasets.items():
        if len(split) == 0:
            continue
        errors = validate_row(split[0])
        if errors:
            raise ValueError(
                f"split {split_name!r} row 0 violates the char-offset "
                f"contract:\n  - " + "\n  - ".join(errors)
            )

    return datasets, train_types, holdout_types
