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


def _split_words_with_offsets(text: str, words_splitter) -> tuple[list[str], list[tuple[int, int]]]:
    """Run *words_splitter* over *text* and normalise its output into parallel
    ``(tokens, char_offsets)`` lists. Tolerates the 1/2/3-tuple shapes the
    different gliner releases have produced over time.
    """
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for w in words_splitter(text):
        if isinstance(w, str):
            tokens.append(w)
            offsets.append((-1, -1))
        elif len(w) == 3:
            tokens.append(w[0])
            offsets.append((int(w[1]), int(w[2])))
        elif len(w) == 2:
            tokens.append(str(w[0]))
            offsets.append((int(w[1]), -1))
        else:
            tokens.append(str(w[0]))
            offsets.append((-1, -1))
    return tokens, offsets


def _spans_for_window(
    spans: list[dict],
    offsets: list[tuple[int, int]],
    start_tok: int,
    end_tok: int,
) -> list[list]:
    """Compute the ``ner`` list for the word-token window ``[start_tok, end_tok)``.

    Spans entirely outside the window are dropped. Spans that overlap the
    window are kept with token indices remapped to the window-local frame.
    Spans that *straddle* the window boundary are clipped to the visible
    tokens — the same span may then appear (clipped) in the next overlapping
    chunk, which is the whole point of stride > 0.
    """
    out: list[list] = []
    window_offsets = offsets[start_tok:end_tok]
    for span in spans:
        s_char, e_char, label = int(span["start"]), int(span["end"]), span["label"]
        first_local: int | None = None
        last_local: int | None = None
        for i, (ts, te) in enumerate(window_offsets):
            if ts < 0 or te < 0:
                continue
            if te <= s_char or ts >= e_char:
                continue
            if first_local is None:
                first_local = i
            last_local = i
        if first_local is not None and last_local is not None:
            out.append([first_local, last_local, label])
    return out


def chunk_to_gliner_native(
    row: dict,
    words_splitter,
    *,
    max_words: int,
    stride: int,
) -> list[dict]:
    """Char-offset row → list of self-contained chunk rows.

    Long documents would otherwise be silently truncated to ``model.config.max_len``
    inside GLiNER's data processor, losing supervision on entities past the
    cap. We window the *word* sequence (matching the model's splitter, not
    subword) with overlap, so every entity stays fully visible in at least
    one chunk.

    Each emitted row is a **standalone document**: ``text`` is sliced to the
    chunk's char range and ``spans`` are clipped to that range with offsets
    rebased to chunk-local coordinates. This lets the same row be consumed
    by both the GLiNER trainer (via ``tokenized_text``/``ner``) and the eval
    callback (via ``text``/``spans``) without re-truncating long inputs.
    """
    if max_words <= 0:
        raise ValueError(f"max_words must be > 0, got {max_words}")
    if stride < 0 or stride >= max_words:
        raise ValueError(
            f"stride must satisfy 0 <= stride < max_words; "
            f"got stride={stride}, max_words={max_words}"
        )

    text = row["text"]
    tokens, offsets = _split_words_with_offsets(text, words_splitter)
    n = len(tokens)
    spans = row["spans"]

    if n <= max_words:
        ner = _spans_for_window(spans, offsets, 0, n)
        return [{**row, "tokenized_text": tokens, "ner": ner}]

    step = max_words - stride
    out: list[dict] = []
    start = 0
    while True:
        end = min(start + max_words, n)
        ner = _spans_for_window(spans, offsets, start, end)

        # Self-contained chunk: slice the text to the char range covered by
        # this window's words (keeping whitespace/punctuation between them)
        # and rebase span offsets to be chunk-local.
        valid = [(s, e) for (s, e) in offsets[start:end] if s >= 0 and e >= 0]
        if valid:
            chunk_char_start = valid[0][0]
            chunk_char_end = valid[-1][1]
        else:
            chunk_char_start, chunk_char_end = 0, 0
        chunk_text = text[chunk_char_start:chunk_char_end]
        chunk_spans: list[dict] = []
        for span in spans:
            s_char, e_char = int(span["start"]), int(span["end"])
            if e_char <= chunk_char_start or s_char >= chunk_char_end:
                continue
            # Clip to the chunk's range and rebase to chunk-local offsets.
            cs = max(s_char, chunk_char_start) - chunk_char_start
            ce = min(e_char, chunk_char_end) - chunk_char_start
            chunk_spans.append({"start": cs, "end": ce, "label": span["label"]})

        out.append({
            **row,
            "text": chunk_text,
            "spans": chunk_spans,
            "tokenized_text": tokens[start:end],
            "ner": ner,
        })
        if end >= n:
            break
        start += step
    return out


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
    tokens, offsets = _split_words_with_offsets(row["text"], words_splitter)
    ner = _spans_for_window(row["spans"], offsets, 0, len(tokens))
    return {"tokenized_text": tokens, "ner": ner}


def to_native_dataset(
    split,
    words_splitter,
    *,
    max_words: int | None = None,
    stride: int = -1,
    desc: str | None = None,
):
    """Convert a char-offset ``Dataset`` to the GLiNER native format.

    With ``stride < 0`` chunking is disabled — equivalent to calling
    ``to_gliner_native`` per row. With ``stride >= 0`` (and a positive
    ``max_words``), long documents are split into overlapping chunks via
    ``chunk_to_gliner_native``: each emitted row is a self-contained
    char-offset document (text/spans rebased to chunk-local offsets) that
    also carries the matching ``tokenized_text``/``ner``. The same row can
    therefore be consumed by the GLiNER trainer (native form) and by the
    eval callback (char-offset form) without re-truncating.

    Shared by ``build_trainer`` (which chunks all splits for training and
    chunk-level eval) and ``train_gliner.py`` (which chunks the test split
    the same way before computing final metrics).
    """
    chunking = stride >= 0
    if chunking:
        if max_words is None or max_words <= 0:
            raise ValueError(
                f"max_words must be positive when stride >= 0; got {max_words}"
            )
        if stride >= max_words:
            raise ValueError(
                f"stride={stride} must be < max_words={max_words}"
            )

    def _batch(batch: dict) -> dict:
        out: dict[str, list] = {k: [] for k in batch}
        out["tokenized_text"] = []
        out["ner"] = []
        for i in range(len(batch["text"])):
            row = {k: batch[k][i] for k in batch}
            if chunking:
                chunks = chunk_to_gliner_native(
                    row, words_splitter,
                    max_words=max_words, stride=stride,
                )
            else:
                native = to_gliner_native(row, words_splitter)
                chunks = [{**row, **native}]
            for chunk in chunks:
                for k in batch:
                    out[k].append(chunk[k])
                out["tokenized_text"].append(chunk["tokenized_text"])
                out["ner"].append(chunk["ner"])
        return out

    return split.map(
        _batch,
        batched=True,
        remove_columns=split.column_names,
        desc=desc or "Converting -> GLiNER native",
    )


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
