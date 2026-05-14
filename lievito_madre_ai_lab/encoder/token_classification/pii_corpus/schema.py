"""Intermediate format shared by every PII source adapter.

Every adapter's ``load()`` must return a ``DatasetDict`` whose splits expose
these four columns exactly:

    source_text   : str        - raw, untouched text (NO masking applied)
    privacy_mask  : list[dict] - [{"label", "value", "start", "end"}, ...]
                                 sorted by ``start``; spans MUST NOT overlap
                                 within a single example
    language      : str        - BCP-47-ish code ("en", "it", "fr", ...);
                                 use "unk" when the source doesn't say
    source        : str        - short tag ("openpii", "nemotron", ...)
                                 so the combiner can stratify train/eval

This shape mirrors what ai4privacy publishes for OpenPII, so the existing
``tokenize_for_trainer`` works on the combined corpus without changes.
"""
from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset

INTERMEDIATE_COLUMNS = ("source_text", "privacy_mask", "language", "source")


def load_raw(dataset_id: str, *, limit: int | None = None) -> DatasetDict:
    """Load a HF Hub dataset, either fully or via streaming top-N per split.

    ``limit=None`` (default) uses the normal cached download — the right
    choice for real training runs.

    ``limit=N`` (smoke test) opens the dataset in streaming mode and
    materialises only the first ``N`` rows of each split into a regular
    in-memory ``Dataset``. The Hub archive is NOT fully downloaded — only
    enough shards are pulled to satisfy ``take(N)`` per split. The rest of
    the adapter pipeline (.map/.filter) then runs on this small in-memory
    Dataset exactly as it would on the full one.
    """
    if limit is None:
        return load_dataset(dataset_id)

    streamed = load_dataset(dataset_id, streaming=True)
    out: dict = {}
    for split, ds in streamed.items():
        rows = list(ds.take(limit))
        out[split] = Dataset.from_list(rows) if rows else Dataset.from_list([])
    return DatasetDict(out)


def stamp_source(ds: Dataset, source_id: str) -> Dataset:
    """Drop non-intermediate columns and add the ``source`` tag column.

    Convenience used by every adapter once it has produced the three
    content columns (``source_text``, ``privacy_mask``, ``language``).
    """
    keep = {"source_text", "privacy_mask", "language"}
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)
    return ds.add_column("source", [source_id] * len(ds))


def validate(ds: DatasetDict, source_id: str) -> None:
    """Fail fast if an adapter's output doesn't conform to the contract.

    Checks column presence, the shape of one non-empty ``privacy_mask`` per
    split, and that spans are sorted and non-overlapping.
    """
    required = set(INTERMEDIATE_COLUMNS)
    for split, d in ds.items():
        missing = required - set(d.column_names)
        if missing:
            raise ValueError(
                f"[{source_id}/{split}] missing required columns: {sorted(missing)}. "
                f"Got: {d.column_names}"
            )
        # Probe the first row that actually has spans
        for ex in d:
            mask = ex["privacy_mask"]
            if not mask:
                continue
            for span in mask:
                for k in ("label", "value", "start", "end"):
                    if k not in span:
                        raise ValueError(
                            f"[{source_id}/{split}] privacy_mask entry missing '{k}': {span}"
                        )
            # Sorted, non-overlapping
            prev_end = -1
            for span in mask:
                if span["start"] < prev_end:
                    raise ValueError(
                        f"[{source_id}/{split}] overlapping or unsorted spans in: {mask}"
                    )
                prev_end = span["end"]
            break  # one example is enough
