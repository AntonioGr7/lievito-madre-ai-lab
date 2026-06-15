"""Combine multiple PII source adapters into a single training DatasetDict.

Each adapter returns a DatasetDict with the intermediate schema
(``source_text``, ``privacy_mask``, ``language``, ``source``). This module
concatenates those per-source DatasetDicts into one, harmonising split
names across sources whose layouts differ:

    OpenPII   : train / validation
    Nemotron  : train / test          → carve val from train (see below)
    Gretel    : train / validation / test
    Privy     : train / test / dev    → "dev" renamed to "validation"

For sources missing a validation split, ``carve_validation`` slices off a
small fraction of train. The resulting combined DatasetDict has the three
standard splits (``train``, ``validation``, ``test``) where data exists.
"""
from __future__ import annotations

from typing import Iterable

from datasets import Dataset, DatasetDict, Features, Sequence, Value, concatenate_datasets

from . import gretel, nemotron, openpii, privy
from .schema import validate

# Canonical schema. Forced onto every shard before concatenation:
#  - OpenPII publishes an extra `label_index` field in privacy_mask records
#    and lists fields in a different order — incompatible with our adapter
#    output without normalisation.
#  - HF type inference on tiny streamed batches sometimes pegs `value` as
#    Json(decode=True) instead of string, which then fails to align.
# Casting through ``map(features=...)`` resolves both.
CANONICAL_FEATURES = Features({
    "source_text": Value("string"),
    "privacy_mask": Sequence({
        "label": Value("string"),
        "value": Value("string"),
        "start": Value("int64"),
        "end": Value("int64"),
    }),
    "language": Value("string"),
    "source": Value("string"),
})


def _canonicalise(shard: Dataset) -> Dataset:
    """Project a per-source shard onto ``CANONICAL_FEATURES``."""
    keep = set(CANONICAL_FEATURES)
    drop = [c for c in shard.column_names if c not in keep]
    if drop:
        shard = shard.remove_columns(drop)

    def _strip(batch: dict) -> dict:
        out_masks: list[list[dict]] = []
        for masks in batch["privacy_mask"]:
            normalised: list[dict] = []
            for m in (masks or []):
                normalised.append({
                    "label": str(m.get("label", "")),
                    "value": str(m.get("value", "")),
                    "start": int(m["start"]),
                    "end": int(m["end"]),
                })
            out_masks.append(normalised)
        return {
            "source_text": [str(x) for x in batch["source_text"]],
            "privacy_mask": out_masks,
            "language": [str(x) for x in batch["language"]],
            "source": [str(x) for x in batch["source"]],
        }

    return shard.map(
        _strip,
        batched=True,
        features=CANONICAL_FEATURES,
        remove_columns=shard.column_names,
        desc="Canonicalising features",
    )

# Canonical mapping: source-specific split names → unified split names.
# Anything not in this map keeps its original name.
SPLIT_ALIASES: dict[str, str] = {
    "dev": "validation",
    "valid": "validation",
}

# Adapter registry — keyed by the SOURCE_ID stamped onto each row.
ADAPTERS = {
    openpii.SOURCE_ID: openpii.load,
    nemotron.SOURCE_ID: nemotron.load,
    gretel.SOURCE_ID: gretel.load,
    privy.SOURCE_ID: privy.load,
}


def _normalize_split_names(ds: DatasetDict) -> DatasetDict:
    """Rename non-canonical splits (e.g. ``dev`` → ``validation``)."""
    renamed = {SPLIT_ALIASES.get(name, name): split for name, split in ds.items()}
    return DatasetDict(renamed)


def carve_validation(
    ds: DatasetDict,
    *,
    fraction: float = 0.05,
    seed: int = 42,
    min_size: int = 200,
) -> DatasetDict:
    """If ``ds`` has no validation split, slice one off the train split.

    Uses ``train_test_split`` on the train shard. No-op when ``validation``
    already exists or when train is too small to give up ``min_size`` rows.
    """
    if "validation" in ds or "train" not in ds:
        return ds
    n_train = len(ds["train"])
    n_val = max(min_size, int(round(n_train * fraction)))
    if n_train <= n_val * 2:
        # Not enough data to carve a useful validation split — leave alone.
        return ds
    split = ds["train"].train_test_split(test_size=n_val, seed=seed, shuffle=True)
    new = dict(ds)
    new["train"] = split["train"]
    new["validation"] = split["test"]
    return DatasetDict(new)


def load_sources(sources: Iterable[str]) -> dict[str, DatasetDict]:
    """Run each adapter and return a ``{source_id: DatasetDict}`` map.

    Splits are normalised (``dev`` → ``validation``) and any source missing a
    validation split has one carved off its train. Every output passes the
    intermediate-schema ``validate()`` check.
    """
    out: dict[str, DatasetDict] = {}
    for src in sources:
        if src not in ADAPTERS:
            raise ValueError(
                f"Unknown source {src!r}. Known: {sorted(ADAPTERS)}"
            )
        print(f"[combine] loading source: {src}")
        ds = ADAPTERS[src]()
        ds = _normalize_split_names(ds)
        ds = carve_validation(ds)
        validate(ds, src)
        for split, shard in ds.items():
            print(f"           {src}/{split:10s}  {len(shard):>8d} rows")
        out[src] = ds
    return out


def combine(per_source: dict[str, DatasetDict]) -> DatasetDict:
    """Concatenate per-source DatasetDicts on matching split names.

    Only the four intermediate columns survive; any extra columns produced by
    a future adapter would be aligned-or-dropped here.
    """
    grouped: dict[str, list[Dataset]] = {}
    for src, ds in per_source.items():
        for split, shard in ds.items():
            grouped.setdefault(split, []).append(_canonicalise(shard))

    return DatasetDict({
        split: concatenate_datasets(shards) for split, shards in grouped.items()
    })
