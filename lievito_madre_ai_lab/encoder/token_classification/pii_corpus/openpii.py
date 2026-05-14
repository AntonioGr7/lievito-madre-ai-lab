"""Adapter for the ai4privacy OpenPII datasets (500k or 1M variant).

OpenPII already publishes ``source_text``, ``privacy_mask``, and ``language``
in the exact shape we want. The adapter just drops extra metadata columns
(masked_text, tokenized_text, …) and stamps a ``source`` tag.
"""
from __future__ import annotations

from datasets import DatasetDict

from .schema import load_raw, stamp_source

DEFAULT_DATASET_ID = "ai4privacy/pii-masking-openpii-1m"
SOURCE_ID = "openpii"


def load(
    dataset_id: str = DEFAULT_DATASET_ID,
    *,
    languages: list[str] | None = None,
    limit: int | None = None,
) -> DatasetDict:
    """Load OpenPII into the intermediate format.

    Parameters
    ----------
    dataset_id
        HF Hub id. Defaults to the 1M variant; pass
        ``"ai4privacy/open-pii-masking-500k-ai4privacy"`` to use the smaller
        original.
    languages
        Optional whitelist of language codes (e.g. ``["en", "it", "fr"]``).
        ``None`` keeps every language.
    limit
        Smoke-test cap: stream-and-take only N rows per split, avoiding the
        full archive download. Applied BEFORE language filtering, so a
        ``limit=100`` + ``languages=["en"]`` combo may return fewer than 100
        English rows.
    """
    raw = load_raw(dataset_id, limit=limit)

    out: dict = {}
    for split, ds in raw.items():
        if languages:
            allow = set(languages)
            ds = ds.filter(lambda r: r["language"] in allow,
                           desc=f"Filtering {split} by language")
        out[split] = stamp_source(ds, SOURCE_ID)
    return DatasetDict(out)
