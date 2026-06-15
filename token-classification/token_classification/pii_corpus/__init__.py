"""PII corpus adapters and tooling.

Each adapter normalizes its source dataset into a DatasetDict that follows the
shared intermediate schema (see :mod:`.schema`). ``scripts/pii_corpus/`` then
combines these adapter outputs into one training-ready DatasetDict that drops
straight into the existing ``tokenize_for_trainer`` pipeline.
"""
from .schema import INTERMEDIATE_COLUMNS, stamp_source, validate
from .combine import ADAPTERS, combine, load_sources

__all__ = [
    "INTERMEDIATE_COLUMNS",
    "stamp_source",
    "validate",
    "ADAPTERS",
    "load_sources",
    "combine",
]
