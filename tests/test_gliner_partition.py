"""Unit tests for the entity-type partitioning helper."""
import sys
import types

for mod in ("datasets", "transformers", "gliner"):
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)

import pytest

from lievito_madre_ai_lab.encoder.gliner_entity_extraction.dataset import (  # noqa: E402
    partition_entity_types,
)


def test_basic_partition():
    all_types = ["GIVENNAME", "SURNAME", "EMAIL", "PASSPORTNUM", "AGE"]
    holdout = ["PASSPORTNUM", "AGE"]
    train, held = partition_entity_types(all_types, holdout)
    assert sorted(train) == ["EMAIL", "GIVENNAME", "SURNAME"]
    assert sorted(held) == ["AGE", "PASSPORTNUM"]


def test_holdout_not_in_all_types_raises():
    all_types = ["GIVENNAME", "EMAIL"]
    holdout = ["NEVER_SEEN"]
    with pytest.raises(ValueError, match="not in the entity vocabulary"):
        partition_entity_types(all_types, holdout)


def test_empty_holdout():
    all_types = ["GIVENNAME", "EMAIL"]
    train, held = partition_entity_types(all_types, [])
    assert sorted(train) == ["EMAIL", "GIVENNAME"]
    assert held == []


def test_holdout_equals_all_raises():
    all_types = ["GIVENNAME", "EMAIL"]
    holdout = ["GIVENNAME", "EMAIL"]
    with pytest.raises(ValueError, match="empty train set"):
        partition_entity_types(all_types, holdout)
