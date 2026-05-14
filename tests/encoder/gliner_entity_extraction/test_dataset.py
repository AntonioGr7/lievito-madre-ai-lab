"""Char-offset dataset contract tests."""
import sys
import types

for mod in ("gliner",):
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)

import pytest

from lievito_madre_ai_lab.encoder.gliner_entity_extraction.dataset import (  # noqa: E402
    validate_row,
    partition_entity_types,
    collect_entity_types,
)


def test_validate_row_accepts_well_formed():
    row = {"text": "Hello Maria", "spans": [{"start": 6, "end": 11, "label": "PERSON"}]}
    assert validate_row(row) == []


def test_validate_row_accepts_empty_spans():
    assert validate_row({"text": "no entities here", "spans": []}) == []


def test_validate_row_missing_text_key():
    errs = validate_row({"spans": []})
    assert any("text" in e for e in errs)


def test_validate_row_missing_spans_key():
    errs = validate_row({"text": "x"})
    assert any("spans" in e for e in errs)


def test_validate_row_text_not_string():
    errs = validate_row({"text": 42, "spans": []})
    assert any("text" in e and "str" in e for e in errs)


def test_validate_row_span_end_le_start():
    row = {"text": "abc", "spans": [{"start": 2, "end": 2, "label": "X"}]}
    errs = validate_row(row)
    assert any("end > start" in e for e in errs)


def test_validate_row_span_past_eot():
    row = {"text": "abc", "spans": [{"start": 0, "end": 100, "label": "X"}]}
    errs = validate_row(row)
    assert any("end <= len(text)" in e for e in errs)


def test_validate_row_span_negative_start():
    row = {"text": "abc", "spans": [{"start": -1, "end": 2, "label": "X"}]}
    errs = validate_row(row)
    assert any("start >= 0" in e for e in errs)


def test_validate_row_empty_label():
    row = {"text": "abc", "spans": [{"start": 0, "end": 1, "label": ""}]}
    errs = validate_row(row)
    assert any("label" in e for e in errs)


def test_validate_row_extra_columns_ignored():
    row = {"text": "x", "spans": [], "language": "en", "doc_id": 1}
    assert validate_row(row) == []


def test_load_processed_round_trip(tmp_path):
    """`load_processed` reads a DatasetDict we just wrote, returning rows + types."""
    from datasets import Dataset, DatasetDict
    from lievito_madre_ai_lab.encoder.gliner_entity_extraction.dataset import load_processed

    ds = DatasetDict({
        "train": Dataset.from_list([{"text": "Hi Maria", "spans": [{"start": 3, "end": 8, "label": "PERSON"}]}]),
        "validation": Dataset.from_list([{"text": "Bob", "spans": []}]),
        "test": Dataset.from_list([{"text": "Eve", "spans": [{"start": 0, "end": 3, "label": "PERSON"}]}]),
    })
    ds.save_to_disk(str(tmp_path))
    (tmp_path / "train_types.json").write_text('["PERSON"]')
    (tmp_path / "holdout_types.json").write_text('["AGE"]')

    loaded, train_types, holdout_types = load_processed(tmp_path)
    assert train_types == ["PERSON"]
    assert holdout_types == ["AGE"]
    assert loaded["train"][0]["text"] == "Hi Maria"


def test_load_processed_missing_train_types(tmp_path):
    from datasets import Dataset, DatasetDict
    from lievito_madre_ai_lab.encoder.gliner_entity_extraction.dataset import load_processed

    ds = DatasetDict({"train": Dataset.from_list([{"text": "x", "spans": []}])})
    ds.save_to_disk(str(tmp_path))
    with pytest.raises(FileNotFoundError, match="train_types.json"):
        load_processed(tmp_path)


def test_load_processed_invalid_row_raises(tmp_path):
    from datasets import Dataset, DatasetDict
    from lievito_madre_ai_lab.encoder.gliner_entity_extraction.dataset import load_processed

    bad = {"text": "x", "spans": [{"start": 5, "end": 2, "label": "X"}]}
    ds = DatasetDict({"train": Dataset.from_list([bad])})
    ds.save_to_disk(str(tmp_path))
    (tmp_path / "train_types.json").write_text('["X"]')
    with pytest.raises(ValueError, match="end > start"):
        load_processed(tmp_path)
