"""Strict char-offset F1: tp = |pred ∩ gold|, fp = pred − gold, fn = gold − pred."""
import sys
import types

for mod in ("gliner",):
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)

from lievito_madre_ai_lab.encoder.gliner_entity_extraction.evaluate import (  # noqa: E402
    score_predictions,
)


def test_perfect_match():
    gold = [[{"start": 0, "end": 4, "label": "PERSON"}]]
    pred = [[{"start": 0, "end": 4, "label": "PERSON"}]]
    m = score_predictions(pred, gold)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_one_tp_one_fp_one_fn():
    gold = [[{"start": 0, "end": 4, "label": "A"}, {"start": 5, "end": 8, "label": "B"}]]
    pred = [[{"start": 0, "end": 4, "label": "A"}, {"start": 9, "end": 12, "label": "C"}]]
    m = score_predictions(pred, gold)
    # TP=1 (A 0-4), FP=1 (C 9-12), FN=1 (B 5-8) → P=R=F1=0.5
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5
    assert m["tp"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 1


def test_label_mismatch_is_fp_and_fn():
    """Same offsets, different labels → both pred and gold count separately."""
    gold = [[{"start": 0, "end": 4, "label": "PERSON"}]]
    pred = [[{"start": 0, "end": 4, "label": "ORG"}]]
    m = score_predictions(pred, gold)
    assert m["tp"] == 0
    assert m["fp"] == 1
    assert m["fn"] == 1


def test_per_label_breakdown():
    gold = [[{"start": 0, "end": 4, "label": "A"}, {"start": 5, "end": 8, "label": "B"}]]
    pred = [[{"start": 0, "end": 4, "label": "A"}]]
    m = score_predictions(pred, gold, labels=["A", "B"])
    assert m["f1_A"] == 1.0
    assert m["f1_B"] == 0.0


def test_empty_pred_and_gold():
    m = score_predictions([[], []], [[], []])
    assert m["f1"] == 0.0
    assert m["tp"] == 0
    assert m["fp"] == 0
    assert m["fn"] == 0
