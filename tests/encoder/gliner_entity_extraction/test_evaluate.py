"""Strict char-offset F1: tp = |pred ∩ gold|, fp = pred − gold, fn = gold − pred."""
import sys
import types

for mod in ("gliner",):
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)

from lievito_madre_ai_lab.finetuning.encoder.gliner_entity_extraction.evaluate import (  # noqa: E402
    score_predictions,
    _harmonic_mean,
    _has_label_gold,
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


# --- generalist (zero-shot-aware) monitoring helpers ----------------------

def test_harmonic_mean_penalises_lopsided():
    # A model that aces closed-set but collapses zero-shot must NOT win the
    # combined metric — harmonic mean drags it far below the arithmetic mean.
    assert _harmonic_mean(0.9, 0.9) == 0.9
    assert _harmonic_mean(0.95, 0.0) == 0.0
    assert _harmonic_mean(0.95, 0.10) < 0.20   # arithmetic mean would be 0.525


def test_has_label_gold():
    ds = [
        {"spans": [{"start": 0, "end": 3, "label": "first_name"},
                   {"start": 4, "end": 6, "label": "age"}]},
        {"spans": [{"start": 0, "end": 5, "label": "email"}]},
    ]
    assert _has_label_gold(ds, ["age"]) is True
    assert _has_label_gold(ds, ["swift_bic"]) is False
    assert _has_label_gold(ds, []) is False
