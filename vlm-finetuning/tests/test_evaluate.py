"""Grounding scoring tests (pure functions — no torch/transformers needed)."""
from vlm_finetuning.evaluate import _count_matches, score_objects, score_text


def test_perfect_box_match():
    preds = [[{"label": "a", "box": [0.0, 0.0, 0.5, 0.5]}]]
    gold = [[{"label": "a", "box": [0.0, 0.0, 0.5, 0.5]}]]
    m = score_objects(preds, gold, task="box", iou_threshold=0.5)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
    assert m["f1_iou_avg"] == 1.0


def test_label_mismatch_is_not_a_match():
    preds = [[{"label": "a", "box": [0.0, 0.0, 0.5, 0.5]}]]
    gold = [[{"label": "b", "box": [0.0, 0.0, 0.5, 0.5]}]]
    m = score_objects(preds, gold, task="box")
    assert m["tp"] == 0 and m["fp"] == 1 and m["fn"] == 1


def test_low_iou_below_threshold_misses():
    preds = [[{"label": "a", "box": [0.0, 0.0, 0.30, 0.30]}]]
    gold = [[{"label": "a", "box": [0.0, 0.0, 1.00, 1.00]}]]
    # IoU = 0.09 — below 0.5
    m = score_objects(preds, gold, task="box", iou_threshold=0.5)
    assert m["tp"] == 0


def test_greedy_one_to_one_no_double_counting():
    # two predictions both overlap one gold box: only one can match
    preds = [[
        {"label": "a", "box": [0.0, 0.0, 0.5, 0.5]},
        {"label": "a", "box": [0.0, 0.0, 0.55, 0.55]},
    ]]
    gold = [[{"label": "a", "box": [0.0, 0.0, 0.5, 0.5]}]]
    m = score_objects(preds, gold, task="box", iou_threshold=0.5)
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 0


def test_point_in_box_match():
    preds = [[{"label": "a", "point": [0.25, 0.25]}]]
    gold = [[{"label": "a", "box": [0.0, 0.0, 0.5, 0.5]}]]
    m = score_objects(preds, gold, task="point")
    assert m["tp"] == 1 and m["f1"] == 1.0
    assert "f1_iou_avg" not in m  # only emitted for the box task


def test_point_outside_box_misses():
    preds = [[{"label": "a", "point": [0.9, 0.9]}]]
    gold = [[{"label": "a", "box": [0.0, 0.0, 0.5, 0.5]}]]
    assert score_objects(preds, gold, task="point")["tp"] == 0


def test_point_to_point_distance():
    preds = [[{"label": "a", "point": [0.50, 0.50]}]]
    gold = [[{"label": "a", "point": [0.52, 0.50]}]]   # dist 0.02 < 0.05
    assert score_objects(preds, gold, task="point")["tp"] == 1
    far = [[{"label": "a", "point": [0.70, 0.50]}]]    # dist 0.2 > 0.05 default
    assert score_objects(far, gold, task="point")["tp"] == 0


def test_per_label_breakdown():
    preds = [[
        {"label": "a", "box": [0.0, 0.0, 0.5, 0.5]},
        {"label": "b", "box": [0.5, 0.5, 1.0, 1.0]},
    ]]
    gold = [[
        {"label": "a", "box": [0.0, 0.0, 0.5, 0.5]},
        {"label": "b", "box": [0.0, 0.5, 0.4, 1.0]},   # wrong place
    ]]
    m = score_objects(preds, gold, task="box", labels=["a", "b"])
    assert m["f1_a"] == 1.0
    assert m["f1_b"] == 0.0


def test_count_matches_direct():
    preds = [{"label": "x", "box": [0, 0, 1, 1]}]
    gold = [{"label": "x", "box": [0, 0, 1, 1]}]
    assert _count_matches(preds, gold, task="box", iou_threshold=0.5, point_dist=0.05) == 1


# --- generic text metric --------------------------------------------------

def test_score_text_exact_match():
    m = score_text(['{"a":1}', "hello world"], ['{"a":1}', "hello world"])
    assert m["exact_match"] == 1.0 and m["token_f1"] == 1.0


def test_score_text_whitespace_normalized():
    m = score_text(["hello   world\n"], ["hello world"])
    assert m["exact_match"] == 1.0


def test_score_text_partial_token_f1():
    # half the tokens overlap -> exact_match 0, token_f1 between 0 and 1
    m = score_text(["a b c d"], ["a b x y"])
    assert m["exact_match"] == 0.0
    assert 0.0 < m["token_f1"] < 1.0


def test_score_text_no_overlap():
    m = score_text(["a b"], ["c d"])
    assert m["exact_match"] == 0.0 and m["token_f1"] == 0.0
