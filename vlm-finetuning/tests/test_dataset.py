"""Grounding contract + coordinate-token tests (no torch/transformers needed)."""
import pytest

from vlm_finetuning.dataset import (
    build_target,
    collect_labels,
    dequantize,
    format_object,
    iou,
    parse_objects,
    point_in_box,
    quantize,
    serialize_objects,
    validate_row,
)


# --- coordinate tokens ----------------------------------------------------

@pytest.mark.parametrize("value,bins,expected", [
    (0.0, 1000, 0),
    (1.0, 1000, 999),
    (0.5, 1001, 500),
    (-0.3, 1000, 0),       # clamped
    (1.7, 1000, 999),      # clamped
])
def test_quantize(value, bins, expected):
    assert quantize(value, bins) == expected


def test_quantize_dequantize_round_trips_within_grid():
    for v in (0.0, 0.123, 0.5, 0.777, 1.0):
        q = quantize(v, 1000)
        assert abs(dequantize(q, 1000) - v) <= 1 / 999 + 1e-9


def test_format_object_box():
    obj = {"label": "Mask", "box": [0.1, 0.2, 0.8, 0.9]}
    s = format_object(obj, task="box", coord_bins=1000)
    # 0.1*999=99.9->100, 0.2->200, 0.8->799, 0.9->899
    assert s == "Mask<box> 100, 200, 799, 899 </box>"


def test_format_object_point_from_box_center():
    obj = {"label": "Mask", "box": [0.0, 0.0, 1.0, 1.0]}
    s = format_object(obj, task="point", coord_bins=1000)
    # centre of the full image -> 0.5*999=499.5 -> 500 (round half-to-even)
    assert s == "Mask<box> 500, 500 </box>"


def test_format_object_returns_none_without_coords():
    assert format_object({"label": "x"}, task="box") is None


def test_serialize_empty_uses_empty_text():
    assert serialize_objects([], task="box", empty_text="none") == "none"


def test_serialize_multiline():
    objs = [
        {"label": "a", "box": [0.0, 0.0, 0.5, 0.5]},
        {"label": "b", "box": [0.5, 0.5, 1.0, 1.0]},
    ]
    out = serialize_objects(objs, task="box")
    assert out.count("\n") == 1
    assert out.startswith("a<box>")


def test_parse_objects_box_and_point():
    text = "Coverall<box> 100, 200, 800, 900 </box>\nMask<box> 450, 550 </box>"
    objs = parse_objects(text, coord_bins=1000)
    assert len(objs) == 2
    assert objs[0]["label"] == "Coverall" and "box" in objs[0]
    assert objs[1]["label"] == "Mask" and "point" in objs[1]


def test_parse_round_trips_serialize():
    objs = [{"label": "Face Shield", "box": [0.12, 0.34, 0.88, 0.96]}]
    text = serialize_objects(objs, task="box", coord_bins=1000)
    parsed = parse_objects(text, coord_bins=1000)
    assert parsed[0]["label"] == "Face Shield"
    for a, b in zip(parsed[0]["box"], objs[0]["box"]):
        assert abs(a - b) <= 1 / 999 + 1e-6


def test_parse_tolerates_messy_text_and_bad_arity():
    text = "- red box <box>10,20,30,40</box> junk <box> 1, 2, 3 </box> ok<box>5,6</box>"
    objs = parse_objects(text)
    # the 3-number match is dropped; box + point survive
    assert [("box" in o) for o in objs] == [True, False]
    assert objs[0]["label"] == "red box"
    assert objs[1]["label"] == "ok" and "point" in objs[1]


def test_parse_empty_text_returns_empty():
    assert parse_objects("No objects detected.") == []


# --- geometry -------------------------------------------------------------

def test_iou_identical_is_one():
    assert iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0


def test_iou_disjoint_is_zero():
    assert iou([0, 0, 0.4, 0.4], [0.6, 0.6, 1.0, 1.0]) == 0.0


def test_iou_half_overlap():
    # two unit-ish boxes overlapping on half -> 1/3
    val = iou([0, 0, 2, 2], [1, 0, 3, 2])
    assert abs(val - (2 / 6)) < 1e-9


def test_point_in_box():
    assert point_in_box([0.5, 0.5], [0.0, 0.0, 1.0, 1.0])
    assert not point_in_box([1.5, 0.5], [0.0, 0.0, 1.0, 1.0])


# --- contract validation --------------------------------------------------

def _row(**kw):
    base = {"image": "img", "prompt": "find things", "objects": []}
    base.update(kw)
    return base


def test_validate_row_well_formed():
    assert validate_row(_row(objects=[{"label": "x", "box": [0.1, 0.1, 0.5, 0.5]}])) == []


def test_validate_row_point_only_ok():
    assert validate_row(_row(objects=[{"label": "x", "point": [0.5, 0.5]}])) == []


def test_validate_row_missing_image():
    errs = validate_row({"prompt": "p", "objects": []})
    assert any("image" in e for e in errs)


def test_validate_row_empty_prompt():
    errs = validate_row(_row(prompt="   "))
    assert any("prompt" in e for e in errs)


def test_validate_row_box_out_of_unit_range():
    errs = validate_row(_row(objects=[{"label": "x", "box": [0.1, 0.1, 1.5, 0.5]}]))
    assert any("normalized to [0,1]" in e for e in errs)


def test_validate_row_box_unordered():
    errs = validate_row(_row(objects=[{"label": "x", "box": [0.5, 0.5, 0.2, 0.2]}]))
    assert any("x2>x1" in e for e in errs)


def test_validate_row_object_without_coords():
    errs = validate_row(_row(objects=[{"label": "x"}]))
    assert any("at least one of 'box' or 'point'" in e for e in errs)


def test_validate_row_empty_label():
    errs = validate_row(_row(objects=[{"label": "", "box": [0.1, 0.1, 0.5, 0.5]}]))
    assert any("label" in e for e in errs)


def test_collect_labels():
    raw = {
        "train": [{"objects": [{"label": "b"}, {"label": "a"}]}],
        "test": [{"objects": [{"label": "c"}]}],
    }
    assert collect_labels(raw) == ["a", "b", "c"]


# --- generic text target + build_target ----------------------------------

def test_build_target_response_wins():
    row = {"response": "tool_call(foo=1)", "objects": [{"label": "x", "box": [0, 0, 1, 1]}]}
    assert build_target(row, task="box") == "tool_call(foo=1)"


def test_build_target_serializes_objects_when_no_response():
    row = {"objects": [{"label": "a", "box": [0.0, 0.0, 1.0, 1.0]}]}
    assert build_target(row, task="box") == "a<box> 0, 0, 999, 999 </box>"


def test_build_target_text_without_response_raises():
    with pytest.raises(ValueError, match="requires a 'response'"):
        build_target({"objects": []}, task="text")


def test_validate_row_response_target_ok():
    assert validate_row({"image": "img", "prompt": "p", "response": '{"k": 1}'}) == []


def test_validate_row_empty_response_rejected():
    errs = validate_row({"image": "img", "prompt": "p", "response": "   "})
    assert any("response" in e for e in errs)


def test_validate_row_no_target_rejected():
    errs = validate_row({"image": "img", "prompt": "p"})
    assert any("target" in e for e in errs)
