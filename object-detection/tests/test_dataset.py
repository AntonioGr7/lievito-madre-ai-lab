"""COCO contract + geometry tests (no torch/transformers needed)."""
import pytest

from object_detection.dataset import coco_to_xyxy, validate_row, xyxy_to_coco


def test_coco_to_xyxy():
    assert coco_to_xyxy([10, 20, 30, 40]) == [10, 20, 40, 60]


def test_xyxy_to_coco_round_trip():
    assert xyxy_to_coco(coco_to_xyxy([10, 20, 30, 40])) == [10, 20, 30, 40]


def _row(**kw):
    base = {"image": "img", "image_id": 1,
            "objects": {"bbox": [[1.0, 2.0, 3.0, 4.0]], "category_id": [0]}}
    base.update(kw)
    return base


def test_validate_row_well_formed():
    assert validate_row(_row()) == []


def test_validate_row_empty_objects_ok():
    assert validate_row(_row(objects={"bbox": [], "category_id": []})) == []


def test_validate_row_missing_image():
    errs = validate_row({"image_id": 1, "objects": {"bbox": [], "category_id": []}})
    assert any("image" in e for e in errs)


def test_validate_row_image_id_not_int():
    errs = validate_row(_row(image_id="x"))
    assert any("image_id" in e for e in errs)


def test_validate_row_missing_objects():
    errs = validate_row({"image": "i", "image_id": 1})
    assert any("objects" in e for e in errs)


def test_validate_row_parallel_mismatch():
    errs = validate_row(_row(objects={"bbox": [[1, 2, 3, 4]], "category_id": [0, 1]}))
    assert any("parallel" in e for e in errs)


def test_validate_row_bbox_wrong_arity():
    errs = validate_row(_row(objects={"bbox": [[1, 2, 3]], "category_id": [0]}))
    assert any("4 numbers" in e for e in errs)


def test_validate_row_zero_area():
    errs = validate_row(_row(objects={"bbox": [[1, 2, 0, 4]], "category_id": [0]}))
    assert any("w>0 and h>0" in e for e in errs)


def test_validate_row_negative_origin():
    errs = validate_row(_row(objects={"bbox": [[-1, 2, 3, 4]], "category_id": [0]}))
    assert any("x>=0 and y>=0" in e for e in errs)


def test_validate_row_negative_category():
    errs = validate_row(_row(objects={"bbox": [[1, 2, 3, 4]], "category_id": [-1]}))
    assert any("non-negative int" in e for e in errs)
