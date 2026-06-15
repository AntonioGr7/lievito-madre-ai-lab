"""Serve-side helpers that don't need a loaded model."""
import pytest

from vlm_finetuning.serve import _denormalize, _to_pil, draw_objects


def test_denormalize_box_and_point():
    objs = [
        {"label": "a", "box": [0.1, 0.2, 0.5, 0.6]},
        {"label": "b", "point": [0.5, 0.25]},
    ]
    out = _denormalize(objs, width=100, height=200)
    assert out[0]["box_px"] == [10.0, 40.0, 50.0, 120.0]
    assert out[1]["point_px"] == [50.0, 50.0]
    # normalized coords are preserved alongside the pixel ones
    assert out[0]["box"] == [0.1, 0.2, 0.5, 0.6]


def test_to_pil_passthrough():
    Image = pytest.importorskip("PIL.Image")
    img = Image.new("RGB", (8, 8))
    assert _to_pil(img).size == (8, 8)


def test_draw_objects_returns_image():
    Image = pytest.importorskip("PIL.Image")
    img = Image.new("RGB", (64, 64))
    objs = [
        {"label": "a", "box": [0.1, 0.1, 0.5, 0.5]},
        {"label": "b", "point": [0.7, 0.7]},
    ]
    out = draw_objects(img, objs)
    assert out.size == (64, 64)
    # drawing must not mutate the input image object
    assert out is not img
