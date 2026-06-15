"""Serve helpers that don't need a loaded model."""
import pytest

from object_detection.serve import _to_pil, draw_detections


def test_to_pil_passthrough():
    Image = pytest.importorskip("PIL.Image")
    img = Image.new("RGB", (10, 8))
    assert _to_pil(img).size == (10, 8)


def test_draw_detections_returns_copy():
    Image = pytest.importorskip("PIL.Image")
    img = Image.new("RGB", (64, 64))
    dets = [
        {"label": "Mask", "score": 0.91, "box": [5, 5, 30, 30]},
        {"label": "Gloves", "score": 0.42, "box": [40, 40, 60, 60]},
    ]
    out = draw_detections(img, dets)
    assert out.size == (64, 64)
    assert out is not img
