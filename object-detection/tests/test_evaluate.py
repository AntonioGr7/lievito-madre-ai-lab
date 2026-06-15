"""mAP summary flattening (pure — no torch/torchmetrics needed)."""
from object_detection.evaluate import summarize_map


def test_summarize_core_keys():
    result = {
        "map": 0.512, "map_50": 0.78, "map_75": 0.55,
        "map_small": 0.30, "map_medium": 0.50, "map_large": 0.62,
        "mar_1": 0.40, "mar_10": 0.60, "mar_100": 0.65,
    }
    out = summarize_map(result)
    assert out["map"] == 0.512 and out["map_50"] == 0.78
    assert set(out) == set(result)  # no per-class without id2label


def test_summarize_per_class_expansion():
    result = {
        "map": 0.5, "map_50": 0.7, "map_75": 0.5,
        "map_per_class": [0.6, 0.4],
        "classes": [0, 1],
    }
    out = summarize_map(result, {0: "Coverall", 1: "Mask"})
    assert out["map_Coverall"] == 0.6
    assert out["map_Mask"] == 0.4


def test_summarize_handles_scalar_single_class():
    # torchmetrics returns a scalar (not list) when exactly one class is present
    result = {"map": 0.5, "map_per_class": 0.6, "classes": 0}
    out = summarize_map(result, {0: "only"})
    assert out["map_only"] == 0.6


def test_summarize_ignores_missing_keys():
    out = summarize_map({"map": 0.1})
    assert out == {"map": 0.1}
