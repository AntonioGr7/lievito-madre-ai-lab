"""TrainConfig.precision field replaces fp16/bf16/tf32 triplet."""
from lievito_madre_ai_lab.shared.config import TrainConfig


def test_precision_defaults_to_auto():
    cfg = TrainConfig()
    assert cfg.precision == "auto"


def test_precision_accepts_explicit_values():
    for value in ("auto", "bf16", "fp16", "fp32"):
        cfg = TrainConfig(precision=value)
        assert cfg.precision == value


def test_precision_rejects_unknown_value():
    import pytest
    with pytest.raises(ValueError, match="precision must be one of"):
        TrainConfig(precision="bogus")


def test_old_fp16_field_removed():
    cfg = TrainConfig()
    assert not hasattr(cfg, "fp16")
    assert not hasattr(cfg, "bf16")
    assert not hasattr(cfg, "tf32")
