"""TrainConfig.precision field + scientific-notation coercion."""
import pytest

from vlm_finetuning.shared.config import TrainConfig, compute_total_training_steps


def test_precision_defaults_to_auto():
    assert TrainConfig().precision == "auto"


def test_precision_accepts_explicit_values():
    for value in ("auto", "bf16", "fp16", "fp32"):
        assert TrainConfig(precision=value).precision == value


def test_precision_rejects_unknown_value():
    with pytest.raises(ValueError, match="precision must be one of"):
        TrainConfig(precision="bogus")


def test_learning_rate_string_coerced_to_float():
    cfg = TrainConfig(learning_rate="1e-4")
    assert cfg.learning_rate == 1e-4 and isinstance(cfg.learning_rate, float)


def test_experiment_id_appends_to_output_dir():
    cfg = TrainConfig(output_dir="outputs/run", experiment_id="exp1")
    assert cfg.output_dir.endswith("run/exp1") or cfg.output_dir.endswith("run\\exp1")


def test_compute_total_training_steps_epochs():
    cfg = TrainConfig(
        num_train_epochs=3, per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
    )
    # 100 examples, effective batch 4 -> 25 steps/epoch * 3 = 75
    assert compute_total_training_steps(100, cfg) == 75


def test_compute_total_training_steps_max_steps_wins():
    cfg = TrainConfig(max_steps=10)
    assert compute_total_training_steps(100000, cfg) == 10
