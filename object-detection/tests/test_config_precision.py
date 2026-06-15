"""TrainConfig.precision + step math."""
import pytest

from object_detection.shared.config import TrainConfig, compute_total_training_steps


def test_precision_defaults_to_auto():
    assert TrainConfig().precision == "auto"


def test_precision_rejects_unknown():
    with pytest.raises(ValueError, match="precision must be one of"):
        TrainConfig(precision="bogus")


def test_lr_string_coerced():
    cfg = TrainConfig(learning_rate="1e-4")
    assert cfg.learning_rate == 1e-4 and isinstance(cfg.learning_rate, float)


def test_experiment_id_appends():
    cfg = TrainConfig(output_dir="outputs/run", experiment_id="r1")
    assert cfg.output_dir.endswith("run/r1") or cfg.output_dir.endswith("run\\r1")


def test_total_steps_epochs():
    cfg = TrainConfig(num_train_epochs=2, per_device_train_batch_size=2, gradient_accumulation_steps=1)
    assert compute_total_training_steps(100, cfg) == 100


def test_total_steps_max_steps_wins():
    assert compute_total_training_steps(10_000, TrainConfig(max_steps=5)) == 5
