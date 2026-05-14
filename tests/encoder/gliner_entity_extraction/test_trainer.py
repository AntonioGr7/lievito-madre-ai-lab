"""Trainer factory: precision resolver, optimizer param groups, eval callback wiring."""
from unittest.mock import patch

import pytest


# trainer.py imports `from gliner.training import TrainingArguments` at module
# load time, so the real gliner package is required to run these tests. No
# stubbing — the gliner extras venv must be active.


def test_resolve_precision_auto_on_ampere():
    from lievito_madre_ai_lab.encoder.gliner_entity_extraction.trainer import _resolve_precision
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_capability", return_value=(8, 0)):
        assert _resolve_precision("auto") == (False, True, True)


def test_resolve_precision_auto_on_turing():
    from lievito_madre_ai_lab.encoder.gliner_entity_extraction.trainer import _resolve_precision
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_capability", return_value=(7, 5)):
        assert _resolve_precision("auto") == (True, False, True)


def test_resolve_precision_auto_on_cpu():
    from lievito_madre_ai_lab.encoder.gliner_entity_extraction.trainer import _resolve_precision
    with patch("torch.cuda.is_available", return_value=False):
        assert _resolve_precision("auto") == (False, False, False)


def test_resolve_precision_explicit_bf16():
    from lievito_madre_ai_lab.encoder.gliner_entity_extraction.trainer import _resolve_precision
    with patch("torch.cuda.is_available", return_value=True):
        assert _resolve_precision("bf16") == (False, True, True)


def test_resolve_precision_explicit_fp32():
    from lievito_madre_ai_lab.encoder.gliner_entity_extraction.trainer import _resolve_precision
    with patch("torch.cuda.is_available", return_value=True):
        assert _resolve_precision("fp32") == (False, False, False)


def test_resolve_precision_unknown_value():
    from lievito_madre_ai_lab.encoder.gliner_entity_extraction.trainer import _resolve_precision
    with pytest.raises(ValueError, match="unknown precision"):
        _resolve_precision("octal")


def test_gliner_train_cfg_carries_focal_fields():
    """Focal-loss fields default to gliner's "disabled" sentinel (-1 / 0)."""
    from lievito_madre_ai_lab.encoder.gliner_entity_extraction.trainer import GLiNERTrainCfg

    cfg = GLiNERTrainCfg()
    assert cfg.focal_loss_alpha == -1.0
    assert cfg.focal_loss_gamma == 0.0
    assert cfg.head_lr_multiplier == 5.0

    enabled = GLiNERTrainCfg(focal_loss_alpha=0.75, focal_loss_gamma=2.0)
    assert enabled.focal_loss_alpha == 0.75
    assert enabled.focal_loss_gamma == 2.0
