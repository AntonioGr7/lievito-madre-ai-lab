"""load_gliner stamps the right fields on model.config and wraps with LoRA."""
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_gliner(monkeypatch):
    """Patch `gliner.GLiNER.from_pretrained` to return a configurable mock."""
    fake_module = types.ModuleType("gliner")

    class FakeGLiNER:
        @staticmethod
        def from_pretrained(name):
            inst = MagicMock()
            inst.config = SimpleNamespace()
            # The encoder lives at model.model in real GLiNER; the mock provides it.
            inst.model = MagicMock()
            inst.data_processor = MagicMock()
            return inst

    fake_module.GLiNER = FakeGLiNER
    monkeypatch.setitem(sys.modules, "gliner", fake_module)
    return fake_module


def test_load_gliner_stamps_train_and_holdout_types(fake_gliner):
    from lievito_madre_ai_lab.encoder.gliner_entity_extraction.model import load_gliner

    model = load_gliner("any-model", train_types=["PERSON", "ORG"], holdout_types=["AGE"])
    assert list(model.config.train_types) == ["PERSON", "ORG"]
    assert list(model.config.holdout_types) == ["AGE"]


def test_load_gliner_stamps_loss_and_sampling(fake_gliner):
    from lievito_madre_ai_lab.encoder.gliner_entity_extraction.model import load_gliner

    model = load_gliner(
        "any-model",
        train_types=["X"],
        loss_cfg={"funcs": ["focal"], "focal_alpha": 0.75, "focal_gamma": 2.0},
        sampling_cfg={"max_types": 30, "max_neg_type_ratio": 1.0, "random_drop": 0.1},
    )
    assert model.config.loss_funcs == ["focal"]
    assert model.config.focal_alpha == 0.75
    assert model.config.focal_gamma == 2.0
    assert model.config.max_types == 30
    assert model.config.max_neg_type_ratio == 1.0
    assert model.config.random_drop == 0.1


def test_load_gliner_stamps_label_aliases(fake_gliner):
    from lievito_madre_ai_lab.encoder.gliner_entity_extraction.model import load_gliner

    aliases = {"GIVENNAME": "given name", "SURNAME": "surname"}
    model = load_gliner("any-model", train_types=["GIVENNAME"], label_aliases=aliases)
    assert dict(model.config.label_aliases) == aliases


def test_load_gliner_max_span_width(fake_gliner):
    from lievito_madre_ai_lab.encoder.gliner_entity_extraction.model import load_gliner

    model = load_gliner("any-model", train_types=["X"], max_span_width=20)
    assert model.config.max_width == 20


def test_load_gliner_empty_aliases_default(fake_gliner):
    from lievito_madre_ai_lab.encoder.gliner_entity_extraction.model import load_gliner

    model = load_gliner("any-model", train_types=["X"])
    assert dict(model.config.label_aliases) == {}


def test_load_gliner_lora_wraps_encoder(fake_gliner, monkeypatch):
    """When peft_cfg.enabled is True, model.model is replaced by a PeftModel."""
    from lievito_madre_ai_lab.encoder.gliner_entity_extraction.model import load_gliner, PeftConfig

    peft_module = types.ModuleType("peft")

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    sentinel = MagicMock(name="peft_wrapped")

    def fake_get_peft_model(module, cfg):
        return sentinel

    peft_module.LoraConfig = FakeLoraConfig
    peft_module.get_peft_model = fake_get_peft_model

    peft_utils = types.ModuleType("peft.utils")
    peft_utils_other = types.ModuleType("peft.utils.other")
    peft_utils_other.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {}
    peft_utils.other = peft_utils_other

    monkeypatch.setitem(sys.modules, "peft", peft_module)
    monkeypatch.setitem(sys.modules, "peft.utils", peft_utils)
    monkeypatch.setitem(sys.modules, "peft.utils.other", peft_utils_other)

    model = load_gliner(
        "any-model",
        train_types=["X"],
        peft_cfg=PeftConfig(enabled=True, r=8, alpha=16, dropout=0.0),
    )
    assert model.model is sentinel
    assert model.config.peft_enabled is True
    assert model.config.peft_target_modules == ["query_proj", "key_proj", "value_proj", "dense"]
