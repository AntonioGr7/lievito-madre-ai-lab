"""GLiNERPredictor loads both full-FT and LoRA saves via the same constructor."""
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


for mod in ("gliner",):
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)


@pytest.fixture
def patch_gliner(monkeypatch):
    """GLiNER.from_pretrained returns a mock with a callable batch_predict_entities."""
    fake_module = types.ModuleType("gliner")

    class FakeGLiNER:
        @staticmethod
        def from_pretrained(name):
            m = MagicMock()
            m.config = SimpleNamespace(
                train_types=["PERSON"], holdout_types=[], label_aliases={},
                base_model_name_or_path=name,
            )
            m.model = MagicMock()
            m.batch_predict_entities = MagicMock(
                return_value=[[{"label": "PERSON", "text": "Eve",
                                "start": 0, "end": 3, "score": 0.9}]]
            )
            m.eval = MagicMock()
            return m

    fake_module.GLiNER = FakeGLiNER
    monkeypatch.setitem(sys.modules, "gliner", fake_module)
    return fake_module


def test_predictor_loads_full_ft_save(patch_gliner, tmp_path):
    from lievito_madre_ai_lab.encoder.gliner_entity_extraction.serve import GLiNERPredictor

    pred = GLiNERPredictor(tmp_path, use_compile=False, warmup_steps=0)
    out = pred.predict_one("Eve was here")
    assert out == [{"label": "PERSON", "text": "Eve", "start": 0, "end": 3, "score": 0.9}]


def test_predictor_loads_lora_save(patch_gliner, tmp_path, monkeypatch):
    """When adapter_config.json exists, load the base model + PeftModel wrap."""
    (tmp_path / "adapter_config.json").write_text("{}")

    peft_module = types.ModuleType("peft")
    sentinel_wrapped = MagicMock(name="lora_wrapped")
    sentinel_merged = MagicMock(name="merged")
    sentinel_wrapped.merge_and_unload = MagicMock(return_value=sentinel_merged)

    class FakePeftModel:
        @staticmethod
        def from_pretrained(base, path):
            return sentinel_wrapped

    peft_module.PeftModel = FakePeftModel
    monkeypatch.setitem(sys.modules, "peft", peft_module)

    from lievito_madre_ai_lab.encoder.gliner_entity_extraction.serve import GLiNERPredictor

    pred = GLiNERPredictor(tmp_path, use_compile=False, warmup_steps=0)
    # After merge, the predictor's underlying encoder is the merged module.
    assert pred._model.model is sentinel_merged


def test_predict_uses_batch_predict_entities(patch_gliner, tmp_path):
    """The new _forward path calls batch_predict_entities, not the per-text loop."""
    from lievito_madre_ai_lab.encoder.gliner_entity_extraction.serve import GLiNERPredictor

    pred = GLiNERPredictor(tmp_path, use_compile=False, warmup_steps=0)
    # Mock returns one empty span-list per input text (so the result-zipping below
    # doesn't IndexError). The test asserts batching behavior, not span content.
    pred._model.batch_predict_entities = MagicMock(side_effect=lambda texts, *a, **kw: [[] for _ in texts])
    pred.predict(["a", "b", "c"])
    # Single call covers all three texts (batched).
    assert pred._model.batch_predict_entities.call_count == 1
