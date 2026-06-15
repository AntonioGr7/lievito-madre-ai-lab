"""LoRA target resolution + dtype mapping (needs torch for nn.Module trees)."""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from vlm_finetuning.model import (  # noqa: E402
    _is_vision_module,
    _resolve_lora_targets,
    resolve_torch_dtype,
)


class _FakeVLM(nn.Module):
    """Mimics a VLM: a vision tower with q_proj/v_proj and a language model
    with the standard projection names. Auto-resolution must pick the LM
    projections and leave the vision tower frozen."""

    def __init__(self):
        super().__init__()
        self.visual = nn.ModuleDict({
            "blocks": nn.ModuleDict({
                "attn": nn.ModuleDict({
                    "q_proj": nn.Linear(4, 4),
                    "v_proj": nn.Linear(4, 4),
                }),
            }),
        })
        self.language_model = nn.ModuleDict({
            "layers": nn.ModuleDict({
                "self_attn": nn.ModuleDict({
                    "q_proj": nn.Linear(4, 4),
                    "k_proj": nn.Linear(4, 4),
                    "v_proj": nn.Linear(4, 4),
                    "o_proj": nn.Linear(4, 4),
                }),
                "mlp": nn.ModuleDict({
                    "gate_proj": nn.Linear(4, 4),
                    "up_proj": nn.Linear(4, 4),
                    "down_proj": nn.Linear(4, 4),
                }),
            }),
        })
        self.lm_head = nn.Linear(4, 8)


def test_resolve_auto_targets_lm_only():
    targets = _resolve_lora_targets(_FakeVLM(), "auto")
    assert all("visual" not in t for t in targets)
    assert any("language_model" in t and t.endswith("q_proj") for t in targets)
    # all 7 LM projection types present, none from vision
    suffixes = {t.split(".")[-1] for t in targets}
    assert suffixes == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert len(targets) == 7


def test_resolve_explicit_passthrough():
    assert _resolve_lora_targets(_FakeVLM(), ["q_proj", "v_proj"]) == ["q_proj", "v_proj"]


def test_resolve_raises_when_nothing_matches():
    class Empty(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = nn.Linear(2, 2)

    with pytest.raises(ValueError, match="no language-model projection"):
        _resolve_lora_targets(Empty(), "auto")


@pytest.mark.parametrize("frag", ["visual", "model.vision_model.x", "a.vision_tower.b", "image_encoder.l"])
def test_is_vision_module_true(frag):
    assert _is_vision_module(frag)


def test_is_vision_module_false():
    assert not _is_vision_module("language_model.layers.0.self_attn.q_proj")


def test_resolve_torch_dtype_explicit():
    assert resolve_torch_dtype("fp32") == torch.float32
    assert resolve_torch_dtype("bf16") == torch.bfloat16
    assert resolve_torch_dtype("fp16") == torch.float16
