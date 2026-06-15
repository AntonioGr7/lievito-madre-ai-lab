"""Discriminative-LR param groups, no-decay rule, backbone freeze (needs torch)."""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from object_detection.model import (  # noqa: E402
    _freeze_backbone,
    _is_no_decay,
    build_param_groups,
    resolve_torch_dtype,
)


class _Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 4))  # backbone.0.{weight,bias}
        self.norm = nn.LayerNorm(4)                      # norm.{weight,bias}
        self.head = nn.Linear(4, 2)                      # head.{weight,bias}


@pytest.mark.parametrize("name,expected", [
    ("head.bias", True),
    ("model.norm.weight", True),
    ("encoder.layernorm.weight", True),
    ("backbone.bn1.weight", True),
    ("head.weight", False),
    ("backbone.0.weight", False),
])
def test_is_no_decay(name, expected):
    assert _is_no_decay(name) is expected


def test_param_groups_four_groups_and_lrs():
    m = _Detector()
    groups = build_param_groups(m, base_lr=1e-4, weight_decay=1e-4, backbone_lr_mult=0.1)
    assert len(groups) == 4
    # every backbone param sits in a group at the reduced LR
    bb_ids = {id(p) for n, p in m.named_parameters() if "backbone" in n}
    for g in groups:
        gids = {id(p) for p in g["params"]}
        if gids & bb_ids:
            assert g["lr"] == pytest.approx(1e-5)
        else:
            assert g["lr"] == pytest.approx(1e-4)
    # norm/bias groups carry no weight decay
    for g in groups:
        names = {id(p): n for n, p in m.named_parameters()}
        if all(_is_no_decay(names[id(p)]) for p in g["params"]):
            assert g["weight_decay"] == 0.0


def test_param_groups_skip_frozen():
    m = _Detector()
    _freeze_backbone(m)
    groups = build_param_groups(m, base_lr=1e-4, weight_decay=1e-4)
    flat = {id(p) for g in groups for p in g["params"]}
    for n, p in m.named_parameters():
        if "backbone" in n:
            assert id(p) not in flat   # frozen → excluded


def test_freeze_backbone_counts_and_flags():
    m = _Detector()
    n = _freeze_backbone(m)
    assert n == 2  # weight + bias
    assert all(not p.requires_grad for nm, p in m.named_parameters() if "backbone" in nm)
    assert all(p.requires_grad for nm, p in m.named_parameters() if "backbone" not in nm)


def test_resolve_torch_dtype():
    assert resolve_torch_dtype("fp32") == torch.float32
    assert resolve_torch_dtype("bf16") == torch.bfloat16
    assert resolve_torch_dtype("fp16") == torch.float16
