"""Unit tests for ``_align_labels``.

Runs in milliseconds without GPU. Validates the BIO-alignment contract for
every tokenizer family before any (expensive) training run.
"""
import sys
import types

# ``dataset`` imports from ``datasets`` and ``transformers`` at module top,
# but ``_align_labels`` itself is pure-Python. Stub the heavy deps so this
# test runs in any env (CI, laptop, dev container) without GPU libs.
for mod in ("datasets", "transformers"):
    if mod not in sys.modules:
        stub = types.ModuleType(mod)
        sys.modules[mod] = stub
sys.modules["datasets"].ClassLabel = lambda *a, **kw: None
sys.modules["datasets"].Sequence = lambda *a, **kw: None
sys.modules["datasets"].DatasetDict = type("DatasetDict", (), {})
sys.modules["transformers"].AutoTokenizer = type("AutoTokenizer", (), {})

from lievito_madre_ai_lab.encoder.token_classification.dataset import (  # noqa: E402
    LABEL2ID,
    _align_labels,
)

B_CITY = LABEL2ID["B-CITY"]
I_CITY = LABEL2ID["I-CITY"]
B_NAME = LABEL2ID["B-GIVENNAME"]
I_NAME = LABEL2ID["I-GIVENNAME"]
B_EMAIL = LABEL2ID["B-EMAIL"]
I_EMAIL = LABEL2ID["I-EMAIL"]
O = LABEL2ID["O"]


def run(mask, offsets, specials=None, label_all_tokens=True):
    if specials is None:
        specials = [0] * len(offsets)
    return _align_labels(mask, offsets, specials, label_all_tokens)


def test_sentencepiece_first_subword_gets_B():
    # "▁Napels" with leading "▁" absorbs the preceding space:
    #   text = "... Napels ...", entity "Napels" = (64, 70)
    #   "▁Na" offset = (63, 66)  ← char_start is the SPACE before "N"
    #   "pels" offset = (66, 70)
    mask = [{"label": "CITY", "value": "Napels", "start": 64, "end": 70}]
    offsets = [(63, 66), (66, 70)]
    assert run(mask, offsets) == [B_CITY, I_CITY]


def test_bpe_g_prefix_first_subword_gets_B():
    # RoBERTa/GPT "Ġworld" — same leading-space pattern as SentencePiece
    mask = [{"label": "CITY", "value": "world", "start": 6, "end": 11}]
    offsets = [(5, 11)]
    assert run(mask, offsets) == [B_CITY]


def test_wordpiece_unaffected():
    # mBERT "Do" + "##e" — no leading-space marker, char_start is clean
    mask = [{"label": "GIVENNAME", "value": "Doe", "start": 5, "end": 8}]
    offsets = [(5, 7), (7, 8)]
    assert run(mask, offsets) == [B_NAME, I_NAME]


def test_entity_at_text_start():
    mask = [{"label": "GIVENNAME", "value": "John", "start": 0, "end": 4}]
    offsets = [(0, 4)]
    assert run(mask, offsets) == [B_NAME]


def test_token_entirely_before_entity():
    mask = [{"label": "CITY", "value": "X", "start": 10, "end": 15}]
    offsets = [(5, 8)]
    assert run(mask, offsets) == [O]


def test_token_entirely_after_entity():
    mask = [{"label": "CITY", "value": "X", "start": 10, "end": 15}]
    offsets = [(20, 25)]
    assert run(mask, offsets) == [O]


def test_adjacent_entities_distinct_B():
    # Two back-to-back entities — second one must start a fresh B-
    mask = [
        {"label": "GIVENNAME", "value": "A", "start": 0, "end": 5},
        {"label": "CITY",      "value": "B", "start": 5, "end": 10},
    ]
    offsets = [(0, 5), (5, 10)]
    assert run(mask, offsets) == [B_NAME, B_CITY]


def test_multi_subword_email_spsm():
    # SentencePiece tokenization of " anto.grimaldi@gmail.com" entity (24, 48)
    #   "▁"     (23, 24)   ← whitespace-only token, no overlap → O
    #   "anto"  (24, 28)   ← first content subword → B-EMAIL
    #   "."     (28, 29)   → I-EMAIL
    #   "grim"  (29, 33)   → I-EMAIL
    #   "aldi"  (33, 37)   → I-EMAIL
    #   "@"     (37, 38)   → I-EMAIL
    #   "gmail" (38, 43)   → I-EMAIL
    #   "."     (43, 44)   → I-EMAIL
    #   "com"   (44, 48)   → I-EMAIL (NOTE: 48 == span end → inclusive overlap)
    mask = [{"label": "EMAIL", "value": "anto.grimaldi@gmail.com",
             "start": 24, "end": 48}]
    offsets = [(23, 24), (24, 28), (28, 29), (29, 33), (33, 37),
               (37, 38), (38, 43), (43, 44), (44, 48)]
    expected = [O, B_EMAIL, I_EMAIL, I_EMAIL, I_EMAIL,
                I_EMAIL, I_EMAIL, I_EMAIL, I_EMAIL]
    assert run(mask, offsets) == expected


def test_special_tokens_are_minus_100():
    mask = [{"label": "GIVENNAME", "value": "John", "start": 0, "end": 4}]
    offsets = [(0, 0), (0, 4), (0, 0)]
    specials = [1, 0, 1]
    assert run(mask, offsets, specials) == [-100, B_NAME, -100]


def test_unknown_label_filtered_out():
    # Labels outside ENTITY_TYPES must not corrupt alignment
    mask = [{"label": "MADE_UP", "value": "x", "start": 0, "end": 4}]
    offsets = [(0, 4)]
    assert run(mask, offsets) == [O]


def test_label_all_tokens_false_continuations_ignored():
    mask = [{"label": "GIVENNAME", "value": "Doe", "start": 5, "end": 8}]
    offsets = [(5, 7), (7, 8)]
    assert run(mask, offsets, label_all_tokens=False) == [B_NAME, -100]


def test_zero_width_non_special_token_does_not_match():
    # Defensive: a zero-width non-special token must not be claimed by a span
    # that happens to start at the same position.
    mask = [{"label": "CITY", "value": "X", "start": 5, "end": 10}]
    offsets = [(5, 5), (5, 10)]
    assert run(mask, offsets) == [O, B_CITY]
