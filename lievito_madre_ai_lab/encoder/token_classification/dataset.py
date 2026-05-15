from __future__ import annotations

from datasets import ClassLabel, DatasetDict, Sequence
from transformers import AutoTokenizer

# Re-exported from shared/preprocessing.py so existing imports keep working.
# The helpers themselves are task-agnostic — see that module for the format.
from lievito_madre_ai_lab.shared.preprocessing import (  # noqa: F401
    PREPROCESSING_META_FILE,
    load_preprocessing_meta,
    save_preprocessing_meta,
)

# Default vocabulary — matches ai4privacy/open-pii-masking-500k-ai4privacy.
# When combining sources (Nemotron, Gretel, Privy, …) build the vocabulary
# dynamically with ``collect_entity_types(raw)`` and pass it through.
ENTITY_TYPES: list[str] = [
    "GIVENNAME", "SURNAME", "TITLE",
    "EMAIL", "TELEPHONENUM",
    "IDCARDNUM", "PASSPORTNUM", "DRIVERLICENSENUM", "TAXNUM", "SOCIALNUM",
    "CITY", "STREET", "BUILDINGNUM", "ZIPCODE",
    "DATE", "TIME", "AGE",
    "SEX", "ORGANISATIONPLACEHOLDER",
]


def build_label_names(entity_types: list[str]) -> list[str]:
    """Build BIO label names from an entity-type list: [O, B-…, I-…]."""
    return (
        ["O"]
        + [f"B-{e}" for e in entity_types]
        + [f"I-{e}" for e in entity_types]
    )


def collect_entity_types(raw: DatasetDict, mask_col: str = "privacy_mask") -> list[str]:
    """Scan every split of ``raw`` and return the sorted set of label strings.

    Used by the combined-corpus prep flow: each source adapter keeps its own
    raw label names (``GIVENNAME``, ``first_name``, ``EMAIL_ADDRESS``, …) and
    the union becomes the model's output vocabulary.
    """
    seen: set[str] = set()
    for split in raw.values():
        for row in split:
            for span in row[mask_col]:
                label = span.get("label")
                if label:
                    seen.add(label)
    return sorted(seen)


LABEL_NAMES: list[str] = build_label_names(ENTITY_TYPES)
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABEL_NAMES)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}


def tokenize_for_trainer(
    raw: DatasetDict,
    model_name: str,
    text_col: str = "source_text",
    mask_col: str = "privacy_mask",
    max_length: int = 512,
    label_all_tokens: bool = True,
    entity_types: list[str] | None = None,
    keep_columns: list[str] | None = None,
    stride: int | None = 128,
) -> DatasetDict:
    """Tokenize a token-classification DatasetDict and return it ready for Trainer.

    - Tokenises `text_col` with return_offsets_mapping=True
    - Documents longer than ``max_length`` are split into overlapping chunks
      via ``return_overflowing_tokens`` so no tokens (or entity spans) are
      silently dropped. Each chunk becomes its own row; ``stride`` controls
      the overlap (set ``stride=None`` to fall back to plain truncation).
    - Aligns BIO labels from `mask_col` character spans to subword tokens:
        first subword of each entity span → B- label
        continuation subwords (default, label_all_tokens=True) → I- label
        continuation subwords (label_all_tokens=False) → -100 (ignored in loss)
      Default is True so the model gets training signal on every subword of an
      entity, producing cleaner BIO sequences at inference time.
    - Drops source columns (keep_columns lets callers preserve metadata such
      as ``source`` for per-corpus evaluation, fanned out per chunk),
      keeps input_ids, attention_mask, labels
    - Casts labels to Sequence(ClassLabel) so names survive save_to_disk()

    Parameters
    ----------
    entity_types
        Label vocabulary (without BIO prefixes). When ``None`` falls back to
        ``ENTITY_TYPES`` (the OpenPII set). For combined corpora compute it
        once with ``collect_entity_types(raw)`` and pass it in.
    keep_columns
        Source columns to preserve on the output (e.g. ``["source", "language",
        "uid"]``). Useful for per-corpus evaluation, and — combined with a
        unique row id — for aggregating overflow chunks back to their source
        document at eval time. Defaults to dropping everything except the
        tokenizer outputs + labels.
    stride
        Number of overlapping tokens between consecutive chunks of a long
        document. Default 128. Pass ``None`` to disable chunking (legacy
        truncate-only behaviour); pass 0 to chunk with no overlap.

    Expected privacy_mask format per row::

        [{"label": "GIVENNAME", "value": "John", "start": 0, "end": 4}, ...]
    """
    entity_types = entity_types if entity_types is not None else ENTITY_TYPES
    label_names = build_label_names(entity_types)
    label2id = {label: i for i, label in enumerate(label_names)}
    valid_labels = frozenset(entity_types)
    chunking = stride is not None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    keep = list(keep_columns or [])

    def _tokenize(batch: dict) -> dict:
        tok_kwargs = dict(
            truncation=True,
            max_length=max_length,
            padding=False,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        if chunking:
            tok_kwargs["stride"] = stride
            tok_kwargs["return_overflowing_tokens"] = True

        enc = tokenizer(batch[text_col], **tok_kwargs)

        # Map each output chunk back to its source row within this batch so we
        # can pull the right span list (and replicate any kept columns).
        if chunking:
            sample_map = enc.pop("overflow_to_sample_mapping")
        else:
            sample_map = list(range(len(batch[text_col])))

        enc["labels"] = [
            _align_labels(
                batch[mask_col][src_idx],
                offsets, special_mask,
                label_all_tokens, valid_labels, label2id,
            )
            for src_idx, offsets, special_mask in zip(
                sample_map, enc["offset_mapping"], enc["special_tokens_mask"],
            )
        ]
        del enc["offset_mapping"]
        del enc["special_tokens_mask"]

        # Fan out kept source columns per chunk.
        for col in keep:
            if col in batch:
                enc[col] = [batch[col][src_idx] for src_idx in sample_map]

        return enc

    # With overflow, output row count differs from input — every source column
    # must be dropped (and any kept ones are re-emitted by _tokenize above).
    cols_to_remove = list(raw["train"].column_names)
    tokenized = raw.map(
        _tokenize,
        batched=True,
        remove_columns=cols_to_remove,
        desc="Tokenising",
    )

    # Cast labels to ClassLabel so names are embedded in the Arrow schema
    tokenized = tokenized.cast_column(
        "labels", Sequence(ClassLabel(names=label_names))
    )
    return tokenized


def preview_alignment(
    tokenized: DatasetDict,
    model_name: str,
    *,
    split: str = "train",
    n_examples: int = 2,
    max_tokens: int = 80,
) -> None:
    """Print token→textual-label pairs for a few examples — a one-shot sanity
    check that surfaces BIO-alignment bugs at dataset-prep time.

    A healthy run shows the first subword of every entity carrying a ``B-``
    label. If you instead see ``▁Word/O`` immediately followed by
    ``subword/B-…``, the tokenizer offsets and the alignment logic have
    drifted apart (typically a leading-space marker like SentencePiece "▁"
    or BPE "Ġ" not being handled by ``_align_labels``).
    """
    feature = tokenized[split].features.get("labels")
    if not (isinstance(feature, Sequence) and isinstance(feature.feature, ClassLabel)):
        print(f"[preview] '{split}' labels are not ClassLabel; skipping alignment preview.")
        return
    label_names = feature.feature.names
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"\n[preview] First {n_examples} '{split}' example(s) — token / label")
    print("[preview] If a non-O entity's FIRST subword shows 'O', the alignment is broken.\n")

    shown = 0
    for ex in tokenized[split]:
        if shown >= n_examples:
            break
        has_entity = any(
            l != -100 and label_names[l] != "O" for l in ex["labels"]
        )
        if not has_entity:
            continue  # uninteresting — find an example with entities

        tokens = tokenizer.convert_ids_to_tokens(ex["input_ids"])
        print(f"  ── example {shown + 1}/{n_examples} ──")
        for i, (tok, label_id) in enumerate(zip(tokens, ex["labels"])):
            if i >= max_tokens:
                print(f"  … ({len(tokens) - max_tokens} more tokens truncated)")
                break
            label = "_ign_" if label_id == -100 else label_names[label_id]
            marker = " ◀" if label not in ("O", "_ign_") else ""
            print(f"    {tok!r:24s}  {label}{marker}")
        print()
        shown += 1

    if shown == 0:
        print(f"[preview] No examples with non-O entities found in '{split}'.\n")


def _align_labels(
    privacy_mask: list[dict],
    offset_mapping: list[tuple[int, int]],
    special_tokens_mask: list[int],
    label_all_tokens: bool,
    valid_labels: frozenset[str],
    label2id: dict[str, int],
) -> list[int]:
    # Build span list sorted by start position for early-exit scanning
    spans: list[tuple[int, int, str]] = sorted(
        (
            (int(e["start"]), int(e["end"]), e["label"])
            for e in privacy_mask
            if e.get("label") in valid_labels
        ),
        key=lambda x: x[0],
    )

    o_id = label2id["O"]
    labels: list[int] = []
    active_span: tuple[int, int, str] | None = None

    for (char_start, char_end), is_special in zip(offset_mapping, special_tokens_mask):
        if is_special:
            labels.append(-100)
            active_span = None
            continue

        # Match a token to a span by RANGE OVERLAP, not by "token-start lies
        # inside span". The old "start-inside" rule silently broke SentencePiece
        # (mDeBERTa, XLM-R) and BPE-Ġ (RoBERTa, GPT-style) tokenizers: their
        # leading "▁"/"Ġ" subword absorbs the preceding space, so char_start
        # falls on the space *before* the entity begins and the first subword
        # got labeled "O" instead of "B-…". WordPiece is unaffected because
        # there is no leading-space marker.
        matched: tuple[int, int, str] | None = None
        for span in spans:
            if span[1] <= char_start:   # span ends at/before token starts — skip
                continue
            if span[0] >= char_end:     # span starts at/after token ends — stop (sorted)
                break
            matched = span              # token range overlaps span
            break

        if matched is None:
            labels.append(o_id)
            active_span = None
        elif matched != active_span:
            # First token of this entity span → B-
            labels.append(label2id[f"B-{matched[2]}"])
            active_span = matched
        else:
            # Continuation subword of the same span → I- or ignored
            labels.append(
                label2id[f"I-{matched[2]}"] if label_all_tokens else -100
            )

    return labels
