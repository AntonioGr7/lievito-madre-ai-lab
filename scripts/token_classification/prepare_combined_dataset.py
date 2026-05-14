#!/usr/bin/env python
"""Build the combined PII training corpus (OpenPII + Nemotron + Gretel + Privy).

Pipeline
--------
1. Each adapter (``encoder/token_classification/pii_corpus/<src>.py``) loads
   its raw dataset from the Hub and converts it to the shared intermediate
   schema: ``{source_text, privacy_mask, language, source}``.
2. Per-source DatasetDicts are concatenated by split (``train`` /
   ``validation`` / ``test``). Privy's ``dev`` is renamed to ``validation``;
   Nemotron carves a ~5% validation slice off its train shard.
3. Labels are NOT normalised across sources — OpenPII's ``GIVENNAME`` and
   Nemotron's ``first_name`` remain distinct classes in the BIO vocabulary
   (per project requirement: "keep their entity label separated as they are").
4. The combined raw DatasetDict is tokenised with mDeBERTa (or any HF model
   name) and saved Arrow-side as the standard
   ``{input_ids, attention_mask, labels}`` schema. The label vocabulary is
   embedded into the Arrow schema via ``Sequence(ClassLabel)``, so the
   training script picks it up automatically.

Usage
-----
# Full combined corpus
python scripts/token_classification/prepare_combined_dataset.py

# Subset of sources (e.g. drop Privy for a smoke test)
python scripts/token_classification/prepare_combined_dataset.py \\
    --sources openpii,nemotron,gretel

# Override tokenizer / output paths
python scripts/token_classification/prepare_combined_dataset.py \\
    --model xlm-roberta-base \\
    --out-dir data/processed/pii-combined-xlmr

# OpenPII English only (other sources are English already)
python scripts/token_classification/prepare_combined_dataset.py \\
    --openpii-languages en
"""

import argparse
import json
from pathlib import Path

from lievito_madre_ai_lab.encoder.token_classification.dataset import (
    build_label_names,
    collect_entity_types,
    preview_alignment,
    tokenize_for_trainer,
)
from lievito_madre_ai_lab.encoder.token_classification.pii_corpus import (
    ADAPTERS,
    combine,
)
from lievito_madre_ai_lab.encoder.token_classification.pii_corpus.combine import (
    _normalize_split_names,
    carve_validation,
)
from lievito_madre_ai_lab.encoder.token_classification.pii_corpus.schema import validate

DEFAULT_SOURCES = ["openpii", "nemotron", "gretel", "privy"]
DEFAULT_MODEL = "microsoft/mdeberta-v3-base"
DEFAULT_MAX_LEN = 512
DEFAULT_SLUG = "pii-combined"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help=f"Comma-separated source ids. Known: {sorted(ADAPTERS)}",
    )
    p.add_argument(
        "--openpii-languages",
        default=None,
        help="Comma-separated language codes for OpenPII only (e.g. 'en,it'). "
             "Default keeps all languages.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke-test: cap each split of each source to N rows after loading. "
             "Default: no cap.",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model name for the tokenizer")
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LEN, help="Tokenizer max_length")
    p.add_argument(
        "--no-label-all-tokens",
        dest="label_all_tokens",
        action="store_false",
        help="Only label the first subword of each entity (legacy). "
             "Default labels all subwords with I-.",
    )
    p.set_defaults(label_all_tokens=True)

    p.add_argument(
        "--raw-dir",
        default=f"data/raw/{DEFAULT_SLUG}",
        help="Where to save the combined raw Arrow snapshot",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Where to save the tokenized dataset "
             f"(default: data/processed/{DEFAULT_SLUG}-<model-slug>)",
    )
    p.add_argument(
        "--skip-tokenize",
        action="store_true",
        help="Stop after building/saving the combined raw corpus.",
    )
    return p.parse_args()


def _model_slug(model_name: str) -> str:
    return model_name.split("/")[-1]


def _load_with_filter(
    source_id: str,
    *,
    openpii_languages: list[str] | None,
    limit: int | None = None,
):
    """Run an adapter, then apply source-specific filters + split normalisation.

    ``limit`` is forwarded to the adapter, which uses HF streaming + take(N)
    so a smoke test does NOT trigger the full multi-GB Hub download.
    """
    loader = ADAPTERS[source_id]
    kwargs: dict = {"limit": limit}
    if source_id == "openpii" and openpii_languages:
        kwargs["languages"] = openpii_languages
    ds = loader(**kwargs)
    ds = _normalize_split_names(ds)
    ds = carve_validation(ds)
    validate(ds, source_id)
    return ds


def main() -> None:
    args = parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in sources if s not in ADAPTERS]
    if unknown:
        raise SystemExit(f"Unknown sources: {unknown}. Known: {sorted(ADAPTERS)}")

    openpii_langs = (
        [l.strip() for l in args.openpii_languages.split(",")]
        if args.openpii_languages else None
    )

    raw_dir = Path(args.raw_dir)
    out_dir = Path(
        args.out_dir or f"data/processed/{DEFAULT_SLUG}-{_model_slug(args.model)}"
    )

    # ------------------------------------------------------------------
    # 1. Load each source and combine
    # ------------------------------------------------------------------
    print(f"[1/4] Loading {len(sources)} source(s): {sources}")
    per_source: dict = {}
    for src in sources:
        print(f"  → {src}")
        per_source[src] = _load_with_filter(
            src, openpii_languages=openpii_langs, limit=args.limit,
        )
        for split, shard in per_source[src].items():
            print(f"      {split:10s}  {len(shard):>8d} rows")

    print("\n[1b] Concatenating splits across sources …")
    combined = combine(per_source)
    print(combined)

    raw_dir.mkdir(parents=True, exist_ok=True)
    combined.save_to_disk(str(raw_dir))
    print(f"      Combined raw snapshot saved → {raw_dir}")

    if args.skip_tokenize:
        print("Skipping tokenisation (per --skip-tokenize).")
        return

    # ------------------------------------------------------------------
    # 2. Build the dynamic entity vocabulary (UNION across sources)
    # ------------------------------------------------------------------
    print("\n[2/4] Scanning combined corpus for label vocabulary …")
    entity_types = collect_entity_types(combined)
    label_names = build_label_names(entity_types)
    print(f"      Distinct entity types: {len(entity_types)}")
    print(f"      Total BIO labels:      {len(label_names)} (O + 2×{len(entity_types)})")
    # Show the first chunk so it's easy to eyeball that nothing was normalised
    # accidentally — should see both UPPER and snake_case forms.
    preview = entity_types[:30]
    print(f"      Sample: {preview}{' …' if len(entity_types) > len(preview) else ''}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "entity_types.json").write_text(json.dumps(entity_types, indent=2))
    print(f"      Saved entity vocabulary → {raw_dir / 'entity_types.json'}")

    # ------------------------------------------------------------------
    # 3. Tokenise + align BIO labels with the combined vocabulary
    # ------------------------------------------------------------------
    print(f"\n[3/4] Tokenising with '{args.model}' (max_length={args.max_length}) …")
    processed = tokenize_for_trainer(
        combined,
        args.model,
        max_length=args.max_length,
        label_all_tokens=args.label_all_tokens,
        entity_types=entity_types,
        # Keep ``source`` so per-corpus evaluation (the 5k-per-dataset
        # leaderboard split) can be done without re-tokenising.
        keep_columns=["source"],
    )
    print(processed)

    preview_alignment(processed, args.model)

    # ------------------------------------------------------------------
    # 4. Save processed dataset
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    processed.save_to_disk(str(out_dir))
    print(f"[4/4] Processed dataset saved → {out_dir}")


if __name__ == "__main__":
    main()
