#!/usr/bin/env python
"""Per-recipe smoke-test assertions for the bi-encoder pipeline.

Each recipe gets a "did the saved model do what the recipe promised"
check on top of the shared "paraphrase outscores unrelated" sanity test.
Plumbing-level assertions only — with 100 synthetic rows + 1 epoch we
test that the pipeline runs end-to-end, NOT that quality matches a real
training run.

Usage
-----
python examples/smoke/verify_smoke.py \\
    --recipe 4 \\
    --model-dir outputs/smoke_bi_encoder_r4/final \\
    --processed-dir data/processed/smoke-bi-encoder
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ANCHOR = "How do vector databases work?"
PARAPHRASE = "What is a vector database?"
UNRELATED = "Mount Everest is the tallest mountain on Earth."


def _paraphrase_beats_unrelated(predictor, *, prompt_name: str | None = None,
                                truncate_dim: int | None = None) -> tuple[float, float]:
    """Returns (sim_paraphrase, sim_unrelated). Caller decides what's a pass."""
    sims = predictor.similarity(
        [ANCHOR], [ANCHOR, PARAPHRASE, UNRELATED],
        a_prompt_name=prompt_name, b_prompt_name=prompt_name,
        truncate_dim=truncate_dim,
    )
    return float(sims[0, 1]), float(sims[0, 2])


def _require(cond: bool, msg: str) -> None:
    if not cond:
        sys.exit(f"SMOKE TEST FAILED — {msg}")


def verify_r1_r2_r6(model_dir: str) -> None:
    from bi_encoder.serve import BiEncoderPredictor
    predictor = BiEncoderPredictor(model_dir, use_compile=False, warmup_steps=0)
    sim_p, sim_u = _paraphrase_beats_unrelated(predictor)
    print(f"  sim(anchor, paraphrase) = {sim_p:.4f}")
    print(f"  sim(anchor, unrelated)  = {sim_u:.4f}")
    _require(
        sim_p > sim_u,
        f"paraphrase should outscore unrelated. Got paraphrase={sim_p:.4f} "
        f"<= unrelated={sim_u:.4f}. Pipeline ran but the model did not learn "
        f"the task on synthetic data — check tokenizer / loss config drift.",
    )
    print("  PASS — paraphrase outscored unrelated.")


def verify_r3_r7(model_dir: str, processed_dir: str) -> None:
    """Distill recipes: verify the `label` column was on the input, and that
    the saved model still produces sensible similarity ordering."""
    from datasets import load_from_disk
    ds = load_from_disk(processed_dir)
    train = ds["train"]
    _require(
        "label" in train.column_names,
        f"distill recipe input must have a `label` column (teacher scores). "
        f"Got columns: {train.column_names}. Did score_with_cross_encoder.py "
        f"run successfully?",
    )
    first_label = train[0]["label"]
    _require(
        isinstance(first_label, list) and len(first_label) > 0,
        f"`label` column must be list[float]; got {type(first_label).__name__} "
        f"with first value {first_label!r}.",
    )
    print(f"  distill `label` column present (first row len = {len(first_label)}).")
    verify_r1_r2_r6(model_dir)


def verify_r4(model_dir: str) -> None:
    """Matryoshka 1D: encode at truncate_dim=64 returns (N, 64); paraphrase
    still outscores unrelated at full dim."""
    from bi_encoder.serve import BiEncoderPredictor
    predictor = BiEncoderPredictor(model_dir, use_compile=False, warmup_steps=0,
                                    default_truncate_dim=None)
    emb_full = predictor.encode([ANCHOR, PARAPHRASE, UNRELATED])
    emb_64 = predictor.encode([ANCHOR, PARAPHRASE, UNRELATED], truncate_dim=64)
    print(f"  full-dim shape    : {tuple(emb_full.shape)}")
    print(f"  truncated-dim=64  : {tuple(emb_64.shape)}")
    _require(
        emb_64.shape == (3, 64),
        f"truncate_dim=64 should yield shape (3, 64); got {tuple(emb_64.shape)}.",
    )
    # Renormalisation check: after truncation+renorm, rows are unit-norm.
    norms = emb_64.norm(dim=-1)
    _require(
        ((norms - 1.0).abs() < 1e-3).all().item(),
        f"truncate_dim=64 rows should be L2-normalised; got norms {norms.tolist()}.",
    )
    sim_p, sim_u = _paraphrase_beats_unrelated(predictor)
    print(f"  full-dim sim(p) = {sim_p:.4f} | sim(u) = {sim_u:.4f}")
    _require(
        sim_p > sim_u,
        f"full-dim paraphrase should outscore unrelated. Got {sim_p:.4f} <= {sim_u:.4f}.",
    )
    print("  PASS — matryoshka 1D truncation works and full-dim ordering holds.")


def verify_r5(model_dir: str) -> None:
    """Matryoshka 2D: predictor loads with truncate_layers=3 and produces
    non-empty embeddings. Quality at 3/6 layers after 1 epoch on synthetic
    data is not asserted — plumbing only."""
    from bi_encoder.serve import BiEncoderPredictor
    # Full model first — quality sanity at full depth + dim.
    full = BiEncoderPredictor(model_dir, use_compile=False, warmup_steps=0,
                               default_truncate_dim=None)
    sim_p, sim_u = _paraphrase_beats_unrelated(full)
    print(f"  full-depth sim(p) = {sim_p:.4f} | sim(u) = {sim_u:.4f}")
    _require(
        sim_p > sim_u,
        f"full-depth paraphrase should outscore unrelated. Got {sim_p:.4f} <= {sim_u:.4f}.",
    )

    # Now layer-truncated load — the Matryoshka2d-specific bit.
    edge = BiEncoderPredictor(
        model_dir, use_compile=False, warmup_steps=0,
        truncate_layers=3,           # keep first 3 of 6 layers
        default_truncate_dim=128,
    )
    emb = edge.encode([ANCHOR, PARAPHRASE, UNRELATED])
    print(f"  truncated (3 layers, 128-dim) shape: {tuple(emb.shape)}")
    _require(
        emb.shape == (3, 128),
        f"truncated predictor should yield shape (3, 128); got {tuple(emb.shape)}.",
    )
    print("  PASS — matryoshka 2D layer truncation loads and encodes.")


def verify_r8(model_dir: str) -> None:
    """Prompts: saved model exposes named prompts; encode(prompt_name="query") works."""
    from bi_encoder.serve import BiEncoderPredictor
    predictor = BiEncoderPredictor(model_dir, use_compile=False, warmup_steps=0)
    print(f"  prompts on model: {predictor.prompts}")
    _require(
        "query" in predictor.prompts,
        f"saved model should carry an inference 'query' prompt; "
        f"got {predictor.prompts}.",
    )
    _require(
        "document" in predictor.prompts,
        f"saved model should carry an inference 'document' prompt; "
        f"got {predictor.prompts}.",
    )
    # Encode with the named prompt and confirm shapes match.
    emb_q = predictor.encode([ANCHOR], prompt_name="query")
    emb_d = predictor.encode([PARAPHRASE, UNRELATED], prompt_name="document")
    print(f"  query emb shape   : {tuple(emb_q.shape)}")
    print(f"  document emb shape: {tuple(emb_d.shape)}")
    _require(
        emb_q.shape[1] == emb_d.shape[1],
        f"query and document embeddings must share the same dim; "
        f"got {emb_q.shape[1]} vs {emb_d.shape[1]}.",
    )
    # Quality sanity under the asymmetric prompts — exercises the same
    # protocol a real instruction-tuned model would use at serve time.
    sims = predictor.similarity(
        [ANCHOR], [ANCHOR, PARAPHRASE, UNRELATED],
        a_prompt_name="query", b_prompt_name="document",
    )
    sim_p, sim_u = float(sims[0, 1]), float(sims[0, 2])
    print(f"  sim(query, paraphrase) = {sim_p:.4f}")
    print(f"  sim(query, unrelated)  = {sim_u:.4f}")
    _require(
        sim_p > sim_u,
        f"paraphrase should outscore unrelated under asymmetric prompts. "
        f"Got {sim_p:.4f} <= {sim_u:.4f}.",
    )
    print("  PASS — named prompts persist on the model and serve correctly.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--recipe", type=int, required=True, choices=range(1, 9))
    p.add_argument("--model-dir", required=True)
    p.add_argument("--processed-dir", default=None,
                   help="Required for recipes 3, 7 (distill assertions).")
    args = p.parse_args()

    model_dir = args.model_dir
    if not Path(model_dir).exists():
        sys.exit(f"model dir missing: {model_dir} — did training persist final/?")

    print(f"--- Verifying Recipe {args.recipe} @ {model_dir} ---")
    if args.recipe in (1, 2, 6):
        verify_r1_r2_r6(model_dir)
    elif args.recipe in (3, 7):
        if not args.processed_dir:
            sys.exit(f"recipe {args.recipe} requires --processed-dir to verify the `label` column.")
        verify_r3_r7(model_dir, args.processed_dir)
    elif args.recipe == 4:
        verify_r4(model_dir)
    elif args.recipe == 5:
        verify_r5(model_dir)
    elif args.recipe == 8:
        verify_r8(model_dir)


if __name__ == "__main__":
    main()