"""Hard-negative mining for bi-encoder training.

Mining is the highest-leverage knob in retrieval fine-tuning: in-batch
negatives only hold the model accountable for *random* documents, while
hard negatives force it to learn the fine-grained boundaries that actually
matter at inference.

Three mining strategies, in order of increasing power:

1. **Dense mining** (``mine_hard_negatives_for_dataset``) — single dense
   bi-encoder retriever + optional cross-encoder filter. The default.
2. **BM25 mining** (``mine_with_bm25_for_dataset``) — lexical retriever.
   Surfaces a *different* class of hard negatives (high-overlap but
   semantically wrong) that dense retrievers miss. Useful on its own for
   domains where lexical signal matters, but most powerful in an ensemble.
3. **Ensemble mining** (``mine_ensemble_for_dataset``) — union of one or
   more dense retrievers + optional BM25, deduped, then cross-encoder
   filtered. The strongest setup: each retriever contributes a *different
   slice* of hardness and the cross-encoder filter keeps only the truly-
   wrong-but-confusing ones.

All three strategies share the same "60% of positive score" rule for
filtering false negatives, expressed as ``relative_margin=0.4`` — keep a
negative only when its cross-encoder score is at most 60% of the
positive's.

Diversity knobs:

- ``range_min``: skip the top-K most-similar candidates (usually
  near-duplicates of the positive). K=10 for noisy/web corpora, 0 for
  clean human-curated corpora.
- ``range_max``: sample from the top-K. Smaller = harder, larger = more
  diverse but easier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from datasets import Dataset
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.util import mine_hard_negatives

OutputFormat = Literal["triplet", "n-tuple", "labeled-pair", "labeled-list"]


@dataclass
class MiningConfig:
    """Knobs for the mining helpers. The dense fields map onto ST's
    `mine_hard_negatives` arguments; BM25/ensemble paths read the same
    knobs so callers don't have to re-learn the interface."""
    num_negatives: int = 5
    # Cross-encoder filter: drop any negative whose score exceeds
    # ``(1 - relative_margin) * positive_score``. 0.4 ≈ "negative may be at
    # most 60% as relevant as the positive" — the recipe default.
    relative_margin: float | None = 0.4
    margin: float | None = None  # absolute alternative; None = disabled
    max_score: float | None = None
    min_score: float | None = None
    range_min: int = 10
    range_max: int = 100
    sampling_strategy: Literal["top", "random"] = "top"
    output_format: OutputFormat = "n-tuple"
    use_faiss: bool = False
    batch_size: int = 64
    anchor_column: str = "anchor"
    positive_column: str = "positive"
    verbose: bool = True


# ──────────────────────────────────────────────────────────────────────────
# Dense (single bi-encoder) mining
# ──────────────────────────────────────────────────────────────────────────


def mine_hard_negatives_for_dataset(
    dataset: Dataset,
    *,
    retriever: SentenceTransformer,
    cross_encoder: CrossEncoder | None = None,
    cfg: MiningConfig | None = None,
    corpus: list[str] | None = None,
) -> Dataset:
    """Mine hard negatives with a dense bi-encoder retriever.

    `corpus` defaults to the unique positives in `dataset`. Pass a larger
    pool (e.g. all documents in your archive) for stronger mining.
    """
    cfg = cfg or MiningConfig()

    kwargs = dict(
        dataset=dataset,
        model=retriever,
        cross_encoder=cross_encoder,
        num_negatives=cfg.num_negatives,
        margin=cfg.margin,
        relative_margin=cfg.relative_margin,
        max_score=cfg.max_score,
        min_score=cfg.min_score,
        range_min=cfg.range_min,
        range_max=cfg.range_max,
        sampling_strategy=cfg.sampling_strategy,
        output_format=cfg.output_format,
        use_faiss=cfg.use_faiss,
        batch_size=cfg.batch_size,
        anchor_column_name=cfg.anchor_column,
        positive_column_name=cfg.positive_column,
        verbose=cfg.verbose,
    )
    if corpus is not None:
        kwargs["corpus"] = corpus

    return mine_hard_negatives(**kwargs)


# ──────────────────────────────────────────────────────────────────────────
# BM25 mining
# ──────────────────────────────────────────────────────────────────────────


def mine_with_bm25_for_dataset(
    dataset: Dataset,
    *,
    cross_encoder: CrossEncoder | None = None,
    cfg: MiningConfig | None = None,
    corpus: list[str] | None = None,
    stopwords: str | None = "en",
) -> Dataset:
    """Mine hard negatives with a BM25 lexical retriever.

    BM25 surfaces a *different* failure mode than dense retrieval — high
    lexical overlap with the query but wrong meaning. Pairing BM25
    negatives with dense negatives in an ensemble is the recipe for
    maximum diversity.

    The cross-encoder filter still applies the same 60%-of-positive rule.
    """
    import bm25s

    cfg = cfg or MiningConfig()
    anchors = dataset[cfg.anchor_column]
    positives = dataset[cfg.positive_column]

    # Build the BM25 index from the corpus (or the unique positives).
    corpus_texts = list(corpus) if corpus is not None else list(dict.fromkeys(positives))
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords=stopwords)
    bm25 = bm25s.BM25()
    bm25.index(corpus_tokens)

    # Retrieve top-K candidates, K wide enough to span the range_min:range_max window.
    k = max(cfg.range_max, cfg.range_min + cfg.num_negatives + 1)
    query_tokens = bm25s.tokenize(anchors, stopwords=stopwords)
    doc_indices, doc_scores = bm25.retrieve(query_tokens, k=k)

    pos_to_idx = {p: i for i, p in enumerate(corpus_texts)}

    # For each anchor, slice candidates[range_min:range_max], drop the
    # positive if present, optionally cross-encoder-filter, then keep the
    # top num_negatives.
    rows = []
    for i, (anchor, positive) in enumerate(zip(anchors, positives)):
        cand_ids = list(doc_indices[i])
        # Drop the positive itself before applying the range window.
        pos_idx = pos_to_idx.get(positive)
        if pos_idx is not None:
            cand_ids = [c for c in cand_ids if c != pos_idx]
        windowed = cand_ids[cfg.range_min : cfg.range_max]
        if cfg.sampling_strategy == "random":
            import random
            random.shuffle(windowed)

        # Pull candidate texts.
        cand_texts = [corpus_texts[c] for c in windowed]

        # Optional cross-encoder filter at the 60%-of-positive threshold.
        if cross_encoder is not None and cand_texts:
            pos_score = float(cross_encoder.predict([(anchor, positive)])[0])
            cand_scores = cross_encoder.predict(
                [(anchor, t) for t in cand_texts],
                batch_size=cfg.batch_size,
                show_progress_bar=False,
            )
            cap = _filter_cap(cfg, pos_score)
            kept = [
                (t, float(s)) for t, s in zip(cand_texts, cand_scores)
                if (cap is None or s <= cap)
                and (cfg.min_score is None or s >= cfg.min_score)
                and (cfg.max_score is None or s <= cfg.max_score)
            ]
            # Take the top-`num_negatives` after filtering (or keep order on `random`).
            if cfg.sampling_strategy == "top":
                kept.sort(key=lambda kv: -kv[1])
            cand_texts = [t for t, _ in kept]

        # Pad / truncate to exactly num_negatives by sampling more if needed.
        if len(cand_texts) >= cfg.num_negatives:
            negs = cand_texts[: cfg.num_negatives]
        else:
            # Not enough candidates passed the filter — skip the row,
            # consistent with how ST's `mine_hard_negatives` handles it.
            continue

        row = {cfg.anchor_column: anchor, cfg.positive_column: positive}
        for j, neg in enumerate(negs, start=1):
            row[f"negative_{j}" if cfg.num_negatives > 1 else "negative"] = neg
        rows.append(row)

    if not rows:
        raise RuntimeError(
            "BM25 mining produced no rows after filtering. Loosen "
            "--relative-margin or expand --range-max."
        )
    return Dataset.from_list(rows)


def _filter_cap(cfg: MiningConfig, pos_score: float) -> float | None:
    """The maximum negative score allowed, given the filter config."""
    if cfg.relative_margin is not None:
        return (1.0 - cfg.relative_margin) * pos_score
    if cfg.margin is not None:
        return pos_score - cfg.margin
    return None


# ──────────────────────────────────────────────────────────────────────────
# Ensemble mining
# ──────────────────────────────────────────────────────────────────────────


def mine_ensemble_for_dataset(
    dataset: Dataset,
    *,
    retrievers: list[SentenceTransformer],
    use_bm25: bool = False,
    cross_encoder: CrossEncoder | None = None,
    cfg: MiningConfig | None = None,
    corpus: list[str] | None = None,
) -> Dataset:
    """Mine with multiple retrievers (and optionally BM25), then union +
    dedupe + cross-encoder-filter the per-anchor candidate pool.

    Each retriever contributes its own slice of "hard": dense bi-encoders
    flag semantically-close-but-wrong passages, BM25 flags lexically-close-
    but-wrong ones. The union maximises diversity; the cross-encoder filter
    keeps only candidates that are genuinely wrong.

    Per-anchor pipeline:

    1. Mine ``2 * num_negatives`` candidates per retriever (over-mine to
       leave headroom after dedup).
    2. Union the per-retriever multi-negative rows.
    3. Dedupe by negative text.
    4. Re-filter at the union level with the cross-encoder + 60% rule.
    5. Keep top ``num_negatives``.
    """
    if not retrievers and not use_bm25:
        raise ValueError("Ensemble mining needs at least one retriever or use_bm25=True.")

    cfg = cfg or MiningConfig()
    # Over-mine per source so dedup doesn't starve us.
    per_source_cfg = MiningConfig(**{**cfg.__dict__, "num_negatives": cfg.num_negatives * 2})

    per_source: list[Dataset] = []
    for retriever in retrievers:
        per_source.append(
            mine_hard_negatives_for_dataset(
                dataset, retriever=retriever,
                # Skip the per-retriever cross-encoder filter — we do one
                # pass at the union level instead. Cheaper and consistent.
                cross_encoder=None,
                cfg=per_source_cfg,
                corpus=corpus,
            )
        )
    if use_bm25:
        per_source.append(
            mine_with_bm25_for_dataset(
                dataset, cross_encoder=None, cfg=per_source_cfg, corpus=corpus,
            )
        )

    anchors = dataset[cfg.anchor_column]
    positives = dataset[cfg.positive_column]
    rows = []
    for i, (anchor, positive) in enumerate(zip(anchors, positives)):
        # Union the candidate negatives from every source for this row.
        seen: set[str] = set()
        candidates: list[str] = []
        for src in per_source:
            if i >= len(src):
                continue
            row = src[i]
            for col in row:
                if col in (cfg.anchor_column, cfg.positive_column):
                    continue
                cand = row[col]
                if cand and cand not in seen and cand != positive:
                    seen.add(cand)
                    candidates.append(cand)

        # Cross-encoder filter at the union level (60%-of-positive rule).
        if cross_encoder is not None and candidates:
            pos_score = float(cross_encoder.predict([(anchor, positive)])[0])
            cand_scores = cross_encoder.predict(
                [(anchor, t) for t in candidates],
                batch_size=cfg.batch_size,
                show_progress_bar=False,
            )
            cap = _filter_cap(cfg, pos_score)
            paired = [
                (t, float(s)) for t, s in zip(candidates, cand_scores)
                if (cap is None or s <= cap)
                and (cfg.min_score is None or s >= cfg.min_score)
                and (cfg.max_score is None or s <= cfg.max_score)
            ]
            if cfg.sampling_strategy == "top":
                paired.sort(key=lambda kv: -kv[1])
            else:
                import random
                random.shuffle(paired)
            candidates = [t for t, _ in paired]

        if len(candidates) < cfg.num_negatives:
            continue  # not enough — drop the row, consistent with ST's behavior
        negs = candidates[: cfg.num_negatives]

        row = {cfg.anchor_column: anchor, cfg.positive_column: positive}
        for j, neg in enumerate(negs, start=1):
            row[f"negative_{j}" if cfg.num_negatives > 1 else "negative"] = neg
        rows.append(row)

    if not rows:
        raise RuntimeError(
            "Ensemble mining produced no rows after filtering. Loosen "
            "--relative-margin, expand --range-max, or add more retrievers."
        )
    return Dataset.from_list(rows)
