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


# ──────────────────────────────────────────────────────────────────────────
# Instruction-prompt registry for prefix-required retrievers
# ──────────────────────────────────────────────────────────────────────────
# Several SoTA embedders were pre-trained with mandatory query/document
# prefixes. Calling them without the prefix degrades (BGE, mxbai) or fully
# collapses (E5, Nomic) the embedding space — every candidate then scores
# ~as high as the positive and the relative_margin filter wipes them all
# out. Matched by name prefix so a whole sub-family (small/base/large,
# v1/v1.5/v2) shares one entry.
KNOWN_PROMPTS: list[tuple[str, str | None, str | None]] = [
    # E5 — prefixes are MANDATORY.
    ("intfloat/e5-",                       "query: ",          "passage: "),
    ("intfloat/multilingual-e5-",          "query: ",          "passage: "),
    # BGE-en — asymmetric: long query instruction, bare passages.
    ("BAAI/bge-large-en",                  "Represent this sentence for searching relevant passages: ", None),
    ("BAAI/bge-base-en",                   "Represent this sentence for searching relevant passages: ", None),
    ("BAAI/bge-small-en",                  "Represent this sentence for searching relevant passages: ", None),
    # mxbai / Snowflake Arctic — same BGE-style instruction.
    ("mixedbread-ai/mxbai-embed-large-v1", "Represent this sentence for searching relevant passages: ", None),
    ("Snowflake/snowflake-arctic-embed-",  "Represent this sentence for searching relevant passages: ", None),
    # Nomic — search_query / search_document.
    ("nomic-ai/nomic-embed-text",          "search_query: ",   "search_document: "),
    ("nomic-ai/modernbert-embed-",         "search_query: ",   "search_document: "),
]


def resolve_prompts(model_name: str) -> tuple[str | None, str | None]:
    """Auto-detect ``(query_prompt, corpus_prompt)`` for a retriever by name.

    Returns ``(None, None)`` for unknown models — the caller decides whether
    that's safe (genuine symmetric model: gte, mpnet, all-MiniLM) or a
    misconfiguration the registry hasn't learned yet. The mining script
    prints the resolved prompts so misconfiguration is visible.
    """
    for prefix, q, c in KNOWN_PROMPTS:
        if model_name.startswith(prefix):
            return q, c
    return None, None


@dataclass
class RetrieverSpec:
    """A retriever paired with its (optional) instruction prefixes.

    The ensemble path needs to track prompts per retriever — members of
    the ensemble use different protocols. Wrapping them avoids parallel
    lists at the call site.
    """
    model: SentenceTransformer
    query_prompt: str | None = None
    corpus_prompt: str | None = None


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
    query_prompt: str | None = None,
    corpus_prompt: str | None = None,
) -> Dataset:
    """Mine hard negatives with a dense bi-encoder retriever.

    `corpus` defaults to the unique positives in `dataset`. Pass a larger
    pool (e.g. all documents in your archive) for stronger mining.

    `query_prompt` / `corpus_prompt` are the instruction prefixes the
    retriever was trained with (``"query: "`` / ``"passage: "`` for E5,
    ``"search_query: "`` / ``"search_document: "`` for Nomic, etc.). Skip
    them on a prefix-required model and the embedding space collapses —
    candidates score ~as high as the positive and `relative_margin` drops
    everything. Use :func:`resolve_prompts` to look these up by model name.
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
        query_prompt=query_prompt,
        corpus_prompt=corpus_prompt,
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

    # Build per-anchor candidate windows up front so the CE pass can batch
    # across all anchors. Per-anchor CE calls are what makes this step
    # silently hang on CPU — dispatch overhead × len(anchors), no
    # progress bar.
    if cfg.sampling_strategy == "random":
        import random
        rng = random.Random()
    else:
        rng = None

    all_cand_texts: list[list[str]] = []
    for i, positive in enumerate(positives):
        cand_ids = list(doc_indices[i])
        pos_idx = pos_to_idx.get(positive)
        if pos_idx is not None:
            cand_ids = [c for c in cand_ids if c != pos_idx]
        windowed = cand_ids[cfg.range_min : cfg.range_max]
        if rng is not None:
            rng.shuffle(windowed)
        all_cand_texts.append([corpus_texts[c] for c in windowed])

    # Cross-encoder filter: two batched calls — one for positives, one
    # for the flattened candidate pool — instead of 2 × len(anchors)
    # tiny calls.
    if cross_encoder is not None:
        pos_scores = cross_encoder.predict(
            list(zip(anchors, positives)),
            batch_size=cfg.batch_size,
            show_progress_bar=cfg.verbose,
        )
        flat_pairs = [(anchors[i], t) for i, texts in enumerate(all_cand_texts) for t in texts]
        flat_scores = cross_encoder.predict(
            flat_pairs,
            batch_size=cfg.batch_size,
            show_progress_bar=cfg.verbose,
        )
        per_anchor_scores: list[list[float]] = []
        offset = 0
        for texts in all_cand_texts:
            n = len(texts)
            per_anchor_scores.append([float(s) for s in flat_scores[offset:offset + n]])
            offset += n
    else:
        pos_scores = None
        per_anchor_scores = [[] for _ in anchors]

    rows = []
    for i, (anchor, positive) in enumerate(zip(anchors, positives)):
        cand_texts = all_cand_texts[i]

        if cross_encoder is not None and cand_texts:
            cap = _filter_cap(cfg, float(pos_scores[i]))
            kept = [
                (t, s) for t, s in zip(cand_texts, per_anchor_scores[i])
                if (cap is None or s <= cap)
                and (cfg.min_score is None or s >= cfg.min_score)
                and (cfg.max_score is None or s <= cfg.max_score)
            ]
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
    retrievers: list[RetrieverSpec],
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

    Each :class:`RetrieverSpec` carries its own instruction prefixes — the
    members of an ensemble routinely use different protocols (BGE's long
    English instruction vs. E5's ``"query: "`` vs. Nomic's
    ``"search_query: "``), so prompts can't be set once at the call site.

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
    # Over-mine per source so dedup doesn't starve us, AND disable every
    # score-based filter — the union-level cross-encoder pass below is the
    # one filter calibrated for the 60%-of-positive rule. Leaving
    # relative_margin/margin/min_score/max_score on at the per-source step
    # makes ST apply them to the *retriever's own* cosine scores, which
    # are model-dependent: BGE's spread tolerates it, E5's tight cluster
    # silently drops ~all candidates, and the per-retriever signal we
    # asked for never reaches the union.
    per_source_cfg = MiningConfig(**{
        **cfg.__dict__,
        "num_negatives": cfg.num_negatives * 2,
        "relative_margin": None,
        "margin": None,
        "min_score": None,
        "max_score": None,
    })

    per_source: list[Dataset] = []
    for spec in retrievers:
        per_source.append(
            mine_hard_negatives_for_dataset(
                dataset, retriever=spec.model,
                # Skip the per-retriever cross-encoder filter — we do one
                # pass at the union level instead. Cheaper and consistent.
                cross_encoder=None,
                cfg=per_source_cfg,
                corpus=corpus,
                query_prompt=spec.query_prompt,
                corpus_prompt=spec.corpus_prompt,
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

    # Per-anchor candidate pool: union across sources, dedupe, drop the
    # positive itself. No scoring yet — we batch that next.
    all_candidates: list[list[str]] = []
    for i, positive in enumerate(positives):
        seen: set[str] = set()
        cands: list[str] = []
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
                    cands.append(cand)
        all_candidates.append(cands)

    # Cross-encoder filter at the union level (60%-of-positive rule).
    # Score every (anchor, positive) and every (anchor, candidate) in two
    # large batched calls — one CE call per anchor is what makes this step
    # appear to hang on CPU: dispatch overhead dominates, the progress bar
    # is suppressed, and the user sees nothing for minutes.
    if cross_encoder is not None:
        pos_scores = cross_encoder.predict(
            list(zip(anchors, positives)),
            batch_size=cfg.batch_size,
            show_progress_bar=cfg.verbose,
        )
        flat_pairs = [(anchors[i], c) for i, cands in enumerate(all_candidates) for c in cands]
        flat_scores = cross_encoder.predict(
            flat_pairs,
            batch_size=cfg.batch_size,
            show_progress_bar=cfg.verbose,
        )
        # Slice the flat candidate-score array back per anchor.
        per_anchor_scores: list[list[float]] = []
        offset = 0
        for cands in all_candidates:
            n = len(cands)
            per_anchor_scores.append([float(s) for s in flat_scores[offset:offset + n]])
            offset += n
    else:
        pos_scores = None
        per_anchor_scores = [[] for _ in anchors]

    if cfg.sampling_strategy == "random":
        import random
        rng = random.Random()
    else:
        rng = None

    rows = []
    for i, (anchor, positive) in enumerate(zip(anchors, positives)):
        candidates = all_candidates[i]
        if cross_encoder is not None and candidates:
            cap = _filter_cap(cfg, float(pos_scores[i]))
            paired = [
                (t, s) for t, s in zip(candidates, per_anchor_scores[i])
                if (cap is None or s <= cap)
                and (cfg.min_score is None or s >= cfg.min_score)
                and (cfg.max_score is None or s <= cfg.max_score)
            ]
            if cfg.sampling_strategy == "top":
                paired.sort(key=lambda kv: -kv[1])
            else:
                rng.shuffle(paired)
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
