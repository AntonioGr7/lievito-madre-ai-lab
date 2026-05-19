"""Quick local inference for a Trainer-saved bi-encoder.

Exercises the four flows that distinguish a bi-encoder from the encoder
predictors — every call goes through the same ``BiEncoderPredictor`` used in
production, so the output here matches ``serve.py`` exactly:

  1. ``encode``     — raw embeddings (shape + L2-norm sanity)
  2. ``similarity`` — ranked cosine similarity for a query vs a small corpus
  3. ``search``     — top-k retrieval against that corpus

Two optional flows fire only if the saved model was trained for them:

  - **Matryoshka truncation** — if ``preprocessing.json`` carries
    ``default_truncate_dim`` and/or ``matryoshka_dims``, the script encodes
    once at the smallest trained dim and confirms outputs stay unit-norm.
  - **Instruction prompts** — if the model carries named ``query`` /
    ``document`` prompts, the script re-runs ``search`` asymmetrically.
"""
from lievito_madre_ai_lab.embedding.bi_encoder import serve as _serve
from lievito_madre_ai_lab.shared.preprocessing import load_preprocessing_meta

print(f"[serve loaded from: {_serve.__file__}]")
BiEncoderPredictor = _serve.BiEncoderPredictor

MODEL_DIR = "outputs/smoke_bi_encoder_r1/final"

QUERY = "How do vector databases work?"
CORPUS = [
    "Vector databases store and retrieve high-dimensional embeddings.",
    "FAISS is a library for approximate nearest-neighbour search.",
    "PyTorch is a tensor framework with automatic differentiation.",
    "Mount Everest is the tallest mountain on Earth.",
    "Sourdough bread rises faster in a warm kitchen.",
]

# Pin default_truncate_dim=None so the baseline encode shows the native dim
# even if the model was trained with Matryoshka (the predictor would otherwise
# silently truncate to the smallest trained dim).
predictor = BiEncoderPredictor(
    MODEL_DIR, use_compile=False, quantize_cpu=False, warmup_steps=0,
    default_truncate_dim=None,
)
meta = load_preprocessing_meta(MODEL_DIR) or {}

print("\n=== encode (full dim) ===")
emb = predictor.encode([QUERY, *CORPUS])
print(f"  shape={tuple(emb.shape)}  dtype={emb.dtype}")
print(f"  L2 norms={[round(n, 4) for n in emb.norm(dim=-1).tolist()]}")

print("\n=== similarity (query vs corpus, ranked) ===")
sims = predictor.similarity([QUERY], CORPUS)
ranked = sorted(zip(CORPUS, sims[0].tolist()), key=lambda x: -x[1])
for text, score in ranked:
    print(f"  {score:+.4f}  {text}")

print("\n=== search top-3 ===")
hits = predictor.search([QUERY], CORPUS, top_k=3)
for h in hits[0]:
    print(f"  score={h['score']:+.4f}  [{h['corpus_id']}] {CORPUS[h['corpus_id']]}")

# Matryoshka — pick the smallest trained dim if any.
dims = meta.get("matryoshka_dims") or []
truncate_dim = meta.get("default_truncate_dim") or (min(dims) if dims else None)
if truncate_dim:
    print(f"\n=== matryoshka truncate_dim={truncate_dim} ===")
    emb_t = predictor.encode([QUERY, *CORPUS], truncate_dim=truncate_dim)
    print(f"  shape={tuple(emb_t.shape)}")
    print(f"  L2 norms={[round(n, 4) for n in emb_t.norm(dim=-1).tolist()]}")
    hits_t = predictor.search([QUERY], CORPUS, top_k=3, truncate_dim=truncate_dim)
    for h in hits_t[0]:
        print(f"  score={h['score']:+.4f}  [{h['corpus_id']}] {CORPUS[h['corpus_id']]}")

# Instruction prompts — asymmetric search if the model carries non-empty
# ones. SentenceTransformer surfaces empty defaults for the standard prompt
# keys, so an explicit truthy check is what gates this block on a real
# instruction-tuned model rather than the default plumbing.
qp = (predictor.prompts.get("query") or "").strip()
dp = (predictor.prompts.get("document") or "").strip()
if qp and dp:
    print("\n=== search top-3 (asymmetric prompts) ===")
    print(f"  query prompt    = {qp!r}")
    print(f"  document prompt = {dp!r}")
    hits_p = predictor.search(
        [QUERY], CORPUS, top_k=3,
        query_prompt_name="query",
        corpus_prompt_name="document",
    )
    for h in hits_p[0]:
        print(f"  score={h['score']:+.4f}  [{h['corpus_id']}] {CORPUS[h['corpus_id']]}")