"""Token-aware document chunking for synthetic-pair generation.

The chunk size is the single biggest lever on the *quality* of generated
queries: too short and the LLM hallucinates context that isn't there, too
long and one chunk covers multiple topics so any single query under-specifies
the positive. 200–500 tokens is the sweet spot for most retrieval workloads.

We split on token boundaries (via tiktoken) rather than characters because
the downstream LLM and the bi-encoder both think in tokens; a character-based
split has unpredictable token counts and risks blowing past the LLM's
context window on token-heavy languages.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import tiktoken


@dataclass
class ChunkingConfig:
    """Knobs for token-aware splitting.

    ``encoding_name`` picks the tokenizer. ``cl100k_base`` matches GPT-4 /
    GPT-3.5-turbo / text-embedding-3-* and is a reasonable proxy for the
    embedding models you'll actually train (Modern BERT tokenizers have
    different vocabs, but token *counts* line up closely enough for sizing).
    """
    chunk_tokens: int = 256
    overlap_tokens: int = 32
    min_chunk_tokens: int = 64  # drop tail chunks that are too short to be useful
    encoding_name: str = "cl100k_base"


def _get_encoder(name: str):
    """tiktoken encoder, cached on the module to skip repeated load cost."""
    cache = _get_encoder.__dict__.setdefault("_cache", {})
    if name not in cache:
        cache[name] = tiktoken.get_encoding(name)
    return cache[name]


def chunk_document(
    text: str,
    *,
    doc_id: str,
    cfg: ChunkingConfig | None = None,
) -> list[dict]:
    """Split one document into a list of overlapping token-aware chunks.

    Each chunk is a dict ``{"doc_id", "chunk_id", "text", "start_token",
    "end_token"}``. ``chunk_id`` is ``"{doc_id}::{i}"`` — globally unique so
    downstream stages can keep per-chunk metadata without a separate join.
    """
    cfg = cfg or ChunkingConfig()
    if cfg.overlap_tokens >= cfg.chunk_tokens:
        raise ValueError(
            f"overlap_tokens ({cfg.overlap_tokens}) must be < chunk_tokens "
            f"({cfg.chunk_tokens}); otherwise the window never advances."
        )

    enc = _get_encoder(cfg.encoding_name)
    token_ids = enc.encode(text)
    if not token_ids:
        return []

    stride = cfg.chunk_tokens - cfg.overlap_tokens
    chunks: list[dict] = []
    i = 0
    while i < len(token_ids):
        window = token_ids[i : i + cfg.chunk_tokens]
        if len(window) < cfg.min_chunk_tokens and i > 0:
            # Tail chunk too short — already covered by the previous overlap.
            break
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}::{len(chunks)}",
            "text": enc.decode(window),
            "start_token": i,
            "end_token": i + len(window),
        })
        if i + cfg.chunk_tokens >= len(token_ids):
            break
        i += stride
    return chunks


def chunk_documents(
    documents: Iterable[dict],
    *,
    cfg: ChunkingConfig | None = None,
    text_column: str = "text",
    id_column: str = "id",
) -> Iterator[dict]:
    """Stream chunks across a corpus of ``{id, text}`` documents.

    Yields one chunk dict at a time so the caller can pipe straight into
    a Dataset.from_generator without materialising 10×-larger intermediate
    lists. Documents missing ``text_column`` or with empty text are skipped
    silently — common with PDF-extracted corpora where some pages are blank.
    """
    cfg = cfg or ChunkingConfig()
    for doc in documents:
        text = doc.get(text_column)
        if not text or not text.strip():
            continue
        doc_id = str(doc.get(id_column, ""))
        if not doc_id:
            raise ValueError(
                f"Document missing required {id_column!r} field. Every "
                f"document needs a stable id so chunks join back correctly."
            )
        for chunk in chunk_document(text, doc_id=doc_id, cfg=cfg):
            yield chunk
