"""Intra-document multi-hop query generation.

Produces queries that require multiple chunks of the *same* document to
answer, trained as standard `(anchor, positive)` rows: one query Q with K
positives emits K rows `(Q, C_i)`, all sharing the same anchor. MNRL
handles this natively — the model learns Q close to every C_i.

Cross-document multi-hop is intentionally out of scope here: it needs a
chunk-clustering step and a different split / hard-negative-mining story.
Add it as a separate module once intra-doc has been validated on real data.

Sizing semantics — ``target_share``:

The knob is the *target fraction of total generated rows that should be
multi-hop*, not the fraction of windows to sample. The pipeline computes
the absolute window count from the corpus stats:

    n_groups ≈ target_share / (1 - target_share) × n_chunks × |styles| / k

This is much easier to reason about than the previous window-sampling
ratio: ``target_share=0.2`` means "out of every 10 generated pairs, ~2 are
multi-hop and ~8 are single-hop", regardless of how many styles or chunks
the run has.

Why adjacent windows: a random pair of chunks from the same doc rarely
shares an answerable thread (the doc may be long, the chunks may discuss
unrelated subjects); adjacent chunks almost always do. The LLM is still
instructed to refuse when no shared thread exists — refusals show up as
empty queries and are dropped silently.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml
from tqdm.auto import tqdm

from bi_encoder_dataset.llm.base import LLMClient, LLMRequest, LLMResponse
from bi_encoder_dataset.synthetic.checkpoint import CheckpointStore


_DEFAULT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "multi_hop_query_gen.yaml"
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class MultiHopConfig:
    """Knobs for the multi-hop stage.

    ``target_share`` is the *target fraction of total generated rows that
    should be multi-hop* (e.g. 0.2 → ~20% of pairs come from multi-hop
    groups, ~80% from single-chunk styles). The pipeline converts this
    into an absolute window count using the corpus stats.

    ``stride`` lets the windows skip ahead (default 1 = every adjacent
    K-tuple, k = non-overlapping).
    """
    enabled: bool = True
    k: int = 2  # chunks per multi-hop query (>= 2)
    target_share: float = 0.2  # fraction of final rows that should be multi-hop
    stride: int = 1  # window step; 1 = every adjacent K-tuple, k = non-overlapping
    batch_size: int = 32  # multi-hop prompts are longer; smaller batch keeps tokens/call sane
    prompt_path: str | None = None
    seed: int = 42


def compute_n_groups(
    n_chunks: int,
    *,
    target_share: float,
    n_styles: int,
    k: int,
) -> int:
    """Convert a target multi-hop share into an absolute group count.

    Derivation: if S = single-hop row estimate = n_chunks × n_styles and
    M = multi-hop row count = n_groups × k, then

        target_share = M / (S + M)
        ⇒ n_groups = target_share / (1 - target_share) × S / k

    Returns 0 when ``target_share == 0``. Caller is responsible for
    capping against the actual number of available adjacent windows.
    """
    if not (0.0 <= target_share < 1.0):
        raise ValueError(
            f"target_share must be in [0, 1); got {target_share}"
        )
    if target_share == 0.0:
        return 0
    single_rows = n_chunks * n_styles
    return int(round((target_share / (1.0 - target_share)) * single_rows / k))


def build_multi_hop_groups(
    chunks: list[dict],
    *,
    k: int = 2,
    stride: int = 1,
    n_groups: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Group adjacent chunks within each document into K-tuples.

    Returns a list of ``{"group_id", "doc_id", "chunks"}`` dicts where
    ``chunks`` is the ordered list of K chunk dicts. Docs with fewer than
    K chunks contribute nothing — silently.

    ``n_groups`` caps the global sample size (across all docs). When
    ``None``, returns every available window. Sampling is uniform across
    docs which naturally biases toward longer docs in proportion to their
    window count — same property the previous per-doc sampling had.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2; got {k}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1; got {stride}")

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for ch in chunks:
        by_doc[ch["doc_id"]].append(ch)

    all_windows: list[dict] = []
    for doc_id, doc_chunks in by_doc.items():
        if len(doc_chunks) < k:
            continue
        for i in range(0, len(doc_chunks) - k + 1, stride):
            win = doc_chunks[i : i + k]
            chunk_ids = [c["chunk_id"] for c in win]
            all_windows.append({
                "group_id": f"{doc_id}::mh::{'+'.join(chunk_ids)}",
                "doc_id": doc_id,
                "chunks": win,
            })

    if n_groups is None or n_groups >= len(all_windows):
        return all_windows

    rng = random.Random(seed)
    return rng.sample(all_windows, k=max(n_groups, 0))


def _parse_response(text: str) -> str | None:
    """Extract the ``query`` field; treat empty / missing as a refusal."""
    if not text:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_OBJECT_RE.search(text)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    query = obj.get("query")
    if not isinstance(query, str):
        return None
    query = query.strip()
    return query or None


def _format_chunks_block(group_chunks: list[dict]) -> str:
    """Number the passages so the prompt can refer to them in order."""
    parts = []
    for i, ch in enumerate(group_chunks, start=1):
        parts.append(f"Passage {i}:\n---\n{ch['text']}\n---")
    return "\n\n".join(parts)


def _build_request(group: dict, prompt: dict) -> LLMRequest:
    user = prompt["user"].format(
        num_chunks=len(group["chunks"]),
        chunks_block=_format_chunks_block(group["chunks"]),
    )
    return LLMRequest(
        system=prompt["system"],
        user=user,
        response_format="json_object",
        metadata={"group_id": group["group_id"], "doc_id": group["doc_id"]},
    )


def _rows_from_response(group: dict, resp: LLMResponse) -> list[dict] | None:
    """Convert one multi-hop response into K rows.

    Same convention as single-hop: ``None`` for API error (don't checkpoint),
    ``[]`` for refusal or parse failure (checkpoint as "tried, no rows").
    """
    if resp.error:
        return None
    query = _parse_response(resp.text)
    if not query:
        return []
    rows: list[dict] = []
    for chunk in group["chunks"]:
        rows.append({
            "anchor": query,
            "positive": chunk["text"],
            "doc_id": chunk["doc_id"],
            "chunk_id": chunk["chunk_id"],
            "style": "multi_hop",
            "group_id": group["group_id"],
            "row_id": f"{group['group_id']}::{chunk['chunk_id']}",
        })
    return rows


async def generate_multi_hop_queries_for_groups(
    groups: list[dict],
    *,
    client: LLMClient,
    cfg: MultiHopConfig | None = None,
    checkpoint: CheckpointStore | None = None,
) -> list[dict]:
    """Generate one query per group, emit K rows per accepted query.

    With ``checkpoint`` set: skips groups already present, appends one
    record per group (``{group_id, rows: [...]}``) per batch, returns the
    consolidated view.
    """
    cfg = cfg or MultiHopConfig()
    if not cfg.enabled or not groups:
        return []

    prompt = yaml.safe_load(Path(cfg.prompt_path or _DEFAULT_PROMPT_PATH).read_text())

    if checkpoint is not None:
        pending = [g for g in groups if not checkpoint.is_done(g["group_id"])]
        skipped = len(groups) - len(pending)
        if skipped:
            print(f"      [resume] multi-hop: skipping {skipped} groups already in checkpoint")
    else:
        pending = groups

    in_memory_records: list[dict] = []

    async def _tick(req: LLMRequest, pbar: tqdm) -> LLMResponse:
        resp = await client.generate(req)
        pbar.update(1)
        return resp

    with tqdm(total=len(pending), desc="multi-hop queries", unit="group") as pbar:
        for start in range(0, len(pending), cfg.batch_size):
            batch = pending[start : start + cfg.batch_size]
            requests = [_build_request(g, prompt) for g in batch]
            responses = await asyncio.gather(*(_tick(r, pbar) for r in requests))

            batch_records: list[dict] = []
            for group, resp in zip(batch, responses):
                rows = _rows_from_response(group, resp)
                if rows is None:
                    continue  # API error — don't checkpoint, retry next run.
                batch_records.append({
                    "group_id": group["group_id"],
                    "doc_id": group["doc_id"],
                    "rows": rows,
                })

            if checkpoint is not None:
                checkpoint.append_many(batch_records)
            else:
                in_memory_records.extend(batch_records)

    all_records = checkpoint.load_all() if checkpoint is not None else in_memory_records
    rows: list[dict] = []
    for rec in all_records:
        rows.extend(rec.get("rows", []))
    return rows


def generate_multi_hop_queries_for_groups_sync(
    groups: list[dict],
    *,
    client: LLMClient,
    cfg: MultiHopConfig | None = None,
    checkpoint: CheckpointStore | None = None,
) -> list[dict]:
    """Sync wrapper for CLI use."""
    return asyncio.run(generate_multi_hop_queries_for_groups(
        groups, client=client, cfg=cfg, checkpoint=checkpoint,
    ))
