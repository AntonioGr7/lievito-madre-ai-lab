"""LLM-based query generation: chunks → (query, chunk) candidate pairs.

One LLM call per chunk asks for ``num_queries`` queries spanning a set of
diverse styles (natural question, keyword, paraphrase, messy). The styles
list determines both the number of queries and the labels echoed back into
the dataset, so callers can stratify evaluation by style later.

Failures are surfaced in-band: if the API call errors the chunk is NOT
checkpointed (so a future resume retries it); if the API returns malformed
JSON the chunk IS checkpointed with an empty row list (a model glitch that
will likely repeat — manual ``--fresh`` is required to retry).

Resumability: when ``checkpoint`` is provided, every successfully-processed
chunk is appended to it per-batch. On the next run with the same
checkpoint, those chunks are skipped.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from tqdm.auto import tqdm

from lievito_madre_ai_lab.pipelines.llm.base import LLMClient, LLMRequest, LLMResponse
from lievito_madre_ai_lab.pipelines.embedding.synthetic.checkpoint import CheckpointStore


_DEFAULT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "query_gen.yaml"


@dataclass
class QueryGenConfig:
    """Knobs for the generation stage.

    ``styles`` is the list of style *keys* to use (must match keys in the
    YAML's ``styles`` map). ``num_queries`` is derived from ``len(styles)``
    so the two can't drift apart — the prompt explicitly lists each style.
    """
    styles: list[str] = field(default_factory=lambda: [
        "natural_question",
        "keyword_query",
        "paraphrased_statement",
    ])
    prompt_path: str | None = None  # defaults to the bundled query_gen.yaml
    batch_size: int = 64  # number of chunks per `generate_batch` call


def _load_prompt(path: str | None) -> dict:
    p = Path(path) if path else _DEFAULT_PROMPT_PATH
    return yaml.safe_load(p.read_text())


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_response(text: str) -> list[dict] | None:
    """Pull the JSON object out of an LLM response, return its ``queries`` list.

    Models occasionally wrap JSON in markdown fences or prepend prose despite
    ``response_format=json_object``. Regex-extracting the outermost ``{...}``
    is robust enough for the cases we've actually seen in practice; if both
    that and a strict json.loads fail, the row is dropped.
    """
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
    queries = obj.get("queries") if isinstance(obj, dict) else None
    if not isinstance(queries, list):
        return None
    return queries


def _build_request(chunk: dict, prompt: dict, styles: list[str]) -> LLMRequest:
    style_list = "\n".join(
        f"  - {s}: {prompt['styles'][s]}" for s in styles
    )
    user = prompt["user"].format(
        chunk_text=chunk["text"],
        num_queries=len(styles),
        style_list=style_list,
    )
    return LLMRequest(
        system=prompt["system"],
        user=user,
        response_format="json_object",
        metadata={"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"]},
    )


def _rows_from_response(chunk: dict, resp: LLMResponse) -> list[dict] | None:
    """Convert one LLM response into a list of pair rows.

    Returns ``None`` when the API errored (caller should NOT checkpoint —
    retry next run). Returns ``[]`` when the API returned but the response
    was unusable (caller SHOULD checkpoint — same response likely on
    retry; force re-run with ``--fresh``).
    """
    if resp.error:
        return None
    parsed = _parse_response(resp.text)
    if not parsed:
        return []
    rows: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        query = item.get("text")
        style = item.get("style")
        if not isinstance(query, str) or not query.strip():
            continue
        style_str = style if isinstance(style, str) else "unknown"
        rows.append({
            "anchor": query.strip(),
            "positive": chunk["text"],
            "doc_id": chunk["doc_id"],
            "chunk_id": chunk["chunk_id"],
            "style": style_str,
            "row_id": f"{chunk['chunk_id']}::{style_str}",
        })
    return rows


async def generate_queries_for_chunks(
    chunks: list[dict],
    *,
    client: LLMClient,
    cfg: QueryGenConfig | None = None,
    checkpoint: CheckpointStore | None = None,
) -> list[dict]:
    """Run query generation across a list of chunks → flat list of pair rows.

    With ``checkpoint`` set: skips chunks already present in the
    checkpoint, appends one record per chunk (``{chunk_id, rows: [...]}``)
    per batch, and returns the consolidated view (past runs + this run).

    Without ``checkpoint``: behaves as before (pure in-memory, no resume).
    """
    cfg = cfg or QueryGenConfig()
    prompt = _load_prompt(cfg.prompt_path)
    unknown = set(cfg.styles) - set(prompt["styles"])
    if unknown:
        raise ValueError(
            f"Unknown style(s) {sorted(unknown)} not in prompt's styles map: "
            f"{sorted(prompt['styles'])}. Either add them to the YAML or "
            f"drop them from the config."
        )

    if checkpoint is not None:
        pending = [c for c in chunks if not checkpoint.is_done(c["chunk_id"])]
        skipped = len(chunks) - len(pending)
        if skipped:
            print(f"      [resume] single-hop: skipping {skipped} chunks already in checkpoint")
    else:
        pending = chunks

    in_memory_records: list[dict] = []

    async def _tick(req: LLMRequest, pbar: tqdm) -> LLMResponse:
        resp = await client.generate(req)
        pbar.update(1)
        return resp

    with tqdm(total=len(pending), desc="single-hop queries", unit="chunk") as pbar:
        for start in range(0, len(pending), cfg.batch_size):
            batch = pending[start : start + cfg.batch_size]
            requests = [_build_request(c, prompt, cfg.styles) for c in batch]
            responses = await asyncio.gather(*(_tick(r, pbar) for r in requests))

            batch_records: list[dict] = []
            for chunk, resp in zip(batch, responses):
                rows = _rows_from_response(chunk, resp)
                if rows is None:
                    # API error — leave uncheckpointed so the next run retries.
                    continue
                batch_records.append({
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "rows": rows,
                })

            if checkpoint is not None:
                checkpoint.append_many(batch_records)
            else:
                in_memory_records.extend(batch_records)

    # Consolidate past + current results.
    all_records = checkpoint.load_all() if checkpoint is not None else in_memory_records
    rows: list[dict] = []
    for rec in all_records:
        rows.extend(rec.get("rows", []))
    return rows


def generate_queries_for_chunks_sync(
    chunks: list[dict],
    *,
    client: LLMClient,
    cfg: QueryGenConfig | None = None,
    checkpoint: CheckpointStore | None = None,
) -> list[dict]:
    """Sync wrapper for CLI use — runs the async pipeline in a new loop."""
    return asyncio.run(generate_queries_for_chunks(
        chunks, client=client, cfg=cfg, checkpoint=checkpoint,
    ))
