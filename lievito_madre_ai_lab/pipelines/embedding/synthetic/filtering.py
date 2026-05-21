"""Filter noisy candidate pairs with cheap heuristics + an LLM judge.

Two-stage by design:

1. **Heuristic** (free): drop near-empty queries, drop pairs where the query
   is just a long verbatim copy of the chunk, drop exact duplicate anchors.
   These are obvious failure modes — paying an LLM judge to look at them is
   wasteful.
2. **LLM judge** (paid): a second LLM call rates each surviving pair 1-5 on
   relevance + specificity. Keeps only pairs at or above the threshold.

Why a second LLM call instead of a cross-encoder: at v1 you don't have a
domain-tuned cross-encoder yet, and the off-the-shelf ones are trained on
generic web data. An LLM judge with a clear rubric is a stronger zero-shot
filter on domain content. Once you've trained a domain CE, swap the judge
for it — same interface, lower cost.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from tqdm.auto import tqdm

from lievito_madre_ai_lab.pipelines.llm.base import LLMClient, LLMRequest, LLMResponse
from lievito_madre_ai_lab.pipelines.embedding.synthetic.checkpoint import CheckpointStore


_DEFAULT_JUDGE_PATH = Path(__file__).parent.parent / "prompts" / "judge.yaml"


@dataclass
class FilterConfig:
    """Heuristic-filter knobs. None of these are tunable per-row on purpose:
    if a domain needs different thresholds, fork the dataclass.

    ``max_verbatim_overlap`` is the longest run of consecutive words a query
    may share verbatim with its positive. Queries that just copy a span of
    the passage make the bi-encoder learn lexical matching, not semantics —
    they're the most common failure mode of generated queries.
    """
    min_query_words: int = 3
    max_query_words: int = 30
    max_verbatim_overlap: int = 6
    drop_duplicate_anchors: bool = True


@dataclass
class JudgeConfig:
    """LLM-judge knobs. ``min_score`` is the inclusive cutoff on the 1-5
    rubric defined in ``prompts/judge.yaml`` — default 4 keeps "answers the
    query, possibly with slightly off phrasing" and drops "related but
    doesn't actually answer"."""
    enabled: bool = True
    prompt_path: str | None = None
    min_score: int = 4
    batch_size: int = 64


_WORD_RE = re.compile(r"\w+")
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _word_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _max_verbatim_run(query: str, passage: str) -> int:
    """Longest run of consecutive query words that appears verbatim in passage.

    Word-level instead of character-level because character runs penalise
    natural paraphrases that share short substrings (e.g. "the" appears
    everywhere) and miss the actual failure mode — a query that copies a
    *phrase* from the chunk.
    """
    q_words = _word_tokens(query)
    p_words = _word_tokens(passage)
    if not q_words or not p_words:
        return 0

    p_word_set = set(p_words)
    longest = 0
    i = 0
    while i < len(q_words):
        if q_words[i] not in p_word_set:
            i += 1
            continue
        # Anchor candidate found — scan the passage for the longest match
        # starting at q_words[i]. O(len(passage)) per anchor; passages are
        # ~250 tokens so this stays cheap even on a 50k-pair corpus.
        for j in range(len(p_words)):
            if p_words[j] != q_words[i]:
                continue
            k = 0
            while (
                i + k < len(q_words)
                and j + k < len(p_words)
                and q_words[i + k] == p_words[j + k]
            ):
                k += 1
            if k > longest:
                longest = k
        i += 1
    return longest


def _heuristic_keep(row: dict, cfg: FilterConfig) -> bool:
    q = row.get("anchor", "")
    p = row.get("positive", "")
    if not q or not p:
        return False
    n_words = len(_word_tokens(q))
    if n_words < cfg.min_query_words or n_words > cfg.max_query_words:
        return False
    if _max_verbatim_run(q, p) > cfg.max_verbatim_overlap:
        return False
    return True


def _parse_judge_score(text: str) -> int | None:
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
    score = obj.get("score") if isinstance(obj, dict) else None
    if isinstance(score, bool):  # bool is an int subclass in Python — guard it
        return None
    if not isinstance(score, int):
        return None
    if score < 1 or score > 5:
        return None
    return score


def _build_judge_request(row: dict, prompt: dict) -> LLMRequest:
    user = prompt["user"].format(query=row["anchor"], chunk_text=row["positive"])
    return LLMRequest(
        system=prompt["system"],
        user=user,
        response_format="json_object",
        metadata={"row_id": row.get("row_id"), "chunk_id": row.get("chunk_id")},
    )


async def _llm_judge(
    rows: list[dict],
    *,
    client: LLMClient,
    cfg: JudgeConfig,
    checkpoint: CheckpointStore | None = None,
) -> list[dict]:
    """Score every row, keep those at/above ``cfg.min_score``.

    With ``checkpoint`` set, scores are written per-batch keyed by
    ``row_id``. Resuming uses the persisted score against the *current*
    ``min_score`` — so you can rerun with a different threshold without
    re-judging anything. Rows whose ``row_id`` is missing fall back to
    the in-memory (non-resumable) path.
    """
    prompt = yaml.safe_load(Path(cfg.prompt_path or _DEFAULT_JUDGE_PATH).read_text())

    # Partition rows by whether they're already judged in the checkpoint.
    persisted_scores: dict[str, dict] = {}
    pending: list[dict] = []
    if checkpoint is not None:
        for rec in checkpoint.load_all():
            rid = rec.get("row_id")
            if rid is not None:
                persisted_scores[str(rid)] = rec

        for row in rows:
            rid = row.get("row_id")
            if rid is not None and str(rid) in persisted_scores:
                continue
            pending.append(row)
        skipped = len(rows) - len(pending)
        if skipped:
            print(f"      [resume] judge: skipping {skipped} rows already judged")
    else:
        pending = list(rows)

    # Process pending rows in batches, writing per-batch to the checkpoint.
    in_memory_records: list[dict] = []

    async def _tick(req: LLMRequest, pbar: tqdm) -> LLMResponse:
        resp = await client.generate(req)
        pbar.update(1)
        return resp

    with tqdm(total=len(pending), desc="judge", unit="row") as pbar:
        for start in range(0, len(pending), cfg.batch_size):
            batch = pending[start : start + cfg.batch_size]
            requests = [_build_judge_request(r, prompt) for r in batch]
            responses = await asyncio.gather(*(_tick(r, pbar) for r in requests))

            batch_records: list[dict] = []
            for row, resp in zip(batch, responses):
                if resp.error:
                    continue  # transient — leave uncheckpointed so we retry.
                score = _parse_judge_score(resp.text)
                if score is None:
                    # Malformed response — checkpoint as None so we don't retry
                    # forever. The row will be dropped below.
                    rec = {"row_id": row.get("row_id"), "score": None}
                else:
                    rec = {"row_id": row.get("row_id"), "score": score}
                batch_records.append(rec)

            if checkpoint is not None:
                # Only records with a row_id are persistable.
                checkpoint.append_many([r for r in batch_records if r.get("row_id") is not None])
            in_memory_records.extend(batch_records)

    # Build the final kept list — merge persisted + current scores.
    scores_by_rid: dict[str, int | None] = {
        rid: rec.get("score") for rid, rec in persisted_scores.items()
    }
    for rec in in_memory_records:
        rid = rec.get("row_id")
        if rid is not None:
            scores_by_rid[str(rid)] = rec.get("score")

    kept: list[dict] = []
    for row in rows:
        rid = row.get("row_id")
        if rid is None:
            # No row_id — can't look up; treat as fresh judge call result.
            continue
        score = scores_by_rid.get(str(rid))
        if score is None:
            continue
        if score < cfg.min_score:
            continue
        kept.append({**row, "judge_score": score})
    return kept


async def filter_pairs(
    rows: list[dict],
    *,
    cfg: FilterConfig | None = None,
    judge_cfg: JudgeConfig | None = None,
    judge_client: LLMClient | None = None,
    judge_checkpoint: CheckpointStore | None = None,
) -> list[dict]:
    """End-to-end filter: heuristics first, then optional LLM judge.

    ``judge_client`` is required iff ``judge_cfg.enabled``. We raise rather
    than silently skip the judge stage because the choice of a judge is
    expensive enough that you almost certainly meant to make it.

    ``judge_checkpoint``: persisted (row_id → score) store. Lets a crashed
    judge run resume from where it stopped, and lets you tweak
    ``judge_cfg.min_score`` between runs without re-judging.
    """
    cfg = cfg or FilterConfig()
    judge_cfg = judge_cfg or JudgeConfig()

    # Stage 1: heuristics.
    survivors = [r for r in rows if _heuristic_keep(r, cfg)]

    # Stage 2: dedup anchors (after heuristics so we don't dedup garbage).
    if cfg.drop_duplicate_anchors:
        seen: set[str] = set()
        deduped: list[dict] = []
        for r in survivors:
            key = r["anchor"].lower().strip()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        survivors = deduped

    # Stage 3: LLM judge (optional).
    if judge_cfg.enabled:
        if judge_client is None:
            raise ValueError(
                "judge_cfg.enabled=True requires judge_client. Pass an "
                "LLMClient or set judge_cfg.enabled=False."
            )
        survivors = await _llm_judge(
            survivors, client=judge_client, cfg=judge_cfg, checkpoint=judge_checkpoint,
        )

    return survivors


def filter_pairs_sync(
    rows: list[dict],
    *,
    cfg: FilterConfig | None = None,
    judge_cfg: JudgeConfig | None = None,
    judge_client: LLMClient | None = None,
    judge_checkpoint: CheckpointStore | None = None,
) -> list[dict]:
    """Sync wrapper for CLI use."""
    return asyncio.run(filter_pairs(
        rows, cfg=cfg, judge_cfg=judge_cfg, judge_client=judge_client,
        judge_checkpoint=judge_checkpoint,
    ))
