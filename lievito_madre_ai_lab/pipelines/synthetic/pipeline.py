"""End-to-end synthetic-pair pipeline: documents → DatasetDict.

Stages, all driven by one config:

1. Load documents (jsonl or HF DatasetDict on disk).
2. Chunk with token-aware sliding window.
3. Generate diverse queries per chunk (LLM call #1).
4. Filter with heuristics + optional LLM judge (LLM call #2).
5. Split train/dev by doc_id (never by row — leaking chunks of the same
   doc across splits makes evaluation look better than it is).
6. Save as a DatasetDict with the (anchor, positive) shape the bi-encoder
   trainer already consumes.

Splits are doc-level so a query and its positive can't end up in different
splits. ``preprocessing.json`` is written next to the dataset with every
knob recorded — same convention as the rest of the lab so the trainer can
audit provenance at load time.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from datasets import Dataset, DatasetDict, load_from_disk

from lievito_madre_ai_lab.pipelines.llm.base import LLMClient
from lievito_madre_ai_lab.pipelines.synthetic.checkpoint import CheckpointStore
from lievito_madre_ai_lab.pipelines.synthetic.chunking import (
    ChunkingConfig,
    chunk_documents,
)
from lievito_madre_ai_lab.pipelines.synthetic.filtering import (
    FilterConfig,
    JudgeConfig,
    filter_pairs,
)
from lievito_madre_ai_lab.pipelines.synthetic.multi_hop import (
    MultiHopConfig,
    build_multi_hop_groups,
    compute_n_groups,
    generate_multi_hop_queries_for_groups,
)
from lievito_madre_ai_lab.pipelines.synthetic.query_generation import (
    QueryGenConfig,
    generate_queries_for_chunks,
)
from lievito_madre_ai_lab.shared.preprocessing import save_preprocessing_meta


CHECKPOINT_DIR = "_checkpoints"


@dataclass
class PipelineConfig:
    """One-stop config for the synthetic-pair pipeline.

    Three input modes are auto-detected from ``input_path`` and
    ``input_format``:

    - ``jsonl``: newline-delimited JSON with ``{id, text}`` per line.
    - ``hf_dataset``: a directory loadable via :func:`datasets.load_from_disk`.
    - ``auto`` (default): jsonl if the path ends in ``.jsonl`` / ``.json``,
      otherwise treat as a HF dataset directory.
    """
    input_path: str
    output_dir: str
    input_format: str = "auto"  # "auto" | "jsonl" | "hf_dataset"
    text_column: str = "text"
    id_column: str = "id"

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    query_gen: QueryGenConfig = field(default_factory=QueryGenConfig)
    multi_hop: MultiHopConfig = field(default_factory=MultiHopConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    # Split strategy — doc-level, never row-level.
    train_ratio: float = 0.9
    dev_ratio: float = 0.05
    test_ratio: float = 0.05
    seed: int = 42

    # Cap for quick iteration / dry runs. None = process the whole corpus.
    max_documents: int | None = None

    # Intermediate artefacts (debugging + resumability). Written under
    # ``output_dir`` when set.
    save_chunks: bool = True
    save_raw_pairs: bool = True

    def __post_init__(self) -> None:
        total = self.train_ratio + self.dev_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train/dev/test ratios must sum to 1.0; got {total:.4f}"
            )


def _iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Stream a JSONL file as dicts.

    Strict by design: one complete JSON object per line. We don't fall back
    to whole-file JSON parsing because that would force the entire corpus
    into RAM, which defeats the point of JSONL for a 10k+ document run.

    Bad lines raise — silent skipping hides upstream bugs in document-prep
    scripts where it really matters. The error message points at the
    most common cause (pretty-printed JSON that should be flattened with
    ``jq -c``) so a misformatted input fails actionable rather than cryptic.
    """
    with open(path, "r") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                hint = ""
                # The classic "line 1 char 1 is `{`" failure mode: file is a
                # pretty-printed JSON object/array, not JSONL.
                if lineno == 1 and line in ("{", "["):
                    hint = (
                        f"\n  Hint: {path} looks like pretty-printed JSON, "
                        f"not JSONL. Convert with one command:\n"
                        f"    jq -c '.[]?  // .' {path} > {path}.tmp && mv {path}.tmp {path}\n"
                        f"  (handles both single-object and JSON-array shapes; "
                        f"emits one document per line.)"
                    )
                raise ValueError(
                    f"{path}:{lineno}: invalid JSON: {exc}{hint}"
                ) from None


def _load_documents(cfg: PipelineConfig) -> list[dict]:
    """Materialise the document list.

    We don't stream further than this because chunking is cheap and the
    downstream batch size logic wants ``len(chunks)`` upfront.
    """
    fmt = cfg.input_format
    if fmt == "auto":
        fmt = "jsonl" if cfg.input_path.endswith((".jsonl", ".json")) else "hf_dataset"

    if fmt == "jsonl":
        docs = list(_iter_jsonl(cfg.input_path))
    elif fmt == "hf_dataset":
        ds = load_from_disk(cfg.input_path)
        if isinstance(ds, DatasetDict):
            # Concatenate all splits — the pipeline re-splits by doc_id anyway.
            docs = []
            for split in ds.values():
                docs.extend(split.to_list())
        else:
            docs = ds.to_list()
    else:
        raise ValueError(f"input_format must be 'auto' | 'jsonl' | 'hf_dataset'; got {fmt!r}")

    if cfg.max_documents is not None:
        docs = docs[: cfg.max_documents]
    return docs


def _split_by_doc_id(rows: list[dict], cfg: PipelineConfig) -> DatasetDict:
    """Deterministic doc-level split.

    Hash the doc_id with the seed to bucket each doc into train/dev/test —
    same doc always lands in the same split across runs with the same
    seed, even if upstream document order changes. Row-level splits leak
    chunks of one document across train and dev, which inflates eval
    scores by ~5-10 nDCG points on retrieval benchmarks.
    """
    import hashlib

    unique_doc_ids = sorted({r["doc_id"] for r in rows})

    def bucket(doc_id: str) -> str:
        # Stable hash → uniform [0, 1) per doc.
        h = hashlib.sha256(f"{cfg.seed}::{doc_id}".encode()).digest()
        u = int.from_bytes(h[:8], "big") / 2**64
        if u < cfg.train_ratio:
            return "train"
        if u < cfg.train_ratio + cfg.dev_ratio:
            return "dev"
        return "test"

    doc_to_split = {d: bucket(d) for d in unique_doc_ids}
    buckets: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for r in rows:
        buckets[doc_to_split[r["doc_id"]]].append(r)

    out = DatasetDict()
    for name, items in buckets.items():
        if not items:
            continue
        # Strip auxiliary columns at write time so the dataset matches
        # `pair` shape exactly (validator expects 2 string columns). We
        # keep doc_id/chunk_id/style/judge_score in the *raw* artefact for
        # debugging but not in the final training dataset.
        pairs = [{"anchor": r["anchor"], "positive": r["positive"]} for r in items]
        out[name] = Dataset.from_list(pairs)
    return out


async def run_pipeline(
    cfg: PipelineConfig,
    *,
    gen_client: LLMClient,
    judge_client: LLMClient | None = None,
) -> DatasetDict:
    """Run all stages and return the resulting DatasetDict.

    ``judge_client`` defaults to ``gen_client`` when the judge is enabled,
    because most users will run both stages against the same model. Pass
    a separate client to use a stronger / cheaper model for one of the two
    (the judge usually warrants a stronger model than the generator).
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint stores — one per LLM stage. Resuming is automatic when
    # these files already exist; ``--fresh`` on the CLI clears them.
    ckpt_dir = out_dir / CHECKPOINT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    single_hop_ckpt = CheckpointStore(ckpt_dir / "single_hop.jsonl", key_field="chunk_id")
    multi_hop_ckpt = CheckpointStore(ckpt_dir / "multi_hop.jsonl", key_field="group_id")
    judge_ckpt = CheckpointStore(ckpt_dir / "judge.jsonl", key_field="row_id")

    print(f"[1/6] Loading documents from {cfg.input_path!r} …")
    documents = _load_documents(cfg)
    print(f"      → {len(documents)} documents")

    print(f"[2/6] Chunking (size={cfg.chunking.chunk_tokens}, "
          f"overlap={cfg.chunking.overlap_tokens}) …")
    chunks = list(chunk_documents(
        documents, cfg=cfg.chunking,
        text_column=cfg.text_column, id_column=cfg.id_column,
    ))
    print(f"      → {len(chunks)} chunks")
    if cfg.save_chunks:
        chunks_path = out_dir / "chunks.jsonl"
        with open(chunks_path, "w") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")

    print(f"[3/6] Generating single-chunk queries (styles={cfg.query_gen.styles}) …")
    if single_hop_ckpt.n_done() > 0:
        print(f"      [resume] {single_hop_ckpt.n_done()} chunks previously processed")
    single_hop_pairs = await generate_queries_for_chunks(
        chunks, client=gen_client, cfg=cfg.query_gen, checkpoint=single_hop_ckpt,
    )
    print(f"      → {len(single_hop_pairs)} single-hop candidate pairs")

    if cfg.multi_hop.enabled:
        # Convert ``target_share`` (fraction of final rows that should be
        # multi-hop) into an absolute group count using the corpus stats.
        n_styles = len(cfg.query_gen.styles)
        target_groups = compute_n_groups(
            n_chunks=len(chunks),
            target_share=cfg.multi_hop.target_share,
            n_styles=n_styles,
            k=cfg.multi_hop.k,
        )
        print(f"[4/6] Generating multi-hop queries "
              f"(k={cfg.multi_hop.k}, target_share={cfg.multi_hop.target_share}, "
              f"target_groups≈{target_groups}, stride={cfg.multi_hop.stride}) …")
        groups = build_multi_hop_groups(
            chunks,
            k=cfg.multi_hop.k,
            stride=cfg.multi_hop.stride,
            n_groups=target_groups,
            seed=cfg.multi_hop.seed,
        )
        if target_groups > 0 and len(groups) < target_groups:
            print(f"      [warn] only {len(groups)} adjacent windows available; "
                  f"target was {target_groups}. Multi-hop share will be lower than "
                  f"{cfg.multi_hop.target_share:.0%}.")
        else:
            print(f"      → {len(groups)} adjacent-window groups")
        if multi_hop_ckpt.n_done() > 0:
            print(f"      [resume] {multi_hop_ckpt.n_done()} groups previously processed")
        multi_hop_pairs = await generate_multi_hop_queries_for_groups(
            groups, client=gen_client, cfg=cfg.multi_hop, checkpoint=multi_hop_ckpt,
        )
        print(f"      → {len(multi_hop_pairs)} multi-hop candidate rows "
              f"({len(multi_hop_pairs) // max(cfg.multi_hop.k, 1)} accepted queries × "
              f"{cfg.multi_hop.k} positives)")
    else:
        print("[4/6] Multi-hop disabled — skipping.")
        multi_hop_pairs = []

    raw_pairs = single_hop_pairs + multi_hop_pairs
    if cfg.save_raw_pairs:
        raw_path = out_dir / "raw_pairs.jsonl"
        with open(raw_path, "w") as f:
            for r in raw_pairs:
                f.write(json.dumps(r) + "\n")

    print(f"[5/6] Filtering (heuristic + judge.enabled={cfg.judge.enabled}) …")
    if cfg.judge.enabled and judge_ckpt.n_done() > 0:
        print(f"      [resume] {judge_ckpt.n_done()} rows previously judged")
    judge = judge_client if judge_client is not None else gen_client
    filtered = await filter_pairs(
        raw_pairs,
        cfg=cfg.filter,
        judge_cfg=cfg.judge,
        judge_client=judge if cfg.judge.enabled else None,
        judge_checkpoint=judge_ckpt if cfg.judge.enabled else None,
    )
    n_mh = sum(1 for r in filtered if r.get("style") == "multi_hop")
    print(f"      → {len(filtered)} pairs survived "
          f"({len(filtered) / max(len(raw_pairs), 1):.1%} retention; "
          f"{n_mh} multi-hop)")

    print(f"[6/6] Splitting by doc_id and saving → {cfg.output_dir} …")
    datasets = _split_by_doc_id(filtered, cfg)
    for name, split in datasets.items():
        print(f"      [{name}] {len(split)} pairs")
    datasets.save_to_disk(str(out_dir))

    save_preprocessing_meta(
        out_dir,
        source=cfg.input_path,
        input_format=cfg.input_format,
        chunking=cfg.chunking.__dict__,
        query_gen={"styles": cfg.query_gen.styles, "batch_size": cfg.query_gen.batch_size},
        multi_hop=cfg.multi_hop.__dict__,
        filter=cfg.filter.__dict__,
        judge=cfg.judge.__dict__,
        split_ratios={"train": cfg.train_ratio, "dev": cfg.dev_ratio, "test": cfg.test_ratio},
        seed=cfg.seed,
        n_documents=len(documents),
        n_chunks=len(chunks),
        n_single_hop_raw=len(single_hop_pairs),
        n_multi_hop_raw=len(multi_hop_pairs),
        n_raw_pairs=len(raw_pairs),
        n_filtered_pairs=len(filtered),
        n_multi_hop_filtered=n_mh,
    )
    return datasets


def run_pipeline_sync(
    cfg: PipelineConfig,
    *,
    gen_client: LLMClient,
    judge_client: LLMClient | None = None,
) -> DatasetDict:
    """Sync wrapper for CLI use."""
    return asyncio.run(run_pipeline(
        cfg, gen_client=gen_client, judge_client=judge_client,
    ))
