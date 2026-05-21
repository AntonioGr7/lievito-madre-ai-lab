#!/usr/bin/env python
"""Generate synthetic (anchor, positive) pairs for bi-encoder fine-tuning.

Reads a YAML config + a corpus of domain documents, runs the full pipeline
(chunk → LLM query generation → heuristic + judge filter → doc-level split),
and saves a HF DatasetDict ready to feed `train_bi_encoder.py` or
`mine_hard_negatives.py`.

Examples
--------
# 1. Production run from a jsonl corpus
python scripts/pipelines/generate_bi_encoder_pairs.py \\
    --config examples/embedding_bi_encoder/custom_pairs/configs/sec_edgar_pairs.yaml

# 2. Dry-run on 50 docs without the judge (cheaper iteration)
python scripts/pipelines/generate_bi_encoder_pairs.py \\
    --config examples/embedding_bi_encoder/custom_pairs/configs/sec_edgar_pairs.yaml \\
    --max-documents 50 \\
    --no-judge

# 3. Override input / output paths from the CLI
python scripts/pipelines/generate_bi_encoder_pairs.py \\
    --config examples/embedding_bi_encoder/custom_pairs/configs/sec_edgar_pairs.yaml \\
    --input data/raw/finance-corpus.jsonl \\
    --output data/processed/finance-pairs
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from lievito_madre_ai_lab.pipelines.llm.providers import OpenAIClient
from lievito_madre_ai_lab.pipelines.embedding.synthetic.chunking import ChunkingConfig
from lievito_madre_ai_lab.pipelines.embedding.synthetic.filtering import FilterConfig, JudgeConfig
from lievito_madre_ai_lab.pipelines.embedding.synthetic.multi_hop import MultiHopConfig
from lievito_madre_ai_lab.pipelines.embedding.synthetic.pipeline import (
    PipelineConfig,
    run_pipeline_sync,
)
from lievito_madre_ai_lab.pipelines.embedding.synthetic.query_generation import QueryGenConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, help="YAML config file")
    p.add_argument("--input", default=None, help="Override input_path from config")
    p.add_argument("--output", default=None, help="Override output_dir from config")
    p.add_argument(
        "--max-documents", type=int, default=None,
        help="Override max_documents (handy for dry runs).",
    )
    p.add_argument(
        "--no-judge", action="store_true",
        help="Disable the LLM judge stage (skip the second LLM pass). "
             "Roughly halves the LLM spend; quality drops noticeably.",
    )
    p.add_argument(
        "--no-multi-hop", action="store_true",
        help="Disable the multi-hop (multi-chunk) query generation stage. "
             "Falls back to the original single-chunk-only pipeline.",
    )
    p.add_argument(
        "--fresh", action="store_true",
        help="Delete the checkpoint directory (output_dir/_checkpoints) before "
             "running, forcing every LLM call to be redone. Resume is the "
             "default — pass this only when you want a clean run.",
    )
    p.add_argument(
        "--gen-model", default=None,
        help="Override the generation model name (gen_llm.model in YAML).",
    )
    p.add_argument(
        "--judge-model", default=None,
        help="Override the judge model name (judge_llm.model in YAML).",
    )
    return p.parse_args()


def _load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _build_client(spec: dict) -> OpenAIClient:
    """Construct an OpenAIClient from a YAML sub-section."""
    return OpenAIClient(
        model=spec["model"],
        api_key=spec.get("api_key"),
        base_url=spec.get("base_url"),
        temperature=float(spec.get("temperature", 0.7)),
        max_tokens=int(spec.get("max_tokens", 1024)),
        max_concurrency=int(spec.get("max_concurrency", 16)),
        timeout=float(spec.get("timeout", 60)),
        max_retries=int(spec.get("max_retries", 4)),
    )


def main() -> None:
    args = parse_args()
    raw = _load_yaml(args.config)

    # CLI overrides (kept narrow on purpose — anything more complex belongs
    # in a separate config file rather than as a long string of flags).
    if args.input:
        raw["input_path"] = args.input
    if args.output:
        raw["output_dir"] = args.output
    if args.max_documents is not None:
        raw["max_documents"] = args.max_documents
    if args.no_judge:
        raw.setdefault("judge", {})["enabled"] = False
    if args.no_multi_hop:
        raw.setdefault("multi_hop", {})["enabled"] = False
    if args.gen_model:
        raw.setdefault("gen_llm", {})["model"] = args.gen_model
    if args.judge_model:
        raw.setdefault("judge_llm", {})["model"] = args.judge_model

    # Build the PipelineConfig — every nested section becomes its own
    # dataclass so YAML stays human-editable but Python keeps strict typing.
    cfg = PipelineConfig(
        input_path=raw["input_path"],
        output_dir=raw["output_dir"],
        input_format=raw.get("input_format", "auto"),
        text_column=raw.get("text_column", "text"),
        id_column=raw.get("id_column", "id"),
        chunking=ChunkingConfig(**raw.get("chunking", {})),
        query_gen=QueryGenConfig(**raw.get("query_gen", {})),
        multi_hop=MultiHopConfig(**raw.get("multi_hop", {})),
        filter=FilterConfig(**raw.get("filter", {})),
        judge=JudgeConfig(**raw.get("judge", {})),
        train_ratio=float(raw.get("train_ratio", 0.9)),
        dev_ratio=float(raw.get("dev_ratio", 0.05)),
        test_ratio=float(raw.get("test_ratio", 0.05)),
        seed=int(raw.get("seed", 42)),
        max_documents=raw.get("max_documents"),
        save_chunks=bool(raw.get("save_chunks", True)),
        save_raw_pairs=bool(raw.get("save_raw_pairs", True)),
    )

    if args.fresh:
        # Drop the per-stage checkpoint files but leave the rest of the
        # output_dir alone — chunks.jsonl and the final DatasetDict are
        # cheap to regenerate but more importantly are not what the user
        # is trying to invalidate when they pass --fresh.
        ckpt_dir = Path(cfg.output_dir) / "_checkpoints"
        if ckpt_dir.exists():
            print(f"[--fresh] removing {ckpt_dir}")
            shutil.rmtree(ckpt_dir)

    gen_client = _build_client(raw["gen_llm"])
    judge_client = (
        _build_client(raw["judge_llm"])
        if cfg.judge.enabled and "judge_llm" in raw
        else None
    )

    run_pipeline_sync(cfg, gen_client=gen_client, judge_client=judge_client)


if __name__ == "__main__":
    main()
