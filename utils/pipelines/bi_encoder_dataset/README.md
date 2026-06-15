# bi_encoder_dataset — synthetic bi-encoder training data

An LLM-driven pipeline that builds `(anchor, positive)` bi-encoder training pairs from a corpus
of domain documents, using an OpenAI-compatible LLM (cloud, or a local vLLM/Ollama server via
`base_url`). Output lands as a HuggingFace `DatasetDict` that feeds bi-encoder training in the
separate `embedding-bi-encoder` project.

This is a lab utility (under `utils/pipelines/`), not a model-training use case — but it is still
a self-contained, installable project. It carries its own vendored copy of the lab's
preprocessing helpers under [`bi_encoder_dataset/shared/`](bi_encoder_dataset/shared/).

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # set OPENAI_API_KEY (or point base_url at a local server)
```

## Quickstart

```bash
python scripts/generate_bi_encoder_pairs.py --help
```

See [scripts/README.md](scripts/README.md) for the full guide (chunking, query generation,
multi-hop, filtering/judging, resuming, and the worked SEC-EDGAR example).

## Layout

```
bi_encoder_dataset/   importable package
  llm/         OpenAI-compatible async client + protocol
  synthetic/   chunking, query generation, multi-hop, filtering, orchestration
  prompts/     editable YAML prompt templates
  shared/      vendored preprocessing helpers
scripts/     CLI entry points
examples/    worked SEC-EDGAR example
data/        generated datasets (gitignored)
```
