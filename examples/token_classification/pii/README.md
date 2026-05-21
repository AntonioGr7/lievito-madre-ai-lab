# PII Token Classification

Fine-tune an encoder for token-level PII detection (BIO scheme). Two data sources are interchangeable:

- **`prepare_openpii.py`** — [ai4privacy/open-pii-masking-500k-ai4privacy](https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy) (8 languages, 19 PII entity types — default).
- **`prepare_nemotron_pii.py`** — [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII).

Default backbone: `microsoft/mdeberta-v3-base` (despite the historical `pii_mbert.yaml` filename). Metrics: entity-level `precision`, `recall`, `f1`, `accuracy` via `seqeval`. Long inputs are split into overlapping `max_length`-token chunks with `stride=128` so a 2048-token document doesn't lose three-quarters of its supervision to truncation.

## Run

```bash
bash examples/token_classification/pii/run.sh
```

Or step by step:

```bash
# Pick one prep script:
python examples/token_classification/pii/prepare_openpii.py
# or:
python examples/token_classification/pii/prepare_nemotron_pii.py

python scripts/token_classification/train_token_classification.py \
    --config examples/token_classification/pii/configs/pii_mbert.yaml
```
