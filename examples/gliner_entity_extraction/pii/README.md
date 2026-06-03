# GLiNER PII Entity Extraction

Fine-tune [GLiNER](https://github.com/urchade/GLiNER) for PII entity extraction. Two data sources are interchangeable — pick the one whose label inventory matches your downstream need:

- **`prepare_openpii.py`** — [ai4privacy/open-pii-masking-500k-ai4privacy](https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy) (default).
- **`prepare_nemotron.py`** — [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII).

GLiNER 0.2.x requires `transformers<5.2.0`, which conflicts with the encoder/decoder/vision groups. Install in its own virtualenv:

```bash
pip install -e ".[gliner]"
```

## Run

```bash
bash examples/gliner_entity_extraction/pii/run.sh
```

Or step by step:

```bash
# Pick one prep script:
python examples/gliner_entity_extraction/pii/dataset/prepare_openpii.py --out-dir data/processed/pii-gliner
# or:
python examples/gliner_entity_extraction/pii/dataset/prepare_nemotron.py --out-dir data/processed/pii-gliner

python scripts/gliner_entity_extraction/train_gliner.py \
    --config examples/gliner_entity_extraction/pii/configs/pii_gliner.yaml
```

## Config variants

Five recipes live in [configs/](configs/), all consuming the same char-offset dataset contract with different model / hardware tradeoffs:

| Config              | When to use |
|---------------------|-------------|
| `smoke.yaml`        | Quick wiring check (1 epoch, tiny subset). |
| `t4_small.yaml`     | T4-class GPU (16 GB) — DeBERTa-v3-small, fp16. |
| `a10_medium.yaml`   | A10-class (24 GB) — DeBERTa-v3-base, bf16, full FT. |
| `pii_gliner.yaml`   | Default reference recipe (DeBERTa-v3 multitask, LoRA). |
| `a100_nemotron.yaml`| **A100 (40 GB) full fine-tuning on Nemotron-PII** — frontier-SOTA `modern-gliner-bi-large-v1.0` (ModernBERT bi-encoder, 8192 ctx, chunking off). |

Swap configs by changing the `--config` flag — the dataset contract is the same across all of them. Note `a100_nemotron.yaml` expects the **Nemotron** dataset, so prep with `prepare_nemotron.py` (the others default to OpenPII).

```bash
# Frontier-SOTA full FT on an A100 (40 GB):
python examples/gliner_entity_extraction/pii/dataset/prepare_nemotron.py \
    --out-dir data/processed/nemotron-gliner
python scripts/gliner_entity_extraction/train_gliner.py \
    --config examples/gliner_entity_extraction/pii/configs/a100_nemotron.yaml
```

> **Smoke-test the bi-encoder first.** `a100_nemotron.yaml` is the only recipe that fine-tunes a *bi-encoder* backbone (two encoders), which exercises the multi-encoder gradient-checkpointing path. Do one short run with `--max-train-samples 200 --max-eval-samples 100 --max-test-samples 100` and confirm `eval_f1` climbs off zero before committing the GPU.

## Decision-threshold tuning

After training, the train script sweeps the decision threshold on the **validation** split, reports the final test metrics at the F1-optimal cutoff, and writes that threshold into the model's `preprocessing.json`. `serve.py` / `GLiNERPredictor` then default to it automatically — no need to hand-pick `0.5`. The full precision/recall curve is saved to `test_metrics.json` (`threshold_curve`). Override at inference with `--threshold` if you want a different precision/recall trade-off.
