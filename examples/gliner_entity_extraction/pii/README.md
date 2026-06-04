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

## Label prompts (natural-language entity types)

GLiNER's label encoder reads entity types as natural language, so `"medical record number"` embeds far better than the raw identifier `medical_record_number`. Labels are therefore **auto-humanized** to prompts — `snake_case`/`kebab-case`/`camelCase` → spaced lower-case — and the *same* prompt is used at **train, eval, and serve** time (training on one string and querying with another would hurt, not help). This is generic: any future dataset gets sensible prompts with no per-label table.

The only labels the humanizer can't fix are concatenated all-caps identifiers with no boundary (`GIVENNAME` → `"givenname"`). Spell those out with an explicit override in the config — it always wins over the auto form:

```yaml
gliner:
  label_aliases:
    GIVENNAME: "given name"
    PASSPORTNUM: "passport number"
```

## Zero-shot baseline (did fine-tuning earn its keep?)

Before trusting a fine-tune, check it actually beats an off-the-shelf model on the same test split. [`baseline_zeroshot.py`](../../../scripts/gliner_entity_extraction/baseline_zeroshot.py) evaluates a pretrained GLiNER **with no training** using the identical scoring, threshold tuning, and chunking, then prints a side-by-side delta against the fine-tune's `test_metrics.json`:

```bash
# Baseline = the base model the fine-tune started from (most direct "did training help?"):
python scripts/gliner_entity_extraction/baseline_zeroshot.py \
    --config examples/gliner_entity_extraction/pii/configs/a100_nemotron.yaml

# Or a dedicated off-the-shelf PII model:
python scripts/gliner_entity_extraction/baseline_zeroshot.py \
    --config examples/gliner_entity_extraction/pii/configs/a100_nemotron.yaml \
    --model knowledgator/gliner-pii-base-v1.0
```

Output (writes `baseline_metrics.json` next to the fine-tuned model):

```
================================================================
Zero-shot baseline (knowledgator/gliner-pii-base-v1.0) vs fine-tune
================================================================
metric                    baseline   finetuned         Δ
----------------------------------------------------------------
closed F1                   0.6120      0.8110   +0.1990
zero-shot F1                0.5380      0.5610   +0.0230
================================================================
```

> Note: when the baseline and fine-tune use different backbones (e.g. an off-the-shelf PII model vs your DeBERTa fine-tune), their word splitters can chunk long docs slightly differently, so the comparison is most exact when `--model` is the same base model the fine-tune started from (the default).

## Decision-threshold tuning

After training, the train script sweeps the decision threshold on the **validation** split, reports the final test metrics at the F1-optimal cutoff, and writes that threshold into the model's `preprocessing.json`. `serve.py` / `GLiNERPredictor` then default to it automatically — no need to hand-pick `0.5`. The full precision/recall curve is saved to `test_metrics.json` (`threshold_curve`). Override at inference with `--threshold` if you want a different precision/recall trade-off.
