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
| `a100_nemotron.yaml`| **A100 (40 GB) full fine-tuning on Nemotron-PII** — frontier-SOTA `modern-gliner-bi-large-v1.0` (ModernBERT bi-encoder, full FT). Chunking off: rows >2048 word-tokens truncate. |
| `a100_nemotron_longdoc.yaml`| Same as above but **sliding-window chunking ON** — use when many rows exceed the 2048-token `max_len` (long docs are split into overlapping windows instead of truncated). |
| `a100_nemotron_generalist.yaml`| **Recommended default.** Keeps **zero-shot generalization** (frozen label encoder + zero-shot-aware selection) **and** handles long docs (chunking on). See below. |

Swap configs by changing the `--config` flag — the dataset contract is the same across all of them. Note `a100_nemotron.yaml` expects the **Nemotron** dataset, so prep with `prepare_nemotron.py` (the others default to OpenPII).

```bash
# Frontier-SOTA full FT on an A100 (40 GB):
python examples/gliner_entity_extraction/pii/dataset/prepare_nemotron.py \
    --out-dir data/processed/nemotron-gliner
python scripts/gliner_entity_extraction/train_gliner.py \
    --config examples/gliner_entity_extraction/pii/configs/a100_nemotron.yaml
```

## Smoke-test on a small GPU before the A100

`a100_nemotron.yaml` is the only recipe that fine-tunes a *bi-encoder* backbone (two encoders + the multi-encoder gradient-checkpointing path), so validate the wiring cheaply before committing the A100. Work outward in tiers — each catches more, the last needs only ~4 GB.

**Tier 0 — pure logic, no GPU (seconds).** Runs the fast unit suite (label humanization, threshold sweep, gold-filtering, dataset contract):

```bash
pip install pytest
pytest tests/encoder/gliner_entity_extraction -q       # smoke tests stay skipped unless RUN_GLINER_SMOKE=1
```

**Tier 1 — full pipeline on the tiny fixture (<60 s, CPU or any GPU).** Exercises the whole train→tune-threshold→save→eval path end-to-end with a small uni-encoder, on the bundled fixture:

```bash
python scripts/gliner_entity_extraction/train_gliner.py \
    --config examples/gliner_entity_extraction/pii/configs/smoke.yaml \
    --max-train-samples 6 --max-eval-samples 2 --max-test-samples 2
```

**Tier 2 — the A100 code paths on 4 GB (a couple of minutes).** [`smoke_modern_bi.yaml`](configs/smoke_modern_bi.yaml) mirrors the A100 recipe (bi-encoder, chunking off, label humanization, gradient checkpointing) but on the smallest modern bi-encoder (`modern-gliner-bi-base`, 194M) with LoRA so it fits 4 GB. Prep a tiny streamed slice of Nemotron, then run:

```bash
# Tiny slice; empty holdout keeps prep from failing if a rare holdout label is absent.
python examples/gliner_entity_extraction/pii/dataset/prepare_nemotron.py \
    --out-dir data/processed/nemotron-smoke --limit 500 --holdout-types

python scripts/gliner_entity_extraction/train_gliner.py \
    --config examples/gliner_entity_extraction/pii/configs/smoke_modern_bi.yaml \
    --max-train-samples 25 --max-eval-samples 20 --max-test-samples 20

# And the baseline script (loads the base model zero-shot — inference only, fits easily):
python scripts/gliner_entity_extraction/baseline_zeroshot.py \
    --config examples/gliner_entity_extraction/pii/configs/smoke_modern_bi.yaml \
    --max-eval-samples 20 --max-test-samples 20
```

What "green" looks like: training reaches the save step without an OOM or a gradient-checkpointing `AttributeError`, `eval_f1` is printed (it'll be low — that's fine, it's 25 samples), a tuned threshold is logged, and `outputs/_smoke_modern_bi/smoke/final/test_metrics.json` is written. Then scale to `a100_nemotron.yaml` (full FT, large backbone) with confidence.

> **Troubleshooting 4 GB:** if you hit OOM, drop `per_device_eval_batch_size` to 1 and `gliner.sampling.max_types` to 12. If loss goes `NaN`, your card is likely Turing (fp16-only) and ModernBERT dislikes fp16 — set `precision: fp32` (tighter on memory) or run this tier on CPU; the A100 itself uses bf16 and is unaffected. Tier 2 covers everything except full-FT optimizer memory, which only the A100 run can exercise.

## Label prompts (natural-language entity types)

GLiNER's label encoder reads entity types as natural language, so `"medical record number"` embeds far better than the raw identifier `medical_record_number`. Labels are therefore **auto-humanized** to prompts — `snake_case`/`kebab-case`/`camelCase` → spaced lower-case — and the *same* prompt is used at **train, eval, and serve** time (training on one string and querying with another would hurt, not help). This is generic: any future dataset gets sensible prompts with no per-label table.

The only labels the humanizer can't fix are concatenated all-caps identifiers with no boundary (`GIVENNAME` → `"givenname"`). Spell those out with an explicit override in the config — it always wins over the auto form:

```yaml
gliner:
  label_aliases:
    GIVENNAME: "given name"
    PASSPORTNUM: "passport number"
```

## Keeping zero-shot generalization (the generalist recipe)

A GLiNER fine-tune is only worth more than a plain token-classifier if it **keeps its zero-shot ability** — extracting label types it never trained on. Naive full fine-tuning destroys that: it overfits the prompt space to your trained labels, and zero-shot F1 on held-out labels collapses. [`a100_nemotron_generalist.yaml`](configs/a100_nemotron_generalist.yaml) is built to retain it:

1. **`freeze_labels_encoder: true`** — the dominant lever. The bi-encoder's label encoder maps a label prompt → embedding; full-FT'ing it is what corrupts the prompt space. Frozen, the prompt space stays exactly as the base model's, so unseen labels still embed sensibly — while the text encoder + span head still fully fine-tune.
2. **`monitor_zeroshot: true`** — measures held-out-label F1 every eval and selects the checkpoint with the best **combined** score, `eval_generalist_f1` (harmonic mean of closed-set and zero-shot F1). The harmonic mean means a model can't win by acing closed-set while collapsing zero-shot.
3. **Conservative training + label dropout** so the text encoder drifts less.

It also keeps **sliding-window chunking on** so the standard trainer never silently truncates long documents (the [longdoc](#config-variants) behavior is folded in). Net: a default that neither overfits labels nor drops long-doc entities.

LoRA is **not** required — with the label encoder frozen, full FT keeps the best closed-set quality, and the monitor tells you if the text encoder is still drifting. Only then turn on the optional `peft` block as an extra clamp.

This needs validation to carry held-out-label gold, so prep with `--val-all-labels` into a dedicated dir:

```bash
python examples/gliner_entity_extraction/pii/dataset/prepare_nemotron.py \
    --out-dir data/processed/nemotron-gliner-genz --val-all-labels
python scripts/gliner_entity_extraction/train_gliner.py \
    --config examples/gliner_entity_extraction/pii/configs/a100_nemotron_generalist.yaml
```

Watch `eval_f1`, `eval_zeroshot_f1`, and `eval_generalist_f1` in the logs — the gap between the first two is exactly the generalization you're spending. (Without `--val-all-labels`, monitoring auto-disables with a warning and selection falls back to closed-set F1.)

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
