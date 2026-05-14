# lievito-madre-ai-lab
Lievito Madre isn't just an ingredient; it is a living, evolving ecosystem of knowledge. Here is where we cook.

---

## Encoder

Encoder models (BERT, RoBERTa, DeBERTa, XLM-R, ModernBERT, …) fine-tuned with the HuggingFace `Trainer`. The lab currently ships two task families: **text classification** (single-label, soon multi-label / regression) and **token classification** (NER, PII detection).

Every run is driven by a single YAML file that loads into the [`TrainConfig`](lievito_madre_ai_lab/shared/config.py) dataclass. Each field maps 1-to-1 to a HuggingFace `TrainingArguments`, `from_pretrained`, or `DataCollator` parameter — no hidden translation, no defaults overridden behind your back.

### Configuration

The tables below cover every field accepted by a YAML config. Defaults reflect what `TrainConfig()` ships with; the **Best practice** column says when to change them.

#### Data & model

| Field | Default | What it does | Notes |
|---|---|---|---|
| `processed_dir` | `data/processed/emotion` | Path to the tokenized Arrow dataset produced by `prepare_dataset.py`. | Required for any real run. |
| `model_name` | `answerdotai/ModernBERT-base` | HuggingFace model id or local path. | English: `ModernBERT-base/large`, `deberta-v3-base/large`. Multilingual: `microsoft/mdeberta-v3-base`, `xlm-roberta-base/large`. |
| `output_dir` | `outputs/run` | Where checkpoints and the final model are saved. | |
| `experiment_id` | `null` | If set, appends to `output_dir` → `outputs/<run>/<experiment_id>`. | Handy for hyperparameter sweeps. |
| `attn_implementation` | `sdpa` | Attention kernel. | See [Choosing the attention implementation](#choosing-the-attention-implementation). |

#### Core hyperparameters

| Field | Default | What it does | Best practice |
|---|---|---|---|
| `num_train_epochs` | `3` | Total training epochs. | 2-4 for fine-tuning; rely on `early_stopping_patience` rather than over-running. |
| `per_device_train_batch_size` | `32` | Batch size per GPU. | Maximize to fill GPU memory; combine with `gradient_accumulation_steps` for effective batch ≥ 32. |
| `per_device_eval_batch_size` | `64` | Eval batch size (no gradients → can be larger). | 2× train batch is a safe default. |
| `gradient_accumulation_steps` | `1` | Optimizer-step interval. Effective batch = `per_device × accum × world_size`. | Use to simulate larger batches when memory-bound. |
| `learning_rate` | `2e-5` | Peak LR after warmup. | `2e-5` for base models; `1e-5` for large/XL; `5e-5` for distilled. |
| `weight_decay` | `0.01` | AdamW L2 regularization. | `0.01` is the BERT-paper default — works almost everywhere. |
| `warmup_ratio` | `0.1` | Fraction of steps spent linearly warming up LR. | `0.06`-`0.1` for fine-tuning; `0` only if dataset is huge. |
| `max_grad_norm` | `1.0` | Gradient clipping threshold. | Keep at `1.0` unless you see divergence. |
| `lr_scheduler_type` | `linear` | LR schedule. | `linear` is the BERT default; try `cosine` for >5 epochs. Options: `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup`. |
| `optim` | `adamw_torch_fused` | Optimizer. | `adamw_torch_fused` is ~15-20% faster than `adamw_torch` on PyTorch 2.x. Other useful options: `adamw_8bit` (bitsandbytes, big memory savings), `adafactor`. |
| `seed` | `42` | Random seed for shuffling, init, dropout. | Set explicitly for reproducibility. |

#### Checkpoint & evaluation

| Field | Default | What it does | Notes |
|---|---|---|---|
| `eval_strategy` | `epoch` | When to run eval. | Options: `no`, `steps`, `epoch`. |
| `save_strategy` | `epoch` | When to checkpoint. | Must match `eval_strategy` when `load_best_model_at_end=true`. |
| `load_best_model_at_end` | `true` | Reload the best checkpoint before the final save. | |
| `metric_for_best_model` | `f1` | Metric used to rank checkpoints. | Must be one of the keys returned by `compute_metrics`. |
| `greater_is_better` | `true` | Whether higher metric is better. | `false` for loss, perplexity. |
| `save_total_limit` | `2` | Max checkpoints kept on disk. | Older checkpoints deleted automatically. |
| `early_stopping_patience` | `null` | Wires up `EarlyStoppingCallback` when set to an integer. | Stops training if the eval metric doesn't improve for N consecutive evals. `2`-`3` is typical. |

#### Compute & throughput

| Field | Default | What it does | Best practice |
|---|---|---|---|
| `fp16` | `false` | FP16 mixed precision (with loss scaling). | Use on pre-Ampere GPUs (sm < 80). Mutually exclusive with `bf16`. |
| `bf16` | `false` | BF16 mixed precision (no loss scaling). | Preferred on Ampere+/Hopper. Wider dynamic range than fp16, more stable. |
| `tf32` | `null` | TF32 for matmuls on Ampere+. | `true` for free ~10% speedup on Ampere+; `null` keeps the PyTorch default. |
| `gradient_checkpointing` | `false` | Recompute activations during backward. | Enable when memory-bound: ~40% memory saved for ~20% speed cost. |
| `torch_compile` | `false` | `torch.compile(model)` during training. | 1.3-1.6× steady-state speedup. Costs 30-60s warmup on the first step; can fragile-fail on unusual model code. |
| `train_sampling_strategy` | `random` | Training batch order. | `random` for reproducibility; `group_by_length` for variable-length sequences (NER, long-form text) — 15-30% faster. Also: `sequential`. |
| `dataloader_num_workers` | `2` | Parallel data-loading workers. | `2`-`8` keeps the GPU fed. Set higher on multi-CPU machines, `0` for debugging. |
| `pad_to_multiple_of` | `8` | Pad batch sequences to a multiple of N tokens. | `8` for fp16/bf16 tensor cores; `16` on Hopper; `null` disables. |

#### Logging & Weights & Biases

| Field | Default | What it does |
|---|---|---|
| `logging_steps` | `50` | Print / log training metrics every N steps. |
| `report_to` | `none` | Logger integration. Set to `wandb` to enable. Options: `none`, `wandb`, `tensorboard`, `mlflow`, etc. |
| `wandb_project` | `null` | W&B project name (required when `report_to: wandb`). |
| `wandb_run_name` | `null` | W&B run name (defaults to auto-generated). |
| `wandb_tags` | `[]` | List of W&B tags. |
| `wandb_notes` | `null` | Free-form notes attached to the W&B run. |

#### Choosing the attention implementation

`sdpa` is the default and the right answer in almost every case — PyTorch's `F.scaled_dot_product_attention` auto-routes to the fastest kernel available on the hardware:

| Hardware | What `sdpa` dispatches to |
|---|---|
| Ampere+ (sm ≥ 80: A100, H100, RTX 30/40, A6000) | Flash Attention 2 — fastest, lowest memory |
| Turing / Volta (sm 70-75: T4, V100, RTX 20) | Memory-efficient fused math kernel |
| CPU / MPS | Standard PyTorch implementation |

Use `flash_attention_2` to fail loudly if FA2 isn't available (e.g., to verify install). Use `eager` only for debugging — it's the slowest path.

#### Choosing the precision dtype

| GPU generation | Recommended |
|---|---|
| Ampere or newer (sm ≥ 80: A100, H100, RTX 30/40) | `bf16: true` + `tf32: true` |
| Turing / Volta (sm 70-75: T4, V100, RTX 20) | `fp16: true` |
| Older GPUs / CPU | (leave both false) |

`bf16` is preferred over `fp16` whenever available — same dynamic range as fp32, no loss scaling needed, more stable in practice.

#### Choosing the sampling strategy

| Strategy | When to use |
|---|---|
| `random` *(default)* | Standard fine-tuning; best when reproducibility matters or sequences are short and uniform. |
| `group_by_length` | Sequences vary widely (NER, multi-paragraph classification). Reduces padding waste; typical 15-30% training-time win. |
| `sequential` | Curriculum learning, debugging. Rarely the right choice for production. |

---

### Text Classification

Fine-tuning encoder models for sequence classification.

#### Quickstart

```bash
# 1. Prepare the dataset (downloads, tokenizes, saves to Arrow)
python scripts/text_classification/prepare_dataset.py

# 2. Train
python scripts/text_classification/train_text_classification.py \
    --config configs/encoder/text_classification/emotion_bert.yaml
```

The default dataset is `dair-ai/emotion` (6 emotion classes). The default model is `answerdotai/ModernBERT-base`.  
Metrics tracked at each epoch: `accuracy`, `f1` (weighted), and `f1_<class>` per label.  
Final test metrics are saved to `outputs/<run>/final/test_metrics.json`.

#### Resuming after a failure

Pass `--resume` to continue from the latest checkpoint automatically:

```bash
python scripts/text_classification/train_text_classification.py \
    --config configs/encoder/text_classification/emotion_bert.yaml \
    --resume
```

Or point at a specific checkpoint:

```bash
python scripts/text_classification/train_text_classification.py \
    --config configs/encoder/text_classification/emotion_bert.yaml \
    --resume outputs/emotion_bert/checkpoint-1000
```

#### Inference

Load the trained model for high-performance inference:

```python
from lievito_madre_ai_lab.encoder.text_classification.serve import TextClassificationPredictor

predictor = TextClassificationPredictor("outputs/emotion_bert/final")

# batch
results = predictor.predict(["I love this!", "I'm furious."])
# [{"label": "joy", "score": 0.97, "scores": {"joy": 0.97, "anger": 0.01, ...}}, ...]

# single
result = predictor.predict_one("What a lovely day.")
```

Run from the CLI:

```bash
python -m lievito_madre_ai_lab.encoder.text_classification.serve \
    outputs/emotion_bert/final "I love this" "I'm so angry"
```

Benchmark throughput:

```bash
python -m lievito_madre_ai_lab.encoder.text_classification.serve \
    outputs/emotion_bert/final --benchmark
```

The predictor automatically applies the best available optimisations for the detected hardware:

| Hardware | Optimisations applied |
|---|---|
| CUDA (Ampere+) | FP16 weights · Flash Attention 2 · `torch.compile` · BF16 autocast |
| CUDA (older) | FP16 weights · SDPA · `torch.compile` · FP16 autocast |
| CPU | FP32 weights · dynamic INT8 quantisation of Linear layers |

#### Switching model or dataset

Duplicate the YAML config and edit `model_name`, `processed_dir`, and `output_dir`:

```bash
cp configs/encoder/text_classification/emotion_bert.yaml \
   configs/encoder/text_classification/emotion_roberta.yaml
```

Then run with `--config configs/encoder/text_classification/emotion_roberta.yaml`.

#### Bringing your own data

The training script only requires a tokenized Arrow dataset with three columns: `input_ids`, `attention_mask`, and `labels`. There are two paths to get there.

**Path 1 — your files already have `text` and `label` columns** (CSV, JSON, or Parquet):

```bash
python scripts/text_classification/prepare_dataset.py \
    --source local \
    --local-path data/raw/my_dataset \
    --text-col my_text_column \
    --label-col my_label_column
```

**Path 2 — your data needs transformation first** (string labels → int, cleaning, custom splits, etc.):  
Write a script in `scripts/` that produces a `DatasetDict` and calls `tokenize_for_trainer`:

```python
from datasets import DatasetDict, Dataset
from lievito_madre_ai_lab.encoder.text_classification.dataset import tokenize_for_trainer

raw = DatasetDict({
    "train": Dataset.from_dict({"text": [...], "label": [...]}),
    "test":  Dataset.from_dict({"text": [...], "label": [...]}),
})

processed = tokenize_for_trainer(raw, model_name="bert-base-uncased")
processed.save_to_disk("data/processed/my_dataset")
```

Point `processed_dir` in your YAML config at `data/processed/my_dataset` and run the training script as normal.

---

### Token Classification

Fine-tuning encoder models for token-level tasks like NER and PII detection. Labels follow the BIO scheme (`O`, `B-<entity>`, `I-<entity>`); only the first subword of each entity carries the BIO tag, continuation subwords are ignored in the loss by default.

#### Quickstart

```bash
# 1. Prepare the dataset (downloads, tokenizes, aligns BIO labels to subwords)
python scripts/token_classification/prepare_dataset.py

# 2. Train
python scripts/token_classification/train_token_classification.py \
    --config configs/encoder/token_classification/pii_mbert.yaml
```

The default dataset is `ai4privacy/open-pii-masking-500k-ai4privacy` (8 languages, 19 PII entity types). The default model is `bert-base-multilingual-cased`.  
Metrics tracked at each epoch via `seqeval` (entity-level, not token-level): `precision`, `recall`, `f1`, `accuracy`.  
Final test metrics are saved to `outputs/<run>/final/test_metrics.json`.

#### Resuming after a failure

Pass `--resume` to continue from the latest checkpoint automatically:

```bash
python scripts/token_classification/train_token_classification.py \
    --config configs/encoder/token_classification/pii_mbert.yaml \
    --resume
```

Or point at a specific checkpoint:

```bash
python scripts/token_classification/train_token_classification.py \
    --config configs/encoder/token_classification/pii_mbert.yaml \
    --resume outputs/pii_mbert/checkpoint-5000
```

#### Inference

Load the trained model for high-performance NER inference. The predictor decodes BIO predictions into entity spans with character offsets back into the original text:

```python
from lievito_madre_ai_lab.encoder.token_classification.serve import TokenClassificationPredictor

predictor = TokenClassificationPredictor("outputs/pii_mbert/final")

# batch
results = predictor.predict([
    "Send the report to Maria Rossi at maria.rossi@example.com.",
    "John called from +1-800-555-0199.",
])
# [
#   [{"text": "Maria Rossi", "label": "GIVENNAME", "start": 19, "end": 30, "score": 0.98}, ...],
#   [{"text": "+1-800-555-0199", "label": "TELEPHONENUM", "start": 17, "end": 32, "score": 0.99}],
# ]

# single
spans = predictor.predict_one("Email me at jane@example.com.")
```

Run from the CLI:

```bash
python -m lievito_madre_ai_lab.encoder.token_classification.serve \
    outputs/pii_mbert/final "Send it to John Doe at john@example.com."
```

Benchmark throughput:

```bash
python -m lievito_madre_ai_lab.encoder.token_classification.serve \
    outputs/pii_mbert/final --benchmark
```

The same hardware-aware optimisation stack as text classification applies (FP16/BF16 weights, SDPA / Flash Attention 2, `torch.compile`, CPU INT8 quantisation).

#### Switching model or dataset

Duplicate the YAML config and edit `model_name`, `processed_dir`, and `output_dir`:

```bash
cp configs/encoder/token_classification/pii_mbert.yaml \
   configs/encoder/token_classification/ner_xlmr.yaml
```

For multilingual NER, `microsoft/mdeberta-v3-base` and `xlm-roberta-base` are stronger backbones than mBERT.

Then run with `--config configs/encoder/token_classification/ner_xlmr.yaml`.

#### Bringing your own data

The training script only requires a tokenized Arrow dataset with three columns: `input_ids`, `attention_mask`, and `labels` (a per-token sequence of BIO label ids, with `-100` for special and non-first-subword tokens). There are two paths to get there.

**Path 1 — your data has text + character-span annotations** (the default ai4privacy format):

Each row needs a text column and a list of entity spans:

```python
{
    "source_text": "Send it to John Doe at john@example.com.",
    "privacy_mask": [
        {"label": "GIVENNAME", "value": "John Doe", "start": 11, "end": 19},
        {"label": "EMAIL",     "value": "john@example.com", "start": 23, "end": 39},
    ],
}
```

Use `tokenize_for_trainer` to tokenize and align labels to subwords:

```python
from datasets import DatasetDict, Dataset
from lievito_madre_ai_lab.encoder.token_classification.dataset import tokenize_for_trainer

raw = DatasetDict({
    "train": Dataset.from_dict({"source_text": [...], "privacy_mask": [...]}),
    "test":  Dataset.from_dict({"source_text": [...], "privacy_mask": [...]}),
})

processed = tokenize_for_trainer(raw, model_name="bert-base-multilingual-cased")
processed.save_to_disk("data/processed/my_ner_dataset")
```

If your label set differs from the default PII entities, edit `ENTITY_TYPES` in [lievito_madre_ai_lab/encoder/token_classification/dataset.py](lievito_madre_ai_lab/encoder/token_classification/dataset.py) — `LABEL_NAMES` is derived from it as `["O"] + B-/I- pairs`.

**Path 2 — your data is already token-and-tag aligned** (e.g. CoNLL format):

Build the `labels` column directly using `LABEL2ID` and save the Arrow dataset — see `tokenize_for_trainer` in the same module for the expected schema (`input_ids`, `attention_mask`, `labels` with `labels` cast to `Sequence(ClassLabel(names=LABEL_NAMES))` so label names survive `save_to_disk`).

Point `processed_dir` in your YAML config at `data/processed/my_ner_dataset` and run the training script as normal.

---

### GLiNER Entity Extraction

Fine-tuning [GLiNER](https://github.com/urchade/GLiNER) for open-vocabulary entity extraction. Entity types are passed as prompts at inference time — the same model can extract any label you ask for, including ones it never saw at training.

**Different transformers version.** GLiNER 0.2.x requires `transformers<5.2.0`, which conflicts with the encoder/decoder/vision groups. Install in its own virtualenv:

```bash
pip install -e ".[gliner]"
```

The default backbone is [`knowledgator/gliner-multitask-large-v0.5`](https://huggingface.co/knowledgator/gliner-multitask-large-v0.5) — the strongest public GLiNER for zero-shot/few-shot at the time of writing. The trainer auto-detects precision (bf16 on Ampere+, fp16 on Turing/T4, fp32 on CPU), so the same YAML runs on a T4 and on a B100 without edits.

#### The dataset contract

Every per-corpus `prepare_dataset.py` you write must produce the following on-disk layout:

```
data/processed/<task>/
├── train/             # HF DatasetDict shards (load_from_disk-compatible)
├── validation/
├── test/              # single split; closed-set + zero-shot are runtime views
├── train_types.json   # ["PERSON", "ORG", ...]
└── holdout_types.json # [] if no zero-shot probe desired
```

Every row in every split has the same shape — char-offset spans, half-open `[start, end)`:

```python
{
    "text": "Maria Rossi lives in Rome.",
    "spans": [
        {"start": 0,  "end": 11, "label": "PERSON"},
        {"start": 21, "end": 25, "label": "CITY"},
    ],
    # any extra columns are silently kept (ignored by trainer)
}
```

**Filtering rules** the prep script enforces:
- `train` + `validation`: keep only spans whose label is in `train_types`. Held-out spans are **dropped**, not relabeled — GLiNER must see no signal on held-out types during training.
- `test`: keep all spans. Eval-time label-set filtering produces the closed-set and zero-shot views.

`scripts/gliner_entity_extraction/prepare_dataset.py` is the worked example targeting OpenPII. Use it as a template.

#### Quickstart (OpenPII reference)

```bash
# 1. Build the char-offset dataset from the ai4privacy OpenPII corpus
python scripts/gliner_entity_extraction/prepare_dataset.py \
    --out-dir data/processed/pii-gliner

# 2. Train
python scripts/gliner_entity_extraction/train_gliner.py \
    --config configs/encoder/gliner_entity_extraction/pii_gliner.yaml
```

Holdout types default to `PASSPORTNUM DRIVERLICENSENUM AGE` (overridable via `--holdout-types`). Final test metrics land in `outputs/<run>/final/test_metrics.json` with two prefixed sections:
- `test_closed_*` — F1 on `train_types`, scored against the test split.
- `test_zeroshot_*` — F1 on `holdout_types` (open-vocabulary probe), scored against the same test split with the holdout labels as the prompt set.

Per-label F1 is reported under `f1_<LABEL>`, `precision_<LABEL>`, `recall_<LABEL>` for both views. The output span schema (`label`, `text`, `start`, `end`, `score`) matches `TokenClassificationPredictor` so the same downstream code consumes both.

#### Bringing your own data

Write a `prepare_dataset.py` for your corpus that emits the char-offset contract. The simplest pattern:

```python
import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from lievito_madre_ai_lab.encoder.gliner_entity_extraction.dataset import (
    collect_entity_types, partition_entity_types, validate_row,
)

# 1. Load whatever raw corpus you have, however you have it. Just produce a
#    DatasetDict with train / validation / test splits whose rows already
#    follow the char-offset schema (or convert them to it here).
raw = DatasetDict({
    "train":      Dataset.from_list(my_train_rows),       # [{"text", "spans"}, ...]
    "validation": Dataset.from_list(my_val_rows),
    "test":       Dataset.from_list(my_test_rows),
})

# 2. Collect the full label vocabulary and partition into train + holdout.
all_types = collect_entity_types(raw, span_col="spans")
train_types, holdout_types = partition_entity_types(all_types, ["AGE", "PASSPORTNUM"])

# 3. Filter spans so train/val only carry train_types; test keeps everything.
train_set = frozenset(train_types)
full_set  = frozenset(all_types)

def _filter(row, allowed):
    return {"text": row["text"],
            "spans": [s for s in row["spans"] if s["label"] in allowed]}

processed = DatasetDict({
    "train":      raw["train"]     .map(lambda r: _filter(r, train_set)),
    "validation": raw["validation"].map(lambda r: _filter(r, train_set)),
    "test":       raw["test"]      .map(lambda r: _filter(r, full_set)),
})

# 4. Fail loudly at prep time, not training step 1.
for name, split in processed.items():
    if len(split) and (errs := validate_row(split[0])):
        raise ValueError(f"{name} row 0: {errs}")

# 5. Save.
out = Path("data/processed/my-gliner")
out.mkdir(parents=True, exist_ok=True)
processed.save_to_disk(str(out))
(out / "train_types.json").write_text(json.dumps(train_types))
(out / "holdout_types.json").write_text(json.dumps(holdout_types))
```

Then point a YAML config at `processed_dir: data/processed/my-gliner` and train.

`validate_row` checks the schema invariants (`text` is a string, every span has `end > start`, `start >= 0`, `end <= len(text)`, `label` is a non-empty string). Call it on every row during prep — schema bugs surface at prep time, not at step 1 of training.

---

#### Fine-tuning paths — LoRA vs full FT

The YAML's `gliner.peft.enabled` switch picks between the two. Choose based on memory, dataset size, and how far the backbone needs to move.

##### LoRA (recommended default — fits T4)

LoRA wraps the encoder with low-rank adapters and freezes the original encoder weights; the span scoring head stays fully trainable. On `knowledgator/gliner-multitask-large-v0.5` (~660M params), LoRA cuts the trainable parameter count by ~99% and fits comfortably on a 16GB T4 with batch 8 + grad-checkpointing.

```yaml
# configs/encoder/gliner_entity_extraction/<your>.yaml
model_name: knowledgator/gliner-multitask-large-v0.5
per_device_train_batch_size: 8
gradient_accumulation_steps: 2          # effective batch = 16
gradient_checkpointing: true            # ~40% memory cut
precision: auto                         # bf16 on Ampere+, fp16 on T4

gliner:
  peft:
    enabled: true
    r: 16                               # rank — 16 is the "first try" default
    alpha: 32                           # scaling = alpha / r
    dropout: 0.1
    target_modules: auto                # auto-detect; falls back to DeBERTa-v3 list
```

**When to use LoRA**: most fine-tuning runs (custom domain, smaller training set, T4-class hardware), or any time you'd want to ship multiple per-customer adapters off one base model.

**Tuning notes**:
- Bump `r` to 32 or 64 if LoRA underperforms full-FT by more than 1-2 F1 points after enough epochs.
- `target_modules: auto` resolves via `peft`'s mapping with a hardcoded DeBERTa-v3 fallback (`query_proj, key_proj, value_proj, dense`). Override with an explicit list if you're on an unusual backbone.

##### Full fine-tuning

Full FT updates every encoder parameter alongside the span head. Strongest ceiling, biggest memory footprint, slowest training.

```yaml
gliner:
  peft:
    enabled: false                      # the only change vs LoRA
```

Plus on T4-class hardware, lean on memory tricks:

```yaml
per_device_train_batch_size: 2          # batch must shrink without LoRA
gradient_accumulation_steps: 8          # effective batch = 16 still
gradient_checkpointing: true
torch_compile: true                     # ~25% speedup once warm
```

**When to use full FT**: large training set (>100k rows), big domain shift from the base GLiNER's pre-training distribution, or you have an A100/H100/B100 and want the strongest possible ceiling.

#### Other GLiNER-specific knobs

All under the `gliner:` block in the YAML. The defaults in `pii_gliner.yaml` are good starting points; tune from there.

```yaml
gliner:
  max_span_width: 16              # longest span (tokens) the scorer considers.
                                  # Set above the longest expected entity in
                                  # your data — addresses, multi-word names.

  loss:
    funcs: ["focal"]              # ["bce"] | ["focal"]  — focal is recommended
    focal_alpha: 0.75             # only used when funcs contains "focal"
    focal_gamma: 2.0

  sampling:
    max_types: 30                 # entity-prompt cap per batch
    max_neg_type_ratio: 1.0       # negative distractor labels per positive
    random_drop: 0.1              # chance to drop a gold label from the prompt

  head_lr_multiplier: 5.0         # head LR = learning_rate × this. Drops
                                  # straight into gliner's TrainingArguments
                                  # as `others_lr`. 3-10x is typical.

  label_aliases:                  # canonical label -> natural-language prompt
    GIVENNAME: "given name"       # GLiNER reads label names as prompts, so
    DRIVERLICENSENUM: "driver license number"   # natural-language helps zero-shot.
    EMAIL: "email address"        # Predictions are un-aliased back to canonical.
```

Loss + negative sampling fields go onto `model.config` via `load_gliner` (consumed by GLiNER's collator at batch time). Focal alpha/gamma also propagate to `gliner.training.TrainingArguments.focal_loss_alpha` / `focal_loss_gamma` (consumed by `compute_loss`). Discriminative LR is handled natively by gliner's Trainer via the `others_lr` field, derived from `head_lr_multiplier`.

#### Resuming after a failure

```bash
# Auto-detect the latest checkpoint
python scripts/gliner_entity_extraction/train_gliner.py \
    --config configs/encoder/gliner_entity_extraction/pii_gliner.yaml --resume

# Or point at a specific checkpoint
python scripts/gliner_entity_extraction/train_gliner.py \
    --config configs/encoder/gliner_entity_extraction/pii_gliner.yaml \
    --resume outputs/pii_gliner/gliner_exp_01/checkpoint-500
```

Smoke-test on a small slice before committing to a full run:

```bash
python scripts/gliner_entity_extraction/train_gliner.py \
    --config configs/encoder/gliner_entity_extraction/pii_gliner.yaml \
    --max-train-samples 200 --max-eval-samples 100 --max-test-samples 100
```

---

#### Efficient serving

[`GLiNERPredictor`](lievito_madre_ai_lab/encoder/gliner_entity_extraction/serve.py) is the single inference entrypoint. It loads a full-FT save and a LoRA save through the same constructor, and applies the right optimisation stack for the detected hardware.

```python
from lievito_madre_ai_lab.encoder.gliner_entity_extraction.serve import GLiNERPredictor

predictor = GLiNERPredictor("outputs/pii_gliner/gliner_exp_01/final")

# Closed-set: uses train_types stamped on the model at training time
spans = predictor.predict_one("Email me at jane@example.com.")
# [{"label": "EMAIL", "text": "jane@example.com", "start": 12, "end": 28, "score": 0.97}]

# Open-vocabulary: pass any labels — even ones the model never trained on
spans = predictor.predict_one(text, labels=["passport number", "driver license"])

# Batched (sorted-by-length internally so longer texts cluster together)
results = predictor.predict([
    "Send the report to Maria Rossi at maria.rossi@example.com.",
    "John called from +1-800-555-0199.",
])
```

##### What the predictor does automatically

| Concern | What happens |
|---|---|
| **Device** | CUDA if available, else MPS, else CPU. Override with `device="cuda:1"` / `"cpu"`. |
| **Precision (CUDA)** | bf16 autocast on Ampere+ (capability ≥ 8.0), fp16 autocast otherwise. |
| **Batching** | `batch_predict_entities` under the hood; sorted-by-length so similar-length texts batch together. |
| **`torch.compile`** | Applied to `model.model` (the encoder) on CUDA / MPS by default. Disable with `use_compile=False`. |
| **LoRA detection** | If `adapter_config.json` exists in the model dir, the base model is loaded and the adapter is wrapped with `PeftModel.from_pretrained`, then `merge_and_unload()` folds the LoRA weights into the base — zero inference overhead vs full FT. Disable the merge with `merge_lora_on_load=False` if you need to swap adapters at runtime. |
| **CPU quantisation** | Opt-in via `quantize_cpu=True` — applies dynamic INT8 to FFN Linears only (the span scorer's small margins do not survive INT8 on the attention path). |
| **Label aliases** | If `label_aliases` was stamped at training time, prompts go out aliased and predictions come back un-aliased — caller sees only canonical labels. |

##### Constructor knobs worth knowing

```python
predictor = GLiNERPredictor(
    "outputs/pii_gliner/gliner_exp_01/final",
    device=None,                  # auto-detect
    batch_size=16,                # tune to your GPU; lower for long texts
    use_compile=True,             # torch.compile the encoder
    compile_mode="default",       # "default" | "reduce-overhead" | "max-autotune"
    amp_dtype=None,               # auto: bf16 on Ampere+, fp16 elsewhere
    quantize_cpu=False,           # True for CPU-only servers
    warmup_steps=3,               # compile warmup passes
    default_threshold=0.5,        # default span-score cutoff
    merge_lora_on_load=True,      # fold LoRA into base for fastest inference
)
```

##### CLI

```bash
python -m lievito_madre_ai_lab.encoder.gliner_entity_extraction.serve \
    outputs/pii_gliner/gliner_exp_01/final \
    "Send it to Maria Rossi at maria.rossi@example.com."

# Open-vocabulary
python -m lievito_madre_ai_lab.encoder.gliner_entity_extraction.serve \
    outputs/pii_gliner/gliner_exp_01/final \
    "Driver license #ABC123 expired" \
    --labels "driver license number" "expiry date"
```

##### Serving LoRA without merging

If you need to swap adapters at runtime (one base model, many per-tenant adapters):

```python
predictor = GLiNERPredictor("outputs/tenant_a/final", merge_lora_on_load=False)
# predictor._model.model is now a PeftModel; you can call .set_adapter() etc.
```

The trade-off: keeping the adapter unmerged costs ~5-10% inference latency vs the merged path.
