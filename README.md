# lievito-madre-ai-lab
Lievito Madre isn't just an ingredient; it is a living, evolving ecosystem of knowledge. Here is where we cook.

---

## Encoder

Encoder models (BERT, RoBERTa, DeBERTa, XLM-R, ModernBERT, …) fine-tuned with the HuggingFace `Trainer`. Every run is driven by a single YAML file that loads into the [`TrainConfig`](lievito_madre_ai_lab/shared/config.py) dataclass. Each field maps 1-to-1 to a HuggingFace `TrainingArguments`, `from_pretrained`, or `DataCollator` parameter — no hidden translation, no defaults overridden behind your back.

### Tasks

| Task | Script | Example dataset |
|---|---|---|
| [Text Classification](scripts/text_classification/README.md) | `scripts/text_classification/train_text_classification.py` | `dair-ai/emotion` |
| [Token Classification](scripts/token_classification/README.md) | `scripts/token_classification/train_token_classification.py` | `ai4privacy/open-pii-masking-500k-ai4privacy` |
| [GLiNER Entity Extraction](scripts/gliner_entity_extraction/README.md) | `scripts/gliner_entity_extraction/train_gliner.py` | `ai4privacy/open-pii-masking-500k-ai4privacy` |

Each task README covers: quickstart, resuming after a failure, inference, switching model/dataset, and bringing your own data.

---

### Configuration

The tables below cover every field accepted by a YAML config. Defaults reflect what `TrainConfig()` ships with; the **Best practice** column says when to change them.

#### Data & model

| Field | Default | What it does | Notes |
|---|---|---|---|
| `processed_dir` | `data/processed/emotion` | Path to the tokenized Arrow dataset produced by a prepare script. | Required for any real run. The prepare scripts also write a `preprocessing.json` sidecar here (tokenizer, `max_length`, sliding-window stride, …) — the train script reads it to guard against tokenizer/`max_length` drift, and the predictor reads it on load so inference defaults match training. See each task README for the per-pipeline details. |
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
