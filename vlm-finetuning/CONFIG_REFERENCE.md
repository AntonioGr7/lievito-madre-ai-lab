# Config reference

A run is one YAML file. Top-level keys load into the
[`TrainConfig`](vlm_finetuning/shared/config.py) dataclass (a thin shadow of HF
`TrainingArguments`); the `vlm:` block carries everything task-specific (the
coordinate scheme, LoRA/quantization, generation-eval settings) and is parsed
separately by the train script.

Defaults below reflect what `TrainConfig()` / the `vlm:` parser ship with; the
**Notes** column says when to change them.

## Top-level fields

### Data & model

| Field | Default | What it does | Notes |
|---|---|---|---|
| `processed_dir` | `data/processed/cppe5` | Path to the processed grounding DatasetDict (a prepare script's output). | Required for any real run. The prep script writes a `preprocessing.json` sidecar here (source, task, coord_bins, labels, default_prompt); the train script reads it and forwards it into the saved model dir. |
| `model_name` | `Qwen/Qwen2.5-VL-3B-Instruct` | HuggingFace model id or local path. | Any `AutoModelForImageTextToText`-loadable VLM: Qwen2.5-VL (2B/3B/7B), SmolVLM / SmolVLM2 (256M/500M/2.2B), Idefics3, LLaVA-family. |
| `output_dir` | `outputs/run` | Where checkpoints and the final model are saved. | |
| `experiment_id` | `null` | If set, appends to `output_dir` → `outputs/<run>/<experiment_id>`. | Handy for sweeps. |
| `attn_implementation` | `sdpa` | Attention kernel. | `sdpa` is right almost always. `flash_attention_2` to fail loud if FA2 is missing; `eager` for debugging. |

### Core hyperparameters

| Field | Default | What it does | Best practice |
|---|---|---|---|
| `num_train_epochs` | `3` | Total training epochs. | 3-8 for LoRA on a small grounding set; lean on `early_stopping_patience`. |
| `max_steps` | `-1` | If > 0, overrides `num_train_epochs`. | |
| `per_device_train_batch_size` | `4` | Batch size per GPU. | VLMs are memory-heavy: 1-2 on a T4, 2-4 on a 24GB card. Use `gradient_accumulation_steps` to reach an effective batch of 16-32. |
| `per_device_eval_batch_size` | `4` | Eval generation batch size. | |
| `gradient_accumulation_steps` | `4` | Optimizer-step interval. Effective batch = `per_device × accum × world_size`. | Raise when memory-bound to keep effective batch ≥ 16. |
| `learning_rate` | `1e-4` | Peak LR after warmup. | `1e-4`-`2e-4` for LoRA; `2e-4` is fine for QLoRA; `~1e-5` for full FT. |
| `weight_decay` | `0.01` | AdamW L2 regularization. | |
| `warmup_ratio` | `0.03` | Fraction of steps spent warming up LR. | `0.03`-`0.1`. Converted to absolute `warmup_steps` at the trainer boundary. |
| `max_grad_norm` | `1.0` | Gradient clipping. | Keep at `1.0`. |
| `lr_scheduler_type` | `cosine` | LR schedule. | `cosine` for LoRA; `linear`/`constant` also fine. |
| `optim` | `adamw_torch_fused` | Optimizer. | `adamw_torch_fused` on PyTorch 2.x. **`paged_adamw_8bit` for QLoRA** (avoids OOM spikes). |
| `seed` | `42` | Random seed. | |

### Checkpoint & evaluation

| Field | Default | What it does | Notes |
|---|---|---|---|
| `eval_strategy` | `epoch` | When to run eval. | `no` / `steps` / `epoch`. Eval runs generation — it's not free; per-epoch is a good default. |
| `save_strategy` | `epoch` | When to checkpoint. | Must match `eval_strategy` when `load_best_model_at_end=true`. |
| `load_best_model_at_end` | `true` | Reload the best checkpoint before the final save. | |
| `metric_for_best_model` | `f1` | Metric to rank checkpoints. | Grounding: `eval_f1` (also `eval_f1_iou_avg`, `eval_precision`, `eval_recall`). **Text task**: set to `token_f1` (or `exact_match`) — the eval callback publishes `eval_token_f1` / `eval_exact_match`. |
| `greater_is_better` | `true` | Whether higher is better. | |
| `save_total_limit` | `2` | Max checkpoints on disk. | |
| `early_stopping_patience` | `null` | Wires `EarlyStoppingCallback` when set. | `2`-`3` is typical. |
| `eval_accumulation_steps` | `null` | Offload eval tensors GPU→CPU every N steps. | Rarely needed here (eval is generation, not logits). |

### Compute & throughput

| Field | Default | What it does | Best practice |
|---|---|---|---|
| `precision` | `auto` | Mixed-precision dtype. | `auto` → bf16+tf32 on Ampere+, fp16 on Turing/T4, fp32 on CPU. Controls both the model load dtype and the autocast dtype. |
| `gradient_checkpointing` | `true` | Recompute activations during backward. | Keep **on** for VLMs — they're activation-heavy. ~40% memory saved. Uses non-reentrant checkpointing (required for PEFT). |
| `torch_compile` | `false` | `torch.compile` during training. | Can fragile-fail on VLM image branches; leave off unless you've verified it. |
| `dataloader_num_workers` | `4` | Parallel data-loading workers. | `0` for debugging; image decode benefits from `4`-`8`. |

### Logging & Weights & Biases

| Field | Default | What it does |
|---|---|---|
| `logging_steps` | `10` | Log training metrics every N steps. |
| `report_to` | `none` | Logger integration. Set to `wandb` to enable. |
| `wandb_project` / `wandb_run_name` / `wandb_tags` / `wandb_notes` | `null` / `null` / `[]` / `null` | W&B run metadata (required: `wandb_project` when `report_to: wandb`). The eval callback logs grounding F1 to W&B by step. |

## The `vlm:` block

| Field | Default | What it does | Notes |
|---|---|---|---|
| `task` | `box` | What to supervise/score: `box`, `point`, or `text`. | `box`/`point` serialize the row's `objects` to `<box>` tokens (`point` uses box centres). **`text`** uses the row's free-form `response` string verbatim (tool calls, JSON, captions) and scores with exact-match / token-F1. |
| `coord_bins` | `1000` | Resolution of the `<box>` integer grid (grounding only). | Coordinates are normalized to `[0,1]` then quantized to `[0, coord_bins)`. 1000 is plenty; raise only for sub-pixel tasks. Ignored when `task: text`. |
| `system_prompt` | `null` | Optional system turn prepended to every conversation. | |
| `default_prompt` | first train row's prompt | Stored in `preprocessing.json` so the predictor has a fallback prompt. | The per-row `prompt` is always used during training/eval; this is only the serve-time default. |
| `empty_text` | `No objects detected.` | Target text when a row has no objects. | Lets the model learn the "found nothing" case explicitly. |
| `max_length` | `null` | Truncate packed sequences to N tokens. | Set to bound memory on long label lists / high-res images (e.g. `2048`). `null` = no truncation. |
| `freeze_vision_tower` | `true` | Freeze the vision encoder / connector. | Recommended on. Turn off only with a large dataset and a real domain shift in the imagery. |
| `add_coord_special_tokens` | `false` | Register `<box>`/`</box>` as special tokens + resize embeddings. | When on, the new embedding rows are auto-added to LoRA `modules_to_save` so they actually train. Off (plain text) is the most portable default. |
| `image_min_pixels` / `image_max_pixels` | `null` | Bound dynamic-resolution image-token counts (Qwen-style processors). | The dominant memory lever. e.g. `200704` (256·28·28) / `1003520` (1280·28·28). Lower the max if you OOM; raise for tiny objects. Ignored by processors that don't support them. |
| `trust_remote_code` | `false` | Pass to `from_pretrained` / `AutoProcessor`. | Needed for some custom-code checkpoints. |

### `vlm.eval`

| Field | Default | What it does |
|---|---|---|
| `max_new_tokens` | `512` | Generation budget at eval/serve time. Raise if images have many objects (each costs ~10-15 tokens). |
| `iou_threshold` | `0.5` | IoU at which a predicted box counts as a match (box task). |
| `max_samples` | `null` | Cap how many eval rows are generated on (generation is the slow part). `null` = full split. |

### `vlm.lora`

| Field | Default | What it does | Notes |
|---|---|---|---|
| `enabled` | `true` | LoRA on (vs full fine-tuning). | |
| `r` | `16` | LoRA rank. | Bump to 32/64 if LoRA underfits full FT by more than a point or two. |
| `alpha` | `32` | Scaling = `alpha / r`. | Keep `alpha = 2·r` as a rule of thumb. |
| `dropout` | `0.05` | LoRA dropout. | |
| `target_modules` | `auto` | Which modules get adapters. | `auto` = the LM's attention+MLP projections, vision tower excluded. Override with an explicit list for unusual backbones. |
| `modules_to_save` | `[]` | Extra full-rank modules to train. | Auto-extended with `embed_tokens`/`lm_head` when `add_coord_special_tokens` is on. |

### `vlm.quant` (QLoRA)

| Field | Default | What it does | Notes |
|---|---|---|---|
| `load_in_4bit` | `false` | Load the base model in 4-bit (bitsandbytes). | Requires `pip install -e ".[quant]"` and CUDA. Pair with `optim: paged_adamw_8bit`. |
| `bnb_4bit_quant_type` | `nf4` | 4-bit data type. | `nf4` (recommended) or `fp4`. |
| `bnb_4bit_use_double_quant` | `true` | Nested quantization of the quant constants. | Small extra memory saving; keep on. |
| `bnb_4bit_compute_dtype` | `bfloat16` | Dtype for the de-quantized matmuls. | `bfloat16` on Ampere+; **`float16` on a T4**. |

## Choosing the precision dtype

| GPU generation | Recommended |
|---|---|
| Ampere or newer (sm ≥ 80: A100, H100, RTX 30/40) | `precision: auto` → bf16 + tf32 |
| Turing / Volta (sm 70-75: T4, V100, RTX 20) | `precision: auto` → fp16 (set `bnb_4bit_compute_dtype: float16` for QLoRA) |
| CPU | `precision: fp32` |

`bf16` is preferred over `fp16` whenever available — same dynamic range as fp32,
no loss scaling, more stable.
