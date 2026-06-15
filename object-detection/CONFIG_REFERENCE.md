# Config reference

A run is one YAML file. Top-level keys load into the
[`TrainConfig`](object_detection/shared/config.py) dataclass (a thin shadow of HF
`TrainingArguments`); the `detection:` block carries the detector-specific recipe
(image size, discriminative LR, augmentation, EMA, mAP eval) and is parsed
separately by the train script.

## Top-level fields

### Data & model

| Field | Default | What it does | Notes |
|---|---|---|---|
| `processed_dir` | `data/processed/cppe5` | Processed COCO-format DatasetDict (a prepare script's output). | Must contain `labels.json`. |
| `model_name` | `ustc-community/dfine-xlarge-obj2coco` | HF model id or local path. | Any `AutoModelForObjectDetection` checkpoint: D-FINE, RT-DETR(v2), Deformable/Conditional DETR, DETR, YOLOS. If a D-FINE slug 404s, check the `ustc-community` org for the current name. |
| `output_dir` | `outputs/run` | Where checkpoints + the final model are saved. | |
| `experiment_id` | `null` | Appends to `output_dir`. | |
| `attn_implementation` | `null` | Passed to `from_pretrained` only if set. | Most detectors don't accept it; leave `null`. |

### Core hyperparameters

| Field | Default | What it does | Best practice |
|---|---|---|---|
| `num_train_epochs` | `50` | Total epochs. | Detection fine-tuning wants many epochs (30–100 on small sets); lean on `early_stopping_patience`. |
| `per_device_train_batch_size` | `8` | Batch per GPU. | 2–4 for D-FINE-X, 8–16 for RT-DETR-R50 on a 24–80GB card; pair with `gradient_accumulation_steps` for effective ≥16. |
| `gradient_accumulation_steps` | `1` | Optimizer-step interval. | |
| `learning_rate` | `1e-4` | Peak **head** LR after warmup. | `1e-4` is the DETR-family default; the backbone gets `× backbone_lr_mult`. |
| `weight_decay` | `1e-4` | AdamW L2 (norms/biases excluded). | |
| `warmup_ratio` | `0.03` | Warmup fraction → absolute steps at the trainer boundary. | |
| `max_grad_norm` | `0.1` | Gradient clipping. | DETR-family train stably with **tight** clipping (0.1); don't raise to 1.0. |
| `lr_scheduler_type` | `cosine` | LR schedule. | `cosine` or `linear`. |
| `optim` | `adamw_torch_fused` | Optimizer. | |
| `seed` | `42` | | |

### Checkpoint & evaluation

| Field | Default | What it does | Notes |
|---|---|---|---|
| `eval_strategy` / `save_strategy` | `epoch` | When to eval / checkpoint. | Must match when `load_best_model_at_end=true`. mAP eval runs inference — per-epoch is sensible. |
| `load_best_model_at_end` | `true` | Reload best checkpoint before final save. | With EMA on, checkpoints store EMA weights, so this stays consistent. |
| `metric_for_best_model` | `map` | Checkpoint-ranking metric. | The eval callback publishes `eval_map` (also `eval_map_50`, `eval_map_75`, size + per-class). |
| `greater_is_better` | `true` | | |
| `save_total_limit` | `2` | Max checkpoints kept. | |
| `early_stopping_patience` | `null` | `EarlyStoppingCallback` when set. | `8`–`10` is typical for detection. |

### Compute & logging

| Field | Default | What it does | Best practice |
|---|---|---|---|
| `precision` | `auto` | Mixed precision. | `auto` → bf16+tf32 (Ampere+), fp16 (T4), fp32 (CPU). Model loads in fp32 regardless (matcher precision); autocast applies in training. |
| `gradient_checkpointing` | `false` | Recompute activations. | Turn on if memory-bound (large image size / batch). |
| `torch_compile` | `false` | `torch.compile`. | Off by default; can fragile-fail on detector code. |
| `dataloader_num_workers` | `4` | Data-loading workers. | `4`–`8`; image decode + augment benefit. |
| `logging_steps` | `25` | Log every N steps. | |
| `report_to` | `none` | `wandb` to enable. | `wandb_project`/`run_name`/`tags`/`notes` set the run; the eval callback logs mAP by step. |

## The `detection:` block

| Field | Default | What it does | Notes |
|---|---|---|---|
| `image_size` | `null` | Square resize fed to the model. | `null` = processor default. RT-DETR/D-FINE use 640. The dominant memory/speed lever. |
| `freeze_backbone` | `false` | Freeze the backbone entirely. | Fast/low-memory; lower ceiling. Composes with discriminative LR (frozen params just dropped). |
| `backbone_lr_mult` | `0.1` | Backbone LR = `learning_rate × this`. | The canonical DETR trick. `0.1` is standard; `0` ≈ frozen. |
| `score_threshold` | `0.3` | **Serve-time** default cutoff, saved to `preprocessing.json`. | Not used for mAP (that's `eval.threshold`). |
| `trust_remote_code` | `false` | Passed to `from_pretrained`/processor. | |

### `detection.augmentation`

| Field | Default | What it does |
|---|---|---|
| `enabled` | `true` | Master switch for train-time augmentation (eval is always clean). |
| `hflip` | `0.5` | Horizontal-flip probability (boxes flipped too). |
| `brightness_contrast` | `0.5` | `RandomBrightnessContrast` probability. |
| `hue_sat` | `0.3` | `HueSaturationValue` probability. |

Geometric resize is intentionally **not** here — the image processor handles it
and rescales boxes. Set any probability to 0 to disable that op.

### `detection.ema`

| Field | Default | What it does | Notes |
|---|---|---|---|
| `enabled` | `true` | Weight EMA, swapped in for eval **and** checkpointing. | +0.5–1.5 AP typically. Turn off for the fastest possible runs / debugging. |
| `decay` | `0.9997` | EMA decay. | Higher = slower-moving average; `0.999`–`0.9999` typical. |

### `detection.eval`

| Field | Default | What it does |
|---|---|---|
| `threshold` | `0.0` | Score cutoff for mAP. Keep at **0.0** so every query's box reaches the metric — mAP integrates the PR curve. |
| `max_samples` | `null` | Cap eval-time inference cost (`null` = full split). |

## Choosing precision

| GPU | Recommended |
|---|---|
| Ampere+ (A100, H100, RTX 30/40) | `auto` → bf16 + tf32 |
| Turing/T4, V100 | `auto` → fp16 |
| CPU | `fp32` |
