# Scripts

Three entry points. All read a single YAML config (see [CONFIG_REFERENCE.md](../CONFIG_REFERENCE.md)).

| Script | What it does |
|---|---|
| `train_vlm.py` | Fine-tune a VLM (LoRA / QLoRA / full) on a grounding dataset, then evaluate on the test split. |
| `baseline_zeroshot.py` | Evaluate the *untrained* base model on the same test split and print the delta vs the fine-tune. Run this first — it's the bar training has to clear. |
| `inference.py` | Load a saved model and print/draw predictions for a few images. |

## Train

```bash
python scripts/train_vlm.py --config examples/grounding/configs/cppe5_qwen25vl_3b.yaml

# Resume after a crash (auto-detect latest checkpoint, or pass a path)
python scripts/train_vlm.py --config ... --resume
python scripts/train_vlm.py --config ... --resume outputs/run/checkpoint-500

# Smoke a tiny slice before committing to a full run
python scripts/train_vlm.py --config ... \
    --max-train-samples 8 --max-eval-samples 4 --max-test-samples 4
```

What it produces under `outputs/<run>/final/`:

```
final/
├── adapter_config.json + adapter_model.safetensors   # (LoRA) — or full weights
├── (processor / tokenizer files)
├── preprocessing.json     # processor id, task, coord_bins, default_prompt, labels, …
└── test_metrics.json      # test_precision / test_recall / test_f1 / test_f1_iou_avg + per-label
```

Training **saves the model and `preprocessing.json` before** running the test
eval — so a crash during generation still leaves a loadable checkpoint.

## Baseline

```bash
# Does fine-tuning beat just prompting the base model?
python scripts/baseline_zeroshot.py --config examples/grounding/configs/cppe5_qwen25vl_3b.yaml
```

Prints a side-by-side table (baseline vs fine-tune) and writes
`baseline_metrics.json` next to the fine-tune.

## Inference

```bash
python scripts/inference.py outputs/<run>/final img1.jpg img2.jpg --draw-dir /tmp/preds
python scripts/inference.py outputs/<run>/final img.jpg --prompt "Point at every mask."
```

Or use the predictor directly — see the [serve module](../vlm_finetuning/serve.py) and the
[project README](../README.md#serving).
