# Scripts

Three entry points. All read a single YAML config (see [CONFIG_REFERENCE.md](../CONFIG_REFERENCE.md)).

| Script | What it does |
|---|---|
| `train_detector.py` | Fine-tune a detector (discriminative-LR AdamW, augmentation, optional EMA), then evaluate COCO mAP on the test split. |
| `baseline_pretrained.py` | Evaluate the off-the-shelf checkpoint on the same test split and print the delta. Sanity bar before training. |
| `inference.py` | Load a saved model and print / draw detections for a few images. |

## Train

```bash
python scripts/train_detector.py --config examples/cppe5/configs/dfine_x.yaml

# Resume after a crash
python scripts/train_detector.py --config ... --resume

# Smoke a tiny slice
python scripts/train_detector.py --config ... \
    --max-train-samples 8 --max-eval-samples 4 --max-test-samples 4
```

Produces under `outputs/<run>/final/`:

```
final/
├── model.safetensors + config.json        # EMA weights if ema.enabled (the default)
├── preprocessor_config.json                # the image processor
├── preprocessing.json                      # processor id, image_size, threshold, labels
└── test_metrics.json                       # test_map / test_map_50 / test_map_75 + per-class
```

The model + `preprocessing.json` are saved **before** the final test eval, so a
crash during evaluation still leaves a loadable checkpoint. With `ema.enabled`,
the saved weights (and every checkpoint, so `load_best_model_at_end` is
consistent) are the EMA weights.

## Baseline

```bash
python scripts/baseline_pretrained.py --config examples/cppe5/configs/dfine_x.yaml
```

A COCO-pretrained detector predicts COCO's classes, not yours — so raw mAP vs
your labels is mostly a pipeline sanity check. For a real comparison, point
`--model` at another fine-tuned run's `final/` dir.

## Inference

```bash
python scripts/inference.py outputs/<run>/final img1.jpg img2.jpg --draw-dir /tmp/preds
python scripts/inference.py outputs/<run>/final img.jpg --threshold 0.5
```

Or use [`ObjectDetectionPredictor`](../object_detection/serve.py) directly — see the
[project README](../README.md#serving).
