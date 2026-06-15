# Object Detection

SOTA fine-tuning of **canonical (DETR-family) object detectors** — D-FINE,
RT-DETR / RT-DETRv2, Deformable DETR, Conditional DETR, DETR — via HuggingFace
`AutoModelForObjectDetection`. Not a VLM: these are dedicated transformer
detectors that regress boxes + classes directly. Runs are configured by a single
YAML that loads into the [`TrainConfig`](object_detection/shared/config.py)
dataclass plus a `detection:` block.

The default base is **D-FINE-X** (Objects365→COCO pretrained) — the highest COCO
AP among canonical detectors available in transformers — with a ready
**RT-DETRv2-R50** config as the always-available fallback. The core is
backbone-agnostic: backbone module name, projection layers, and box
post-processing are all discovered at runtime, so swapping the checkpoint is a
one-line config change. The worked example fine-tunes on
[CPPE-5](https://huggingface.co/datasets/cppe-5) — the **same dataset** as the
`vlm-finetuning` grounding project, so you can compare a classical detector
against VLM grounding head-to-head.

A self-contained project with its own vendored copy of the lab's shared config
and preprocessing helpers under [`object_detection/shared/`](object_detection/shared/).

> **Isolated environment required.** Needs a recent `transformers` (D-FINE landed
> in 4.48), which conflicts with the GLiNER project's pin. Use its own virtualenv.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # GPU
pip install -e ".[dev]"
```

## Quickstart

```bash
# 1. Build the COCO-format dataset from CPPE-5
python examples/cppe5/dataset/prepare_cppe5.py --out-dir data/processed/cppe5
# 2. (optional) pretrained-baseline sanity check
python scripts/baseline_pretrained.py --config examples/cppe5/configs/dfine_x.yaml
# 3. Fine-tune
python scripts/train_detector.py --config examples/cppe5/configs/dfine_x.yaml
```

See [scripts/README.md](scripts/README.md) and [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md).

## Tests

```bash
pytest                     # fast unit tests (most need no torch)
RUN_DET_SMOKE=1 pytest     # end-to-end smoke (YOLOS-tiny on a tiny fixture, CPU)
```

The smoke test uses a tiny synthetic COCO fixture under `tests/fixtures/coco_tiny/`,
**generated not committed** — `tests/conftest.py` rebuilds it on first use from
[tests/fixtures/build_coco_tiny.py](tests/fixtures/build_coco_tiny.py).

## Dataset contract

A prepare script produces a `DatasetDict` (with an `Image` column), a `labels.json`
(ordered category names), and a `preprocessing.json` sidecar. Every row is COCO
layout with **absolute pixel `xywh`** boxes:

```python
{
    "image":    PIL.Image,
    "image_id": 42,
    "objects": {
        "bbox":        [[x, y, w, h], ...],   # absolute pixels, COCO xywh
        "category_id": [0, 3, ...],            # contiguous 0..K-1
        "area":        [..],                   # optional
        "iscrowd":     [0, ..],                # optional
    },
}
```

Pixel `xywh` is exactly what the HF image processors and Albumentations consume —
the processor resizes/normalizes/pads and rescales the boxes to match, so the
dataset stays independent of the model's input resolution. `validate_row` (called
by every prepare script) catches schema bugs at prep time.

## The SOTA recipe (what makes this more than a tutorial)

| Concern | What this project does |
|---|---|
| **Head swap** | `ignore_mismatched_sizes=True` replaces only the pretrained class head with one sized to your classes; backbone + encoder + decoder weights are kept. |
| **Discriminative LR** | The pretrained backbone trains at `backbone_lr_mult ×` the head LR (0.1× by default) — the canonical DETR/RT-DETR/D-FINE schedule and the biggest stability lever. Norms + biases excluded from weight decay. |
| **bbox-aware augmentation** | Albumentations transforms image **and** boxes together (flip + photometric), with `clip`/`min_area`/`min_visibility` so augments never produce phantom targets. Geometric resize is left to the processor. |
| **Weight EMA** | A shadow copy updated every optimizer step, swapped in for **both** evaluation and checkpointing — so the metric you select on and the weights you ship are the EMA weights. Typically +0.5–1.5 AP, free. Toggle with `detection.ema`. |
| **Real COCO mAP eval** | `torchmetrics.MeanAveragePrecision` (pycocotools backend): mAP@[.5:.95], mAP@.50, mAP@.75, size-stratified, and per-class AP — computed by generating + decoding boxes, the way COCO scores. `metric_for_best_model: map` selects the genuinely-best checkpoint, not lowest loss. |
| **Precision auto** | bf16+tf32 on Ampere+ (A100), fp16 on Turing/T4, fp32 on CPU. Model is loaded fp32 (DETR Hungarian matching is precision-sensitive) and autocast applies in the Trainer. |
| **Crash-safe saves** | Model + `preprocessing.json` written before the test eval; with EMA on, every checkpoint stores EMA weights so `load_best_model_at_end` stays consistent. |

## Serving

```python
from object_detection.serve import ObjectDetectionPredictor, draw_detections

predictor = ObjectDetectionPredictor("outputs/cppe5_dfine_x/run1/final")
dets = predictor.predict_one("ward.jpg")            # threshold from preprocessing.json
# [{"label": "Mask", "score": 0.94, "box": [x1, y1, x2, y2]}, ...]   # pixels
draw_detections("ward.jpg", dets).save("annotated.png")
```

CLI:

```bash
python -m object_detection.serve outputs/<run>/final img.jpg --draw-dir /tmp/preds --threshold 0.5
```

## Scope

DETR-family transformer detectors (everything `AutoModelForObjectDetection`
loads). YOLO-family models live in a different ecosystem (ultralytics) and are
out of scope here — a candidate for a separate future lab project.

## Layout

```
object_detection/  importable package (dataset, model, trainer, evaluate, serve
                   + vendored shared/)
scripts/           train / inference / pretrained-baseline entry points
examples/          configs and dataset-prep scripts (CPPE-5 worked example)
tests/             unit tests (+ auto-built tests/fixtures/coco_tiny, gitignored)
data/              processed datasets (gitignored)
```
