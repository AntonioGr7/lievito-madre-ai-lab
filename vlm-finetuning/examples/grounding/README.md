# Grounding example — CPPE-5

Fine-tune a VLM to **detect and point at** objects, emitting the labels-plus-
`<box>`-coordinate-token format. The worked dataset is
[CPPE-5](https://huggingface.co/datasets/cppe-5) — ~1k images, 5 medical-PPE
categories (Coverall, Face Shield, Gloves, Goggles, Mask) with bounding boxes.

```bash
# One-shot: prepare → baseline → train (defaults to the Qwen2.5-VL-3B LoRA config)
bash examples/grounding/run.sh

# Or step by step:
python examples/grounding/dataset/prepare_cppe5.py --out-dir data/processed/cppe5
python scripts/baseline_zeroshot.py --config examples/grounding/configs/cppe5_qwen25vl_3b.yaml
python scripts/train_vlm.py        --config examples/grounding/configs/cppe5_qwen25vl_3b.yaml
```

## What the model learns

Given an image and the prompt the prepare script bakes in, the assistant turn is
supervised on text like:

```
Coverall<box> 118, 95, 902, 974 </box>
Mask<box> 402, 120, 553, 268 </box>
Gloves<box> 631, 712, 770, 889 </box>
```

Coordinates are integers on a `coord_bins`-resolution grid (default `[0, 1000)`),
normalized to the image so they're resolution-independent. Switch
`vlm.task: point` (see `cppe5_smolvlm_500m.yaml`) to supervise on box-centre
**points** instead — `Mask<box> 477, 194 </box>` — the "point at it" variant.

## Configs

| Config | Backbone | Method | Fits |
|---|---|---|---|
| `cppe5_qwen25vl_3b.yaml` | Qwen2.5-VL-3B | LoRA, bf16 | 24GB GPU |
| `cppe5_qwen25vl_3b_qlora.yaml` | Qwen2.5-VL-3B | QLoRA (4-bit) | 16GB T4 |
| `cppe5_smolvlm_500m.yaml` | SmolVLM-500M | LoRA, point task | 8-12GB GPU |
| `smoke.yaml` | SmolVLM-256M | LoRA, CPU | the tiny fixture |

## Results land in

`outputs/<run>/final/test_metrics.json` — `test_f1` (at IoU 0.5), `test_f1_iou_avg`
(COCO 0.50:0.95 sweep), `test_precision`/`test_recall`, and per-label F1
(`test_f1_Coverall`, …). Run `baseline_zeroshot.py` first to see what the
untrained base model scores — that's the bar fine-tuning has to clear.

## Bring your own data

Copy `dataset/prepare_cppe5.py`. You only need to emit, per split, rows of
`{"image", "prompt", "objects"}` where each object has a normalized `box`
(and/or `point`) in `[0, 1]`, then write `labels.json`. `validate_row` (called
in the prep script) catches schema bugs before training starts. The core is
backbone- and dataset-agnostic — the example is just one instantiation.
