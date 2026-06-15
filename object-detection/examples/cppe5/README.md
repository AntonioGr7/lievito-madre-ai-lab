# CPPE-5 example

Fine-tune a canonical detector on [CPPE-5](https://huggingface.co/datasets/cppe-5)
— ~1k images, 5 medical-PPE classes (Coverall, Face Shield, Gloves, Goggles, Mask)
with bounding boxes. Same dataset as the `vlm-finetuning` grounding example, so
you can compare a **classical detector** against **VLM grounding** on identical data.

```bash
# prepare → baseline → train (defaults to D-FINE-X)
bash examples/cppe5/run.sh

# or step by step:
python examples/cppe5/dataset/prepare_cppe5.py --out-dir data/processed/cppe5
python scripts/train_detector.py --config examples/cppe5/configs/dfine_x.yaml
```

## Configs

| Config | Backbone | Notes |
|---|---|---|
| `dfine_x.yaml` | D-FINE-X (obj2coco) | Best AP; default. Confirm the `ustc-community/dfine-*` id on the Hub. |
| `rtdetrv2_r50.yaml` | RT-DETRv2-R50 | Rock-solid fallback (`PekingU/rtdetr_v2_r50vd`). |
| `smoke.yaml` | YOLOS-tiny | CPU end-to-end on the tiny fixture. |

## Results

`outputs/<run>/final/test_metrics.json` — COCO mAP: `test_map` (mAP@[.5:.95]),
`test_map_50`, `test_map_75`, size-stratified `test_map_small/medium/large`, and
per-class `test_map_Coverall`, … Run `baseline_pretrained.py` first; note a
COCO-pretrained model scores ~0 against CPPE-5's labels (different label space),
so the baseline here is mainly a pipeline sanity check.

## Bring your own data

Copy `dataset/prepare_cppe5.py`. Emit, per split, rows of
`{"image", "image_id", "objects": {"bbox" (pixel xywh), "category_id", "area", "iscrowd"}}`
and write `labels.json` (ordered category names). `validate_row` (called in the
prep script) catches schema bugs before training. The core is dataset- and
backbone-agnostic — the example is one instantiation.
