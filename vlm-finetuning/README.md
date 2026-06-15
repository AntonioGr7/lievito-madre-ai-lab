# VLM Fine-tuning

LoRA / QLoRA (and full) fine-tuning of vision-language models — a **generic
image→text supervised-fine-tuning** engine. Every row is `(image, prompt,
target)`; the target is either a free-form string (tool calls, JSON extraction,
captioning, VQA) or a list of objects that gets serialized to **grounding**
coordinate tokens — bounding boxes `Coverall<box> x1, y1, x2, y2 </box>` and
points `Mask<box> x, y </box>`. Runs are configured by a single YAML that loads
into the [`TrainConfig`](vlm_finetuning/shared/config.py) dataclass plus a `vlm:`
block (`task: box | point | text`).

The core is **backbone- and dataset-agnostic**: every model-family detail
(vision-tower name, LM projection names, image-token expansion, chat template)
is discovered at runtime, so [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct),
[SmolVLM/Idefics3](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) and
LLaVA-family models all train through the same path. Two worked examples ship:

| Example | Task | Output |
|---|---|---|
| [`examples/grounding/`](examples/grounding/) (CPPE-5) | `box` / `point` | labels + `<box>` coordinate tokens |
| [`examples/json_extraction/`](examples/json_extraction/) (CORD-v2) | `text` | receipt → JSON string |

Adding a new output format (a tool-calling dataset, captions, …) is a new
`examples/<name>/` sibling — a prepare script + a config — **not** a new project:
the training engine is shared.

A self-contained project: it carries its own vendored copy of the lab's shared
config and preprocessing helpers under [`vlm_finetuning/shared/`](vlm_finetuning/shared/).

> **Isolated environment required.** This project needs a recent `transformers`
> (Qwen2.5-VL landed in 4.49); the GLiNER project pins `transformers<5.2.0`.
> Always install this in its own virtualenv.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # GPU
pip install -e ".[dev]"
pip install -e ".[quant]"     # optional: bitsandbytes, for 4-bit QLoRA
```

## Quickstart

```bash
# 1. Build the grounding dataset from CPPE-5
python examples/grounding/dataset/prepare_cppe5.py --out-dir data/processed/cppe5
# 2. See what the untrained base model scores (the bar to beat)
python scripts/baseline_zeroshot.py --config examples/grounding/configs/cppe5_qwen25vl_3b.yaml
# 3. Fine-tune
python scripts/train_vlm.py --config examples/grounding/configs/cppe5_qwen25vl_3b.yaml
```

See [scripts/README.md](scripts/README.md) for the full guide and
[CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) for every YAML field.

## Tests

```bash
pytest                     # fast unit tests (no torch/transformers needed for most)
RUN_VLM_SMOKE=1 pytest     # also runs the end-to-end smoke test (SmolVLM-256M, CPU)
```

The smoke test uses a tiny synthetic grounding dataset under
`tests/fixtures/vlm_tiny/`. It is **generated, not committed** —
`tests/conftest.py` rebuilds it on first use from
[tests/fixtures/build_vlm_tiny.py](tests/fixtures/build_vlm_tiny.py) (a handful of
images with colored rectangles + matching boxes). To (re)build it manually:

```bash
python tests/fixtures/build_vlm_tiny.py
```

## Dataset contract

A prepare script produces a `DatasetDict` (with an `Image` column) plus a
`preprocessing.json` sidecar (and, for grounding, a `labels.json`). Every row is
`(image, prompt, target)` where the **target** is supplied one of two ways:

**Free-form text** (`task: text`):

```python
{
    "image":  PIL.Image,
    "prompt": "Extract this receipt as JSON.",
    "response": '{"menu":[{"name":"Latte","price":"4.50"}]}',   # the exact target string
}
```

**Grounding** (`task: box | point`) — coordinates normalized to `[0, 1]`,
**resolution-independent**:

```python
{
    "image":  PIL.Image,                       # the HF Image feature decodes it
    "prompt": "Detect every PPE item. ...",
    "objects": [
        {"label": "Coverall",
         "box":   [0.12, 0.10, 0.90, 0.97],     # [x1,y1,x2,y2], or [] if absent
         "point": [0.51, 0.53]},                # [x,y], or [] if absent
    ],
    # any extra columns are silently kept (ignored by the trainer)
}
```

A row may carry both; an explicit `response` always wins. Target resolution
happens in one place — [`build_target`](vlm_finetuning/dataset.py) — so the
collator is identical across tasks.

For grounding, storing **normalized** coordinates (not pixel, not token ids) is deliberate — the
same processed dataset works with any backbone and any image-resize policy.
Quantization to the discrete `<box>` grid (default `[0, 1000)`) happens at train
time inside the collator, exactly as GLiNER defers tokenization to train time.
`validate_row` (called by every prepare script) checks the invariants — boxes
ordered and in `[0, 1]`, every object carrying a box and/or point — so schema
bugs surface at prep time, not at training step 1.

## How training works (the SOTA bits that matter)

| Concern | What this project does |
|---|---|
| **Completion-only loss** | The collator masks the prompt, the image placeholder tokens, and padding to `-100`, so loss is computed **only** on the assistant's `<box>` answer. Training on the whole sequence is the #1 VLM-SFT mistake — it quietly wrecks grounding. |
| **LoRA targeting** | `target_modules: auto` resolves to the language model's attention + MLP projections (`q/k/v/o_proj`, `gate/up/down_proj`) by scanning the module tree, **excluding** the vision encoder (which reuses the same names). |
| **Vision tower frozen** | Default `freeze_vision_tower: true` — adapt the LM only. Cheaper, more stable, and enough for grounded-text output. |
| **QLoRA** | `quant.load_in_4bit: true` loads the base in 4-bit NF4 (bitsandbytes) with `prepare_model_for_kbit_training` + paged optimizer — Qwen2.5-VL-3B fits a 16GB T4. |
| **Coordinate scheme** | Plain-text `<box>` tokens on a normalized integer grid — no vocab surgery, works on any backbone. Opt into real special tokens with `add_coord_special_tokens: true` (auto-routed into LoRA `modules_to_save` so the new embedding rows train). |
| **Generation-based eval** | `eval_f1` is measured on **decoded** predictions (the model generates, we parse `<box>` tokens and IoU-match to gold) — not on teacher-forced loss. `metric_for_best_model` and early stopping select the checkpoint that actually grounds best. |
| **Precision auto** | bf16+tf32 on Ampere+, fp16 on Turing/T4, fp32 on CPU — the same YAML runs on a T4 and an H100 without edits. |
| **Crash-safe saves** | The model + `preprocessing.json` are written **before** the test eval, so a crash mid-generation still leaves a loadable checkpoint. |

## Evaluation metric

Eval is generation-based — the model decodes, then we score the decoded output:

- **Box task** — greedy one-to-one IoU matching → precision / recall / **F1 at
  IoU 0.5**, plus `f1_iou_avg` (F1 averaged over the COCO 0.50:0.95 sweep, which
  rewards tight boxes), plus per-label F1.
- **Point task** — a predicted point is a true positive if it lands inside the
  same-label gold box (standard "pointing accuracy"); point-vs-point gold falls
  back to an L2-distance threshold.
- **Text task** — `exact_match` (whitespace-normalized) and `token_f1`
  (SQuAD-style multiset token F1, which credits near-misses like one wrong JSON
  field). Swap in a task-specific scorer by editing
  [`evaluate.score_text`](vlm_finetuning/evaluate.py).

For grounding we deliberately **don't** report mAP: free-form generation produces no calibrated
per-box confidence to rank by, so an area-under-PR number would be meaningless.
F1 at a fixed IoU is the honest metric for a generative detector.

## Fine-tuning paths

The `vlm.lora.enabled` / `vlm.quant.load_in_4bit` switches pick the method:

- **LoRA** (default) — wraps the LM projections with low-rank adapters, freezes
  the rest. ~1% trainable params; ship many per-customer adapters off one base.
  `cppe5_qwen25vl_3b.yaml`.
- **QLoRA** — LoRA on top of a 4-bit base. Same quality target, ~1/3 the memory;
  fits a 16GB T4. `cppe5_qwen25vl_3b_qlora.yaml`.
- **Full FT** — `lora.enabled: false`. Strongest ceiling, biggest footprint;
  consider keeping `freeze_vision_tower: true` regardless.

## Serving

[`VLMPredictor`](vlm_finetuning/serve.py) loads a full-FT save or a LoRA save
through the same constructor (it detects `adapter_config.json`, loads the base,
and merges the adapter for zero inference overhead):

```python
from vlm_finetuning.serve import VLMPredictor, draw_objects

predictor = VLMPredictor("outputs/cppe5_qwen25vl_3b/lora_r16/final")
objs = predictor.predict_one("photo.jpg")           # uses the prompt baked in at training
# [{"label": "Coverall", "box": [...], "box_px": [...], "point": [...]}, ...]

objs = predictor.predict_one("photo.jpg", prompt="Point at every mask.")
draw_objects("photo.jpg", objs).save("annotated.png")  # eyeball the predictions
```

The predictor reads the coordinate scheme, default prompt and generation length
from `preprocessing.json`, and returns objects in both normalized and pixel
coordinates. CLI:

```bash
python -m vlm_finetuning.serve outputs/<run>/final photo.jpg --draw-dir /tmp/preds
```

## Layout

```
vlm_finetuning/   importable package (dataset, model, trainer, evaluate, serve
                  + vendored shared/)
scripts/          train / inference / zero-shot-baseline entry points
examples/         configs and dataset-prep scripts (CPPE-5 worked example)
tests/            unit tests (+ auto-built tests/fixtures/vlm_tiny, gitignored)
data/             processed datasets (gitignored)
```
