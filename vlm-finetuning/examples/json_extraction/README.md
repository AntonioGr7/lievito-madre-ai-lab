# Free-text example — CORD-v2 (receipt → JSON)

Fine-tune a VLM to read a receipt image and emit its contents as a **JSON
string** — the generic `task: text` path. No boxes, no points: each row is just
`{"image", "prompt", "response}` where `response` is the exact target text. This
is here to show the project is a general image→text SFT lab; grounding (the
`grounding/` example) is one specialization, this is another.

Dataset: [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2)
(~1k receipt photos with structured ground-truth JSON).

```bash
bash examples/json_extraction/run.sh
# or step by step:
python examples/json_extraction/dataset/prepare_cord.py --out-dir data/processed/cord
python scripts/train_vlm.py --config examples/json_extraction/configs/cord_qwen25vl_3b.yaml
```

## What changes vs grounding

| | Grounding (`box`/`point`) | This (`text`) |
|---|---|---|
| Row target | `objects: [{label, box, point}]` | `response: "<exact string>"` |
| Serialization | objects → `<box>` tokens (at train time) | none — the string *is* the target |
| Eval metric | IoU / pointing F1 | `exact_match` + `token_f1` |
| Everything else | — identical (`load_vlm`, completion-only collator, `Trainer`, generation) — | |

`metric_for_best_model: token_f1` selects the checkpoint with the best
generated-vs-gold token overlap. Final numbers land in
`outputs/<run>/final/test_metrics.json` as `test_exact_match` / `test_token_f1`.

## Your own task (tool calls, captions, VQA, …)

Copy `dataset/prepare_cord.py`. Emit rows of `{"image", "prompt", "response"}`
where `response` is whatever the assistant should say — a tool call, a caption,
an answer — set `vlm.task: text`, and (if exact-match/token-F1 isn't the right
yardstick) swap the scorer in
[`evaluate.score_text`](../../vlm_finetuning/evaluate.py) for a task-specific one
(JSON structural match, BLEU, …). The training engine doesn't change.
