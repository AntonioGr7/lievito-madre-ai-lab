# GLiNER PII Entity Extraction

Fine-tune [GLiNER](https://github.com/urchade/GLiNER) for PII entity extraction. Two data sources are interchangeable — pick the one whose label inventory matches your downstream need:

- **`prepare_openpii.py`** — [ai4privacy/open-pii-masking-500k-ai4privacy](https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy) (default).
- **`prepare_nemotron.py`** — [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII).

GLiNER 0.2.x requires `transformers<5.2.0`, which conflicts with the encoder/decoder/vision groups. Install in its own virtualenv:

```bash
pip install -e ".[gliner]"
```

## Run

```bash
bash examples/gliner_entity_extraction/pii/run.sh
```

Or step by step:

```bash
# Pick one prep script:
python examples/gliner_entity_extraction/pii/prepare_openpii.py --out-dir data/processed/pii-gliner
# or:
python examples/gliner_entity_extraction/pii/prepare_nemotron.py --out-dir data/processed/pii-gliner

python scripts/gliner_entity_extraction/train_gliner.py \
    --config examples/gliner_entity_extraction/pii/configs/pii_gliner.yaml
```

## Config variants

Four recipes live in [configs/](configs/), all targeting the same PII task with different model / hardware tradeoffs:

| Config            | When to use |
|-------------------|-------------|
| `smoke.yaml`      | Quick wiring check (1 epoch, tiny subset). |
| `t4_small.yaml`   | T4-class GPU (16 GB) — smaller backbone, fp16. |
| `a10_medium.yaml` | A10/A100-class (24+ GB) — bigger backbone, bf16. |
| `pii_gliner.yaml` | The default reference recipe. |

Swap configs by changing the `--config` flag — the dataset contract is the same across all four.
