# FiQA Bi-Encoder

Fine-tune an Ettin-68m bi-encoder on FiQA-2018 train, then benchmark against the RTEB-finance English open subset (FinanceBench, HC3Finance, FinQA). FiQA is NOT part of RTEB-finance, so training on it does not leak into the evaluation.

## Run

```bash
bash examples/embedding_bi_encoder/fiqa/run.sh
```

Or step by step:

```bash
python examples/embedding_bi_encoder/fiqa/prepare_fiqa.py
python scripts/embedding_bi_encoder/train_bi_encoder.py \
    --config examples/embedding_bi_encoder/fiqa/configs/fiqa_ettin68m.yaml
python examples/embedding_bi_encoder/eval_rteb_finance.py \
    --model-dir outputs/bi_encoder_fiqa_ettin68m/exp_01/final
```

To upgrade to the 150m Ettin variant, change `model_name` and bump `experiment_id` in [configs/fiqa_ettin68m.yaml](configs/fiqa_ettin68m.yaml); every other knob is dimension-agnostic.
