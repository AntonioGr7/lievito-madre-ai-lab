# Bi-Encoder Smoke Test

End-to-end check that the bi-encoder pipeline is wired correctly. Builds a 100-row synthetic dataset, trains for 1 epoch, and verifies the saved model can tell a paraphrase from an unrelated sentence.

Reference for all 8 bi-encoder recipes — each `configs/smoke_r<N>.yaml` is the minimal complete YAML for Recipe N (see [scripts/README.md](../../scripts/README.md)).

## Run

```bash
# Recipe 1 only — fastest sanity check (~1-2 min on CPU, ~30s on GPU).
bash examples/smoke/run.sh
```

```bash
# All 8 recipes end-to-end (mining + scoring side-stages + train + verify).
# First run downloads ~150 MB of HF models; then ~1-2 min per recipe on CPU.
bash examples/smoke/run_all.sh           # all 8
bash examples/smoke/run_all.sh 3         # just Recipe 3
bash examples/smoke/run_all.sh 2 3 6 7   # subset
```

## Layout

```
smoke/
├── prepare_smoke.py    # build the synthetic dataset
├── verify_smoke.py     # per-recipe assertions on the trained model
├── run.sh              # Recipe 1 only
├── run_all.sh          # all 8 recipes
└── configs/
    └── smoke_r1.yaml ... smoke_r8.yaml
```
