"""Quick local inference for a Trainer-saved GLiNER model.

Prints extracted spans for a hardcoded multi-label paragraph using the
train_types stamped on the model, plus an open-vocabulary call using the
held-out entity types.
"""
from lievito_madre_ai_lab.encoder.gliner_entity_extraction import serve as _serve

print(f"[serve loaded from: {_serve.__file__}]")
GLiNERPredictor = _serve.GLiNERPredictor

MODEL_DIR = "outputs/pii_gliner/gliner_exp_01/final"
TEXT = (
    "Project Manager Marcus Sterling (Employee Ref: EMP/882-99-XJ) flagged a "
    "high-risk entry for Sophia-Maria de la Cruz (Tax ID: 998-00-1122), "
    "located at 7722 S. O'Connor Dr, Apt #12-B, Las Vegas, NV 89101. The "
    "security logs from May 14, 2026, confirm an unauthorized access attempt "
    "via IP address 172.16.254.1 and the recovery email "
    "s.m.cruz_temp_99@dev-node.io. A subsequent wire transfer of $12,450.75 "
    "was tied to account #AC-1192-3304-8821."
)

predictor = GLiNERPredictor(MODEL_DIR, use_compile=False, quantize_cpu=False, warmup_steps=0)

print("\n=== Closed-set inference (train_types) ===")
print(f"Labels: {predictor.train_types}")
for s in predictor.predict_one(TEXT):
    print(f"  [{s['label']:24s}] {s['text']!r:30s}  ({s['start']}, {s['end']})  score={s['score']:.3f}")

if predictor.holdout_types:
    print("\n=== Open-vocabulary inference (holdout_types) ===")
    print(f"Labels: {predictor.holdout_types}")
    for s in predictor.predict_one(TEXT, labels=predictor.holdout_types):
        print(f"  [{s['label']:24s}] {s['text']!r:30s}  ({s['start']}, {s['end']})  score={s['score']:.3f}")
