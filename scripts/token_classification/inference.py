"""Quick local inference for a Trainer-saved token classifier.

Prints both the extracted entity spans and a per-token diagnostic table —
handy for sanity-checking a checkpoint. Span extraction goes through the same
``TokenClassificationPredictor`` used in production so the output here matches
``serve.py`` exactly; the per-token table is a separate diagnostic that reuses
the predictor's tokenizer and model so the checkpoint is only loaded once.
"""
import torch

from lievito_madre_ai_lab.encoder.token_classification import serve as _serve
print(f"[serve loaded from: {_serve.__file__}]")
TokenClassificationPredictor = _serve.TokenClassificationPredictor

MODEL_DIR = "outputs/pii/"
TEXT = """"
Project Manager The Honorable Marcus-Aurelius V. Sterling-Holloway (Employee Ref: EMP/882-99-XJ) flagged a high-risk entry for Sophia-Maria de la Cruz (Tax ID: 998-00-1122), currently located at 7722 S. O'Connor Dr, Apt #12-B, Las Vegas, NV 89101. The security logs from May 14, 2026, confirm an unauthorized access attempt via IP address 172.16.254.1 and the recovery email s.m.cruz_temp_99@dev-node.io. A subsequent wire transfer of $12,450.75 was tied to account #AC-1192-3304-8821, verified through the passport document #P44229910. The system last pinged a cellular device at +44 (20) 7946-0958 at the coordinates 36.1699° N, 115.1398° W, which is inconsistent with the user's primary residence.
"""

predictor = TokenClassificationPredictor(
    MODEL_DIR, use_compile=False, quantize_cpu=False, warmup_steps=0
)
tok = predictor.tokenizer
model = predictor._model
id2label = predictor.id2label

enc = tok(TEXT, return_tensors="pt", return_offsets_mapping=True)
offsets = enc.pop("offset_mapping")[0].tolist()
word_ids = enc.word_ids(batch_index=0)

with torch.inference_mode():
    logits = model(**enc).logits[0]

probs = logits.softmax(-1)
top_p, preds = probs.max(-1)
tokens = tok.convert_ids_to_tokens(enc["input_ids"][0].tolist())

print(f"{'token':20s} {'offset':12s} {'word':>5s}  {'pred':24s} prob")
for t, off, wid, p, prob in zip(tokens, offsets, word_ids, preds.tolist(), top_p.tolist()):
    flag = " (special)" if wid is None else ""
    wid_s = "-" if wid is None else str(wid)
    print(f"  {t!r:20s} {str(off):12s} {wid_s:>5s}  {id2label[p]:24s} {prob:.3f}{flag}")

spans = predictor.predict_one(TEXT)
print(f"\n{TEXT}")
if spans:
    for s in spans:
        print(f"  [{s['label']:24s}] {s['text']!r:30s}  ({s['start']}, {s['end']})  score={s['score']:.3f}")
else:
    print("  (no entities found)")
