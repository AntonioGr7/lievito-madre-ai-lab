"""Build the tiny GLiNER fixture used by the end-to-end smoke test.

Run once after cloning; the resulting DatasetDict is committed under
tests/fixtures/gliner_tiny/. Re-run if the fixture rows change.

Usage:
  python tests/fixtures/build_gliner_tiny.py
"""
import json
from pathlib import Path

from datasets import Dataset, DatasetDict


HERE = Path(__file__).parent / "gliner_tiny"

ROWS = [
    {"text": "Maria Rossi lives in Rome.",
     "spans": [{"start": 0, "end": 11, "label": "PERSON"},
               {"start": 21, "end": 25, "label": "CITY"}]},
    {"text": "Call John at +1-555-0101.",
     "spans": [{"start": 5, "end": 9, "label": "PERSON"},
               {"start": 13, "end": 24, "label": "PHONE"}]},
    {"text": "Send mail to a@b.com.",
     "spans": [{"start": 13, "end": 20, "label": "EMAIL"}]},
    {"text": "She works at Acme Inc.",
     "spans": [{"start": 13, "end": 21, "label": "ORG"}]},
    {"text": "Bob's passport is X1234.",
     "spans": [{"start": 0, "end": 3, "label": "PERSON"},
               {"start": 18, "end": 23, "label": "PASSPORT"}]},
    {"text": "Anna stays in Milan.",
     "spans": [{"start": 0, "end": 4, "label": "PERSON"},
               {"start": 14, "end": 19, "label": "CITY"}]},
    {"text": "Reach me at hello@example.org.",
     "spans": [{"start": 12, "end": 29, "label": "EMAIL"}]},
    {"text": "Globex Corp signed the deal.",
     "spans": [{"start": 0, "end": 11, "label": "ORG"}]},
    {"text": "Phone 555-0199 rings twice.",
     "spans": [{"start": 6, "end": 14, "label": "PHONE"}]},
    {"text": "Dr. Smith reviewed the chart.",
     "spans": [{"start": 4, "end": 9, "label": "PERSON"}]},
]


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    ds = DatasetDict({
        "train": Dataset.from_list(ROWS[:6]),
        "validation": Dataset.from_list(ROWS[6:8]),
        "test": Dataset.from_list(ROWS[8:]),
    })
    ds.save_to_disk(str(HERE))
    (HERE / "train_types.json").write_text(json.dumps(["PERSON", "CITY", "PHONE", "EMAIL", "ORG"]))
    (HERE / "holdout_types.json").write_text(json.dumps(["PASSPORT"]))
    print(f"fixture written to {HERE}")


if __name__ == "__main__":
    main()
