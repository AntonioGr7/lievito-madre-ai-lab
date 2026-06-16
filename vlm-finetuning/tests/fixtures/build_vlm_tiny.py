"""Build the tiny grounding fixture used by the end-to-end smoke test.

Generates a handful of small synthetic images with colored rectangles and the
matching normalized boxes/points, so the full train → generate-eval path can run
on CPU in the smoke test without downloading a real dataset. The resulting
DatasetDict is generated output, not committed — `tests/conftest.py` rebuilds it
on first use.

Usage:
  python tests/fixtures/build_vlm_tiny.py
"""
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
from PIL import Image as PILImage, ImageDraw

HERE = Path(__file__).parent / "vlm_tiny"
SIZE = 128
PROMPT = (
    "Detect every colored box. Possible categories: red box, blue box. "
    "For each, output its label followed by its bounding box as <box> tokens."
)

# (label, color, normalized [x1,y1,x2,y2]) per object; one or two per image.
_SPECS = [
    [("red box", (220, 30, 30), [0.10, 0.10, 0.45, 0.45])],
    [("blue box", (30, 30, 220), [0.55, 0.55, 0.90, 0.90])],
    [("red box", (220, 30, 30), [0.20, 0.55, 0.50, 0.85]),
     ("blue box", (30, 30, 220), [0.55, 0.10, 0.85, 0.40])],
    [("blue box", (30, 30, 220), [0.30, 0.30, 0.70, 0.70])],
    [("red box", (220, 30, 30), [0.05, 0.05, 0.40, 0.40])],
    [("red box", (220, 30, 30), [0.60, 0.60, 0.95, 0.95])],
    [("blue box", (30, 30, 220), [0.15, 0.60, 0.45, 0.90])],
    [("red box", (220, 30, 30), [0.50, 0.15, 0.80, 0.45])],
    [("blue box", (30, 30, 220), [0.25, 0.25, 0.65, 0.65])],
    [("red box", (220, 30, 30), [0.35, 0.35, 0.75, 0.75])],
]


def _render(spec) -> PILImage.Image:
    img = PILImage.new("RGB", (SIZE, SIZE), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    for _, color, (x1, y1, x2, y2) in spec:
        draw.rectangle(
            [x1 * SIZE, y1 * SIZE, x2 * SIZE, y2 * SIZE], fill=color
        )
    return img


def _rows(specs):
    images, prompts, objects = [], [], []
    for spec in specs:
        images.append(_render(spec))
        prompts.append(PROMPT)
        objects.append([
            {"label": lbl,
             "box": box,
             "point": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]}
            for lbl, _, box in spec
        ])
    return {"image": images, "prompt": prompts, "objects": objects}


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    features = Features({
        "image": Image(),
        "prompt": Value("string"),
        # A variable-length list of structs (one dict per object). Use the
        # list-of-feature form `[{...}]`, NOT `Sequence({...})`: Sequence of a
        # dict transposes to a struct-of-arrays (dict-of-lists), but the loader
        # iterates `row["objects"]` expecting a list of dicts.
        "objects": [{
            "label": Value("string"),
            "box": Sequence(Value("float32")),
            "point": Sequence(Value("float32")),
        }],
    })
    ds = DatasetDict({
        "train": Dataset.from_dict(_rows(_SPECS[:6]), features=features),
        "validation": Dataset.from_dict(_rows(_SPECS[6:8]), features=features),
        "test": Dataset.from_dict(_rows(_SPECS[8:]), features=features),
    })
    ds.save_to_disk(str(HERE))
    (HERE / "labels.json").write_text(json.dumps(["red box", "blue box"]))
    (HERE / "preprocessing.json").write_text(json.dumps({
        "source": "synthetic-tiny", "task": "box", "coord_bins": 1000,
        "labels": ["red box", "blue box"], "default_prompt": PROMPT,
    }, indent=2))
    print(f"fixture written to {HERE}")


if __name__ == "__main__":
    main()
