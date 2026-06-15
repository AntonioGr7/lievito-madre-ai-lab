"""Build the tiny COCO-format fixture used by the end-to-end smoke test.

A handful of small synthetic images with colored rectangles and matching COCO
pixel boxes, so the full train → mAP-eval path runs on CPU without downloading a
real dataset. Generated output, not committed — rebuilt on demand by conftest.

Usage:
  python tests/fixtures/build_coco_tiny.py
"""
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
from PIL import Image as PILImage, ImageDraw

HERE = Path(__file__).parent / "coco_tiny"
SIZE = 128
LABELS = ["red box", "blue box"]

# (category_id, color, [x,y,w,h] in pixels) per object.
_SPECS = [
    [(0, (220, 30, 30), [12, 12, 45, 45])],
    [(1, (30, 30, 220), [70, 70, 45, 45])],
    [(0, (220, 30, 30), [25, 70, 40, 40]), (1, (30, 30, 220), [70, 12, 38, 38])],
    [(1, (30, 30, 220), [40, 40, 50, 50])],
    [(0, (220, 30, 30), [6, 6, 45, 45])],
    [(0, (220, 30, 30), [78, 78, 44, 44])],
    [(1, (30, 30, 220), [18, 76, 40, 40])],
    [(0, (220, 30, 30), [64, 18, 40, 40])],
    [(1, (30, 30, 220), [30, 30, 55, 55])],
    [(0, (220, 30, 30), [45, 45, 50, 50])],
]


def _render(spec) -> PILImage.Image:
    img = PILImage.new("RGB", (SIZE, SIZE), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    for _, color, (x, y, w, h) in spec:
        draw.rectangle([x, y, x + w, y + h], fill=color)
    return img


def _rows(specs, start_id):
    images, image_ids, objects = [], [], []
    for i, spec in enumerate(specs):
        images.append(_render(spec))
        image_ids.append(start_id + i)
        objects.append({
            "bbox": [[float(v) for v in b] for _, _, b in spec],
            "category_id": [int(c) for c, _, _ in spec],
            "area": [float(b[2] * b[3]) for _, _, b in spec],
            "iscrowd": [0 for _ in spec],
        })
    return {"image": images, "image_id": image_ids, "objects": objects}


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    features = Features({
        "image": Image(),
        "image_id": Value("int64"),
        "objects": {
            "bbox": Sequence(Sequence(Value("float32"))),
            "category_id": Sequence(Value("int64")),
            "area": Sequence(Value("float32")),
            "iscrowd": Sequence(Value("int64")),
        },
    })
    ds = DatasetDict({
        "train": Dataset.from_dict(_rows(_SPECS[:6], 0), features=features),
        "validation": Dataset.from_dict(_rows(_SPECS[6:8], 6), features=features),
        "test": Dataset.from_dict(_rows(_SPECS[8:], 8), features=features),
    })
    ds.save_to_disk(str(HERE))
    (HERE / "labels.json").write_text(json.dumps(LABELS))
    (HERE / "preprocessing.json").write_text(json.dumps({
        "source": "synthetic-tiny", "task": "object-detection", "labels": LABELS,
    }, indent=2))
    print(f"fixture written to {HERE}")


if __name__ == "__main__":
    main()
