"""Build the tiny free-text (generic SFT) fixture used by the text smoke test.

Mirror of ``build_vlm_tiny.py`` for the ``task: text`` path: each row is an
image + prompt + a free-form ``response`` string (no boxes/points), so the full
load -> LoRA -> train -> generate-eval path is exercised for the generic
image->text contract without downloading a real dataset. Scored by
exact-match / token-F1. Generated output, not committed — `tests/conftest.py`
rebuilds it on first use.

Usage:
  python tests/fixtures/build_vlm_text_tiny.py
"""
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage, ImageDraw

HERE = Path(__file__).parent / "vlm_text_tiny"
SIZE = 128
PROMPT = "What color is the box and where is it? Answer in one short sentence."

# (color name, RGB, normalized [x1,y1,x2,y2], position phrase) per single-box image.
_SPECS = [
    ("red", (220, 30, 30), [0.10, 0.10, 0.45, 0.45], "top-left"),
    ("blue", (30, 30, 220), [0.55, 0.55, 0.90, 0.90], "bottom-right"),
    ("red", (220, 30, 30), [0.55, 0.10, 0.90, 0.45], "top-right"),
    ("blue", (30, 30, 220), [0.10, 0.55, 0.45, 0.90], "bottom-left"),
    ("red", (220, 30, 30), [0.30, 0.30, 0.70, 0.70], "center"),
    ("blue", (30, 30, 220), [0.05, 0.05, 0.40, 0.40], "top-left"),
    ("red", (220, 30, 30), [0.60, 0.60, 0.95, 0.95], "bottom-right"),
    ("blue", (30, 30, 220), [0.55, 0.10, 0.90, 0.45], "top-right"),
    ("red", (220, 30, 30), [0.10, 0.55, 0.45, 0.90], "bottom-left"),
    ("blue", (30, 30, 220), [0.30, 0.30, 0.70, 0.70], "center"),
]


def _render(color_rgb, box) -> PILImage.Image:
    img = PILImage.new("RGB", (SIZE, SIZE), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = box
    draw.rectangle([x1 * SIZE, y1 * SIZE, x2 * SIZE, y2 * SIZE], fill=color_rgb)
    return img


def _rows(specs):
    images, prompts, responses = [], [], []
    for color, rgb, box, pos in specs:
        images.append(_render(rgb, box))
        prompts.append(PROMPT)
        responses.append(f"A {color} box in the {pos}.")
    return {"image": images, "prompt": prompts, "response": responses}


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    features = Features({
        "image": Image(),
        "prompt": Value("string"),
        "response": Value("string"),
    })
    ds = DatasetDict({
        "train": Dataset.from_dict(_rows(_SPECS[:6]), features=features),
        "validation": Dataset.from_dict(_rows(_SPECS[6:8]), features=features),
        "test": Dataset.from_dict(_rows(_SPECS[8:]), features=features),
    })
    ds.save_to_disk(str(HERE))
    (HERE / "preprocessing.json").write_text(json.dumps({
        "source": "synthetic-tiny", "task": "text", "default_prompt": PROMPT,
    }, indent=2))
    print(f"fixture written to {HERE}")


if __name__ == "__main__":
    main()
