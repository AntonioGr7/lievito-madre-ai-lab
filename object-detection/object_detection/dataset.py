"""COCO-format dataset contract for object-detection fine-tuning.

Each row in train / validation / test splits has the shape::

    {
      "image":    PIL.Image,                 # decoded by the HF Image feature
      "image_id": int,                        # unique per image
      "objects": {                            # parallel lists (COCO layout)
          "bbox":        [[x, y, w, h], ...], # ABSOLUTE pixels, COCO xywh
          "category_id": [int, ...],          # contiguous 0..K-1
          "area":        [float, ...],        # optional; recomputed if absent
          "iscrowd":     [int, ...],          # optional; defaults to 0
      },
    }

Boxes are stored in **absolute pixel COCO `xywh`** — the format HF image
processors (`DetrImageProcessor`, `RTDetrImageProcessor`, `DFineImageProcessor`)
consume directly, and the format Albumentations augments natively. The processor
handles resize / normalize / pad and rescales the annotations to match, so the
dataset stays independent of any one model's input resolution.

Prepare scripts (see ``examples/``) call :func:`validate_row` before saving.
:func:`load_processed` reads the saved DatasetDict back plus ``labels.json``
(the ordered category names) and returns an ``id2label`` map.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# --- geometry -------------------------------------------------------------


def coco_to_xyxy(bbox: list[float]) -> list[float]:
    """COCO ``[x, y, w, h]`` → ``[x1, y1, x2, y2]`` (absolute pixels)."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def xyxy_to_coco(box: list[float]) -> list[float]:
    """``[x1, y1, x2, y2]`` → COCO ``[x, y, w, h]``."""
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


# --- contract validation + IO ---------------------------------------------


def _num_list(seq: Any) -> bool:
    return isinstance(seq, (list, tuple)) and all(isinstance(v, (int, float)) for v in seq)


def validate_row(row: dict[str, Any]) -> list[str]:
    """Return a list of error strings (empty == OK) for *row*.

    Checks the COCO contract: ``image`` present, integer ``image_id``, and an
    ``objects`` dict whose ``bbox`` / ``category_id`` lists are parallel, with
    positive-area boxes in pixel ``xywh``. Extra keys are ignored; the image is
    only checked for presence (decoding is the loader's job).
    """
    errors: list[str] = []
    if "image" not in row or row["image"] is None:
        errors.append("row missing required key 'image'")
    if "image_id" not in row:
        errors.append("row missing required key 'image_id'")
    elif not isinstance(row["image_id"], int):
        errors.append(f"row['image_id'] must be int, got {type(row['image_id']).__name__}")

    objs = row.get("objects")
    if objs is None:
        errors.append("row missing required key 'objects'")
        return errors
    if not isinstance(objs, dict):
        errors.append(f"row['objects'] must be a dict of parallel lists, got {type(objs).__name__}")
        return errors
    bboxes = objs.get("bbox")
    cats = objs.get("category_id")
    if bboxes is None or cats is None:
        errors.append("row['objects'] must contain 'bbox' and 'category_id' lists")
        return errors
    if len(bboxes) != len(cats):
        errors.append(
            f"objects.bbox and objects.category_id must be parallel; "
            f"got {len(bboxes)} vs {len(cats)}"
        )
    for i, b in enumerate(bboxes):
        if not _num_list(b) or len(b) != 4:
            errors.append(f"objects.bbox[{i}] must be 4 numbers [x,y,w,h], got {b!r}")
            continue
        x, y, w, h = b
        if w <= 0 or h <= 0:
            errors.append(f"objects.bbox[{i}] must have w>0 and h>0, got {b!r}")
        if x < 0 or y < 0:
            errors.append(f"objects.bbox[{i}] must have x>=0 and y>=0, got {b!r}")
    for i, c in enumerate(cats):
        if not isinstance(c, int) or c < 0:
            errors.append(f"objects.category_id[{i}] must be a non-negative int, got {c!r}")
    return errors


def load_processed(processed_dir: str | Path):
    """Load a processed detection DatasetDict + the category list.

    Returns ``(datasets, id2label)`` where ``id2label`` maps contiguous integer
    ids → category names (read from ``labels.json``). Validates the first
    non-empty row of each split against :func:`validate_row`.
    """
    from datasets import load_from_disk

    processed_dir = Path(processed_dir)
    datasets = load_from_disk(str(processed_dir))

    labels_path = processed_dir / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"missing {labels_path}. Did the prepare script run to completion? "
            f"It must write the ordered category names."
        )
    names: list[str] = json.loads(labels_path.read_text())
    id2label = {i: n for i, n in enumerate(names)}

    for split_name, split in datasets.items():
        if len(split) == 0:
            continue
        errors = validate_row(split[0])
        if errors:
            raise ValueError(
                f"split {split_name!r} row 0 violates the COCO contract:\n  - "
                + "\n  - ".join(errors)
            )
    return datasets, id2label


# --- augmentation + collation ---------------------------------------------


def build_transforms(
    *,
    train: bool,
    hflip: float = 0.5,
    brightness_contrast: float = 0.5,
    hue_sat: float = 0.3,
    min_area: float = 1.0,
    min_visibility: float = 0.1,
):
    """Build an Albumentations pipeline that augments image **and** boxes together.

    Returns ``None`` for ``train=False`` — eval/test feed the image processor
    untouched, so metrics reflect the real images. Geometric resize is left to
    the image processor (it rescales the annotations to match), so this pipeline
    only does flips + photometric jitter — the safe, label-preserving augments
    for detection. ``BboxParams(clip=True, min_area, min_visibility)`` drops boxes
    an augment pushes (almost) out of frame so we never train on phantom targets.

    Tune via the ``detection.augmentation`` YAML block; set probabilities to 0 to
    disable individual ops.
    """
    if not train:
        return None
    import albumentations as A

    ops = []
    if hflip > 0:
        ops.append(A.HorizontalFlip(p=hflip))
    if brightness_contrast > 0:
        ops.append(A.RandomBrightnessContrast(p=brightness_contrast))
    if hue_sat > 0:
        ops.append(A.HueSaturationValue(p=hue_sat))
    return A.Compose(
        ops,
        bbox_params=A.BboxParams(
            format="coco", label_fields=["category_id"],
            clip=True, min_area=min_area, min_visibility=min_visibility,
        ),
    )


def make_collate_fn(image_processor, transforms=None):
    """Build a collate_fn that augments, formats COCO annotations, and runs the
    image processor (resize + normalize + pad + box rescale) for one batch.

    The processor returns ``pixel_values`` (+ ``pixel_mask``) and ``labels`` (a
    list of per-image dicts with normalized ``cxcywh`` boxes, class labels, and
    sizes) — exactly what DETR-family ``forward`` consumes to compute the
    Hungarian-matching loss.
    """
    import numpy as np

    def collate(batch: list[dict]) -> dict:
        images, annotations = [], []
        for ex in batch:
            img = np.array(ex["image"].convert("RGB"))
            bboxes = list(ex["objects"]["bbox"])
            cats = list(ex["objects"]["category_id"])
            if transforms is not None:
                out = transforms(image=img, bboxes=bboxes, category_id=cats)
                img, bboxes, cats = out["image"], out["bboxes"], out["category_id"]
            anns = [
                {
                    "image_id": ex["image_id"],
                    "category_id": int(c),
                    "bbox": [float(v) for v in b],
                    "area": float(b[2] * b[3]),
                    "iscrowd": 0,
                }
                for b, c in zip(bboxes, cats)
            ]
            images.append(img)
            annotations.append({"image_id": ex["image_id"], "annotations": anns})
        return image_processor(images=images, annotations=annotations, return_tensors="pt")

    return collate
