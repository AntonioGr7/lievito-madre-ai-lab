"""Dataset contract for VLM supervised fine-tuning.

The core is a **generic** image→text SFT contract; *grounding* (boxes/points) is
one specialization layered on top. Every row is ``(image, prompt, target)`` where
the target is supplied in one of two ways:

**Free-form text** (``task: text`` — tool calls, JSON extraction, captioning, VQA)::

    {
      "image":  PIL.Image,
      "prompt": str,                 # the user instruction
      "response": str,               # the exact assistant target text
    }

**Grounding** (``task: box`` | ``point`` — detect/point at objects)::

    {
      "image":  PIL.Image,
      "prompt": str,
      "objects": [                   # may be empty
          {"label": str,
           "box":   [x1, y1, x2, y2],   # normalized floats in [0, 1], or [] if absent
           "point": [x, y]},            # normalized floats in [0, 1], or [] if absent
          ...
      ],
    }

A row may carry both; an explicit ``response`` always wins over serialized
``objects``. Any extra columns are silently kept (ignored by the trainer).

For grounding, coordinates are stored **resolution-independent** (normalized to
``[0, 1]``) so the same processed dataset works with any backbone and any
image-resize policy — mirroring how the GLiNER project stores char-offset spans
rather than token ids. The mapping to discrete ``<box>`` coordinate tokens
(integers on a ``coord_bins`` grid, default ``[0, 1000)``) happens at *train*
time inside the collator, via :func:`build_target`, not at prep time::

    Coverall<box> 120, 340, 880, 960 </box>       # task=box: one line per box
    Coverall<box> 500, 650 </box>                  # task=point: two numbers

Prepare scripts (see ``examples/``) call :func:`validate_row` before saving to
surface schema bugs at prep time. :func:`load_processed` reads the saved
DatasetDict back, plus the ``labels.json`` written alongside it (grounding only).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# --- coordinate tokens ----------------------------------------------------

DEFAULT_COORD_BINS = 1000
BOX_OPEN = "<box>"
BOX_CLOSE = "</box>"

# Tolerant of any internal whitespace/spacing — `<box> 1, 2, 3, 4 </box>`,
# `<box>1,2</box>`, newlines, etc. Label is everything up to the opening tag
# that is not itself a tag character or a line break.
_OBJECT_RE = re.compile(
    r"([^<>\n\r]*?)\s*" + re.escape(BOX_OPEN) + r"\s*([0-9,\s]*?)\s*" + re.escape(BOX_CLOSE)
)


def quantize(value: float, coord_bins: int = DEFAULT_COORD_BINS) -> int:
    """Map a normalized coordinate in ``[0, 1]`` onto the integer grid
    ``[0, coord_bins)``. Out-of-range inputs are clamped (a box that bleeds a
    hair past the image edge shouldn't crash serialization)."""
    q = round(float(value) * (coord_bins - 1))
    return max(0, min(coord_bins - 1, q))


def dequantize(token: int, coord_bins: int = DEFAULT_COORD_BINS) -> float:
    """Inverse of :func:`quantize` — grid integer → normalized ``[0, 1]`` float."""
    f = float(token) / (coord_bins - 1)
    return max(0.0, min(1.0, f))


def _coords_for(obj: dict, task: str) -> list[float] | None:
    """Pick the coordinate list for *obj* under *task*.

    ``task="box"`` wants a 4-tuple; ``task="point"`` wants a 2-tuple and will
    fall back to the box centre when only a box is present (so a detection
    dataset can train a pointer without a separate annotation pass).
    """
    box = obj.get("box") or None
    point = obj.get("point") or None
    if task == "box":
        return list(box) if box else None
    if task == "point":
        if point:
            return list(point)
        if box and len(box) == 4:
            return [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]
        return None
    raise ValueError(f"task must be 'box' or 'point'; got {task!r}")


def format_object(obj: dict, *, task: str, coord_bins: int = DEFAULT_COORD_BINS) -> str | None:
    """Serialize one object to ``label<box> n, n[, n, n] </box>`` or ``None``
    if the object carries no usable coordinate for *task*."""
    coords = _coords_for(obj, task)
    if not coords:
        return None
    nums = ", ".join(str(quantize(c, coord_bins)) for c in coords)
    label = str(obj.get("label", "")).strip()
    return f"{label}{BOX_OPEN} {nums} {BOX_CLOSE}"


def serialize_objects(
    objects: list[dict],
    *,
    task: str,
    coord_bins: int = DEFAULT_COORD_BINS,
    empty_text: str = "No objects detected.",
    sep: str = "\n",
) -> str:
    """Render a list of objects to the assistant target text.

    Objects with no coordinate for *task* are skipped; if that leaves nothing,
    *empty_text* is emitted so the model is explicitly supervised on the
    "found nothing" case rather than on an empty string.
    """
    parts = [s for o in objects if (s := format_object(o, task=task, coord_bins=coord_bins))]
    return sep.join(parts) if parts else empty_text


def build_target(
    row: dict,
    *,
    task: str,
    coord_bins: int = DEFAULT_COORD_BINS,
    empty_text: str = "No objects detected.",
) -> str:
    """Resolve a row's assistant target text — the single place the generic and
    grounding paths converge.

    An explicit ``response`` string always wins (free-form SFT: tool calls, JSON,
    captions). Otherwise the row is treated as grounding and its ``objects`` are
    serialized under *task*. ``task="text"`` with no ``response`` is a contract
    error (caught by :func:`validate_row` at prep time, re-raised here as a guard).
    """
    resp = row.get("response")
    if resp is not None:
        return str(resp)
    if task == "text":
        raise ValueError(
            "task='text' requires a 'response' field on every row; "
            "row has neither 'response' nor a serializable target."
        )
    return serialize_objects(
        row.get("objects", []) or [], task=task, coord_bins=coord_bins, empty_text=empty_text,
    )


def parse_objects(text: str, *, coord_bins: int = DEFAULT_COORD_BINS) -> list[dict]:
    """Parse model-generated text back into structured objects.

    Each ``label<box> ... </box>`` match becomes ``{"label", "box"|"point"}``
    with coordinates de-quantized to normalized ``[0, 1]`` floats. Matches with
    4 numbers are boxes, 2 are points; any other arity is skipped (a truncated
    or malformed generation shouldn't sink the whole parse). Robust to leading
    separators, list bullets, and arbitrary whitespace.
    """
    out: list[dict] = []
    for m in _OBJECT_RE.finditer(text):
        label = m.group(1).strip().lstrip("-*•").strip()
        nums = [int(n) for n in re.findall(r"\d+", m.group(2))]
        if len(nums) == 4:
            out.append({"label": label, "box": [dequantize(n, coord_bins) for n in nums]})
        elif len(nums) == 2:
            out.append({"label": label, "point": [dequantize(n, coord_bins) for n in nums]})
    return out


# --- geometry -------------------------------------------------------------


def iou(box_a: list[float], box_b: list[float]) -> float:
    """Intersection-over-union of two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def point_in_box(point: list[float], box: list[float]) -> bool:
    """True if ``[x, y]`` lies inside the closed ``[x1, y1, x2, y2]`` box."""
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


# --- contract validation + IO ---------------------------------------------


def _is_unit_pair(seq: Any, n: int) -> bool:
    return (
        isinstance(seq, (list, tuple))
        and len(seq) == n
        and all(isinstance(v, (int, float)) for v in seq)
    )


def validate_row(row: dict[str, Any]) -> list[str]:
    """Return a list of error strings (empty == OK) for *row*.

    Every row needs ``image`` + ``prompt`` and exactly one kind of target:
    either a free-form ``response`` string (generic SFT) or an ``objects`` list
    (grounding). For grounding, coordinates must be normalized to ``[0, 1]`` and
    boxes ordered (``x2 > x1``, ``y2 > y1``). The image itself is only checked
    for presence (decoding is the loader's job).
    """
    errors: list[str] = []
    if "image" not in row or row["image"] is None:
        errors.append("row missing required key 'image'")
    if "prompt" not in row:
        errors.append("row missing required key 'prompt'")
    elif not isinstance(row["prompt"], str) or not row["prompt"].strip():
        errors.append("row['prompt'] must be a non-empty str")

    has_response = row.get("response") is not None
    has_objects = "objects" in row
    if has_response:
        if not isinstance(row["response"], str) or not row["response"].strip():
            errors.append("row['response'] must be a non-empty str")
        return errors
    if not has_objects:
        errors.append("row needs a target: either a 'response' str or an 'objects' list")
        return errors
    if not isinstance(row["objects"], list):
        errors.append(f"row['objects'] must be list, got {type(row['objects']).__name__}")
        return errors

    for i, obj in enumerate(row["objects"]):
        if not isinstance(obj, dict):
            errors.append(f"objects[{i}] must be dict, got {type(obj).__name__}")
            continue
        label = obj.get("label")
        if not isinstance(label, str) or not label.strip():
            errors.append(f"objects[{i}] label must be a non-empty str")
        box = obj.get("box") or None
        point = obj.get("point") or None
        if box is None and point is None:
            errors.append(f"objects[{i}] needs at least one of 'box' or 'point'")
        if box is not None:
            if not _is_unit_pair(box, 4):
                errors.append(f"objects[{i}] box must be 4 numbers, got {box!r}")
            else:
                x1, y1, x2, y2 = box
                if not all(0.0 <= v <= 1.0 for v in box):
                    errors.append(f"objects[{i}] box must be normalized to [0,1], got {box!r}")
                if x2 <= x1 or y2 <= y1:
                    errors.append(f"objects[{i}] box must satisfy x2>x1 and y2>y1, got {box!r}")
        if point is not None:
            if not _is_unit_pair(point, 2):
                errors.append(f"objects[{i}] point must be 2 numbers, got {point!r}")
            elif not all(0.0 <= v <= 1.0 for v in point):
                errors.append(f"objects[{i}] point must be normalized to [0,1], got {point!r}")
    return errors


def collect_labels(raw, objects_col: str = "objects") -> list[str]:
    """Scan every split of *raw* and return the sorted set of distinct labels."""
    seen: set[str] = set()
    for split in raw.values():
        for row in split:
            for obj in row.get(objects_col, []) or []:   # text-task rows have no objects
                lbl = obj.get("label") if isinstance(obj, dict) else None
                if lbl:
                    seen.add(lbl)
    return sorted(seen)


def load_processed(processed_dir: str | Path):
    """Load a processed grounding DatasetDict + the label list.

    Returns ``(datasets, labels)``. Validates the first non-empty row of each
    split against :func:`validate_row` and raises a useful error if the
    contract is violated. ``labels.json`` is optional (falls back to scanning
    the splits) but the prepare scripts always write it.
    """
    from datasets import load_from_disk

    processed_dir = Path(processed_dir)
    datasets = load_from_disk(str(processed_dir))

    labels_path = processed_dir / "labels.json"
    if labels_path.exists():
        labels: list[str] = json.loads(labels_path.read_text())
    else:
        labels = collect_labels(datasets)

    for split_name, split in datasets.items():
        if len(split) == 0:
            continue
        errors = validate_row(split[0])
        if errors:
            raise ValueError(
                f"split {split_name!r} row 0 violates the grounding contract:\n  - "
                + "\n  - ".join(errors)
            )
    return datasets, labels


# --- chat messages + completion-only collator -----------------------------


def build_messages(prompt: str, response: str | None = None, *, system: str | None = None):
    """Build chat messages in the multimodal content format every recent HF
    processor's chat template understands.

    With *response* omitted the messages stop after the user turn — used at
    generation time (the caller adds the generation prompt). With it present
    they include the assistant turn — used to build the supervised target.
    """
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    messages.append({
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
    })
    if response is not None:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}],
        })
    return messages


def _as_image_list(image) -> list:
    """Normalize a row's image field to a flat list (single image → ``[image]``)."""
    if image is None:
        return []
    return list(image) if isinstance(image, (list, tuple)) else [image]


class VLMDataCollator:
    """Collate grounding rows into a batch with completion-only label masking.

    For each row we (1) resolve the assistant target text via :func:`build_target`
    (an explicit ``response`` if present, else ``objects`` serialized under the
    configured ``task``/``coord_bins``), (2) render the full conversation and
    the prompt-only prefix through the processor's chat template, and (3) mask
    everything that isn't the assistant response — prompt tokens, image
    placeholder tokens, and padding all become ``-100`` so the loss is computed
    **only** on the tokens the model must learn to generate. Skipping this step
    (training on the whole sequence, image tokens included) is the single most
    common VLM-SFT mistake and quietly tanks grounding quality.

    Right-padding is forced so the prompt prefix length measured per-row from a
    single-example pass still lines up with that row inside the batched encode.
    """

    def __init__(
        self,
        processor,
        *,
        task: str = "box",
        coord_bins: int = DEFAULT_COORD_BINS,
        max_length: int | None = None,
        system_prompt: str | None = None,
        empty_text: str = "No objects detected.",
        mask_prompt: bool = True,
    ) -> None:
        self.processor = processor
        self.task = task
        self.coord_bins = coord_bins
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.empty_text = empty_text
        self.mask_prompt = mask_prompt
        tok = getattr(processor, "tokenizer", processor)
        # Completion-masking math assumes the prompt sits at the front of the row.
        tok.padding_side = "right"

    def _render(self, prompt: str, response: str | None, *, add_generation_prompt: bool) -> str:
        messages = build_messages(prompt, response, system=self.system_prompt)
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        )

    def __call__(self, features: list[dict]) -> dict:
        import torch

        images_per_row = [_as_image_list(f["image"]) for f in features]
        responses = [
            build_target(
                f, task=self.task, coord_bins=self.coord_bins, empty_text=self.empty_text,
            )
            for f in features
        ]
        full_texts = [
            self._render(f["prompt"], resp, add_generation_prompt=False)
            for f, resp in zip(features, responses)
        ]

        flat_images = [img for imgs in images_per_row for img in imgs] or None
        batch = self.processor(
            text=full_texts,
            images=flat_images,
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100

        if self.mask_prompt:
            for i, f in enumerate(features):
                prompt_text = self._render(f["prompt"], None, add_generation_prompt=True)
                prompt_ids = self.processor(
                    text=[prompt_text], images=images_per_row[i] or None,
                    return_tensors="pt",
                )["input_ids"]
                labels[i, : prompt_ids.shape[1]] = -100

        batch["labels"] = labels
        return batch
