"""Inference for a fine-tuned VLM grounding model.

``generate_grounding`` is the batched generation core (also used by the eval
callback): it renders the user turn through the model's chat template, generates
greedily, decodes only the newly produced tokens, and parses the
``label<box> ... </box>`` text back into normalized objects.

``VLMPredictor`` wraps it for serving — it loads a full-FT save or a LoRA-adapter
save through the same constructor (merging the adapter for zero-overhead
inference), reads the coordinate scheme / default prompt from
``preprocessing.json``, and returns objects in both normalized and pixel
coordinates. ``draw_objects`` renders predictions onto the image for eyeballing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from vlm_finetuning.dataset import (
    DEFAULT_COORD_BINS,
    build_messages,
    parse_objects,
)
from vlm_finetuning.shared.preprocessing import load_preprocessing_meta

log = logging.getLogger(__name__)

DEFAULT_PROMPT = "Detect every object. For each, output its label followed by its bounding box."


def _resolve_device(device: str | None):
    import torch

    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_pil(image):
    """Accept a PIL image, a path, or bytes; return a RGB PIL image."""
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        import io
        return Image.open(io.BytesIO(image)).convert("RGB")
    # datasets may hand back a dict {"bytes":..., "path":...}
    if isinstance(image, dict):
        if image.get("path"):
            return Image.open(image["path"]).convert("RGB")
        if image.get("bytes"):
            import io
            return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
    raise TypeError(f"unsupported image type: {type(image).__name__}")


def generate_texts(
    model,
    processor,
    images: list,
    prompts: list[str],
    *,
    system_prompt: str | None = None,
    max_new_tokens: int = 512,
    batch_size: int = 8,
    do_sample: bool = False,
    progress: bool = False,
) -> list[str]:
    """Generic generation core: decode the assistant continuation for each
    (image, prompt) pair and return the raw text. This is the backbone of *all*
    inference — grounding just parses the strings this returns."""
    import torch

    if len(images) != len(prompts):
        raise ValueError(f"images/prompts length mismatch: {len(images)} vs {len(prompts)}")

    tok = getattr(processor, "tokenizer", processor)
    # Decoder-only generation requires LEFT padding so every row's real tokens
    # end flush against the generated continuation.
    prev_side = tok.padding_side
    tok.padding_side = "left"

    device = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # Training leaves use_cache=False (incompatible with gradient checkpointing);
    # generation is an order of magnitude slower without the KV cache, so flip
    # it on for the duration and restore it after.
    cfg = getattr(model, "config", None)
    prev_use_cache = getattr(cfg, "use_cache", None) if cfg is not None else None
    if cfg is not None:
        cfg.use_cache = True

    indices = range(0, len(images), batch_size)
    if progress:
        try:
            from tqdm.auto import tqdm
            indices = tqdm(indices, desc="generate", unit="batch")
        except Exception:
            pass

    results: list[str] = []
    try:
        for start in indices:
            batch_imgs = [_to_pil(im) for im in images[start : start + batch_size]]
            batch_prompts = prompts[start : start + batch_size]
            texts = [
                processor.apply_chat_template(
                    build_messages(p, system=system_prompt),
                    tokenize=False, add_generation_prompt=True,
                )
                for p in batch_prompts
            ]
            inputs = processor(
                text=texts, images=batch_imgs, padding=True, return_tensors="pt",
            ).to(device)

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    pad_token_id=pad_id,
                )
            # With left padding every row shares the same prompt length.
            gen = out[:, inputs["input_ids"].shape[1]:]
            results.extend(tok.batch_decode(gen, skip_special_tokens=True))
    finally:
        tok.padding_side = prev_side
        if cfg is not None and prev_use_cache is not None:
            cfg.use_cache = prev_use_cache
    return results


def generate_grounding(
    model,
    processor,
    images: list,
    prompts: list[str],
    *,
    system_prompt: str | None = None,
    coord_bins: int = DEFAULT_COORD_BINS,
    max_new_tokens: int = 512,
    batch_size: int = 8,
    do_sample: bool = False,
    progress: bool = False,
) -> list[list[dict]]:
    """Generate, then parse the ``<box>`` text into structured objects.

    Returns one list of ``{"label", "box"|"point"}`` (normalized [0,1]) per input.
    Thin wrapper over :func:`generate_texts` — the grounding specialization.
    """
    texts = generate_texts(
        model, processor, images, prompts,
        system_prompt=system_prompt, max_new_tokens=max_new_tokens,
        batch_size=batch_size, do_sample=do_sample, progress=progress,
    )
    return [parse_objects(t, coord_bins=coord_bins) for t in texts]


def _denormalize(objects: list[dict], width: int, height: int) -> list[dict]:
    """Add pixel-space coordinates alongside the normalized ones."""
    out = []
    for o in objects:
        d = dict(o)
        if "box" in o:
            x1, y1, x2, y2 = o["box"]
            d["box_px"] = [x1 * width, y1 * height, x2 * width, y2 * height]
        if "point" in o:
            x, y = o["point"]
            d["point_px"] = [x * width, y * height]
        out.append(d)
    return out


class VLMPredictor:
    """Drop-in predictor for a Trainer-saved VLM grounding model (full-FT or LoRA)."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str | None = None,
        batch_size: int = 8,
        precision: str = "auto",
        merge_lora_on_load: bool = True,
        max_new_tokens: int | None = None,
        default_prompt: str | None = None,
        attn_implementation: str | None = "sdpa",
    ) -> None:
        from transformers import AutoProcessor

        from vlm_finetuning.model import resolve_torch_dtype

        model_dir = Path(model_dir)
        self.device = _resolve_device(device)
        self.batch_size = batch_size

        meta = load_preprocessing_meta(model_dir) or {}
        self.coord_bins = int(meta.get("coord_bins", DEFAULT_COORD_BINS))
        self.task = str(meta.get("task", "box"))
        self.system_prompt = meta.get("system_prompt")
        self.labels: list[str] = list(meta.get("labels", []) or [])
        self.default_prompt = (
            default_prompt or meta.get("default_prompt") or DEFAULT_PROMPT
        )
        self.max_new_tokens = int(max_new_tokens or meta.get("max_new_tokens", 512))
        if not meta:
            log.warning(
                "no preprocessing.json in %s — coord_bins/task/default_prompt "
                "fall back to defaults. Pass them explicitly if the model was "
                "trained with non-default settings.", model_dir,
            )

        dtype = resolve_torch_dtype(precision)
        adapter_cfg = model_dir / "adapter_config.json"

        if adapter_cfg.exists():
            import json
            from peft import PeftModel

            from vlm_finetuning.model import _load_auto_model

            base_name = json.loads(adapter_cfg.read_text())["base_model_name_or_path"]
            log.info("loading LoRA adapter from %s (base=%s)", model_dir, base_name)
            base = _load_auto_model(
                base_name, torch_dtype=dtype, attn_implementation=attn_implementation,
            )
            wrapped = PeftModel.from_pretrained(base, str(model_dir))
            model = wrapped.merge_and_unload() if merge_lora_on_load else wrapped
        else:
            from vlm_finetuning.model import _load_auto_model

            model = _load_auto_model(
                str(model_dir), torch_dtype=dtype, attn_implementation=attn_implementation,
            )

        model.eval()
        model.to(self.device)
        self._model = model
        self.processor = AutoProcessor.from_pretrained(str(model_dir))
        log.info(
            "predictor ready │ device=%s task=%s coord_bins=%d labels=%d",
            self.device, self.task, self.coord_bins, len(self.labels),
        )

    def predict(
        self,
        images: list,
        prompts: list[str] | str | None = None,
        *,
        max_new_tokens: int | None = None,
        denormalize: bool = True,
    ):
        """Predict targets for *images*.

        Returns one entry per image: a list of objects for the grounding tasks
        (``box``/``point``), or the raw generated string for ``task="text"``.
        """
        if not images:
            return []
        if prompts is None:
            prompts = [self.default_prompt] * len(images)
        elif isinstance(prompts, str):
            prompts = [prompts] * len(images)
        if len(prompts) != len(images):
            raise ValueError("len(prompts) must match len(images)")

        pil_images = [_to_pil(im) for im in images]
        budget = max_new_tokens or self.max_new_tokens

        if self.task == "text":
            return generate_texts(
                self._model, self.processor, pil_images, prompts,
                system_prompt=self.system_prompt,
                max_new_tokens=budget, batch_size=self.batch_size,
            )

        preds = generate_grounding(
            self._model, self.processor, pil_images, prompts,
            system_prompt=self.system_prompt,
            coord_bins=self.coord_bins,
            max_new_tokens=budget,
            batch_size=self.batch_size,
        )
        if denormalize:
            preds = [
                _denormalize(objs, img.width, img.height)
                for objs, img in zip(preds, pil_images)
            ]
        return preds

    def predict_one(self, image, prompt: str | None = None, **kw):
        return self.predict([image], [prompt] if prompt else None, **kw)[0]


def draw_objects(image, objects: list[dict], *, width: int = 3):
    """Render boxes/points onto a copy of *image* for visual inspection.

    Accepts objects in normalized coords (``box``/``point``) and scales them to
    the image; pixel-space keys (``box_px``/``point_px``) win if present.
    """
    from PIL import ImageDraw

    img = _to_pil(image).copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    for o in objects:
        label = o.get("label", "")
        if o.get("box_px") or "box" in o:
            x1, y1, x2, y2 = o["box_px"] if o.get("box_px") else [
                o["box"][0] * W, o["box"][1] * H, o["box"][2] * W, o["box"][3] * H
            ]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=width)
            draw.text((x1 + 2, max(0, y1 - 12)), label, fill=(255, 0, 0))
        if o.get("point_px") or "point" in o:
            x, y = o["point_px"] if o.get("point_px") else [o["point"][0] * W, o["point"][1] * H]
            r = 4 * width
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(0, 128, 255), width=width)
            draw.text((x + r + 2, y), label, fill=(0, 128, 255))
    return img


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="VLM grounding inference")
    parser.add_argument("model_dir")
    parser.add_argument("images", nargs="*", help="image file paths")
    parser.add_argument("--prompt", default=None, help="override the default prompt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--no-merge-lora", action="store_true")
    parser.add_argument("--draw-dir", default=None, help="write annotated images here")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    predictor = VLMPredictor(
        args.model_dir,
        device=args.device,
        batch_size=args.batch_size,
        merge_lora_on_load=not args.no_merge_lora,
    )

    if not args.images:
        raise SystemExit("pass one or more image paths")

    results = predictor.predict(
        args.images, args.prompt, max_new_tokens=args.max_new_tokens,
    )
    for path, objs in zip(args.images, results):
        print(f"\n{path}")
        print(json.dumps(objs, indent=2))
        if args.draw_dir:
            out_dir = Path(args.draw_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            annotated = draw_objects(path, objs)
            dst = out_dir / (Path(path).stem + "_pred.png")
            annotated.save(dst)
            print(f"  annotated -> {dst}")
