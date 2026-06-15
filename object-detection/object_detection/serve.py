"""Inference for a fine-tuned object detector.

``ObjectDetectionPredictor`` loads a Trainer-saved checkpoint (model +
image processor), runs batched inference, decodes boxes to original-image pixel
coordinates via the processor's ``post_process_object_detection``, and returns
clean ``{label, score, box}`` dicts. ``draw_detections`` renders them for
eyeballing. Defaults (score threshold, labels) are read from
``preprocessing.json``.
"""
from __future__ import annotations

import logging
from pathlib import Path

from object_detection.shared.preprocessing import load_preprocessing_meta

log = logging.getLogger(__name__)


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
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        import io
        return Image.open(io.BytesIO(image)).convert("RGB")
    raise TypeError(f"unsupported image type: {type(image).__name__}")


class ObjectDetectionPredictor:
    """Drop-in predictor for a Trainer-saved detector (D-FINE / RT-DETR / DETR / …)."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str | None = None,
        batch_size: int = 8,
        threshold: float | None = None,
    ) -> None:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        model_dir = Path(model_dir)
        self.device = _resolve_device(device)
        self.batch_size = batch_size

        meta = load_preprocessing_meta(model_dir) or {}
        self.default_threshold = float(threshold if threshold is not None else meta.get("threshold", 0.3))
        if not meta:
            log.warning("no preprocessing.json in %s — default threshold falls back to 0.3", model_dir)

        self.model = AutoModelForObjectDetection.from_pretrained(str(model_dir))
        self.model.eval().to(self.device)
        self.image_processor = AutoImageProcessor.from_pretrained(str(model_dir))
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        log.info("predictor ready │ device=%s classes=%d", self.device, len(self.id2label))

    def predict(
        self,
        images: list,
        *,
        threshold: float | None = None,
    ) -> list[list[dict]]:
        """Detect objects in each image. Returns one list of
        ``{"label", "score", "box": [x1,y1,x2,y2]}`` (original-image pixels) per input."""
        import torch

        if not images:
            return []
        thr = self.default_threshold if threshold is None else threshold

        out: list[list[dict]] = []
        for start in range(0, len(images), self.batch_size):
            batch = [_to_pil(im) for im in images[start : start + self.batch_size]]
            target_sizes = torch.tensor([[im.height, im.width] for im in batch])
            enc = self.image_processor(images=batch, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                outputs = self.model(**enc)
            results = self.image_processor.post_process_object_detection(
                outputs, threshold=thr, target_sizes=target_sizes,
            )
            for r in results:
                spans = []
                for score, label, box in zip(r["scores"], r["labels"], r["boxes"]):
                    spans.append({
                        "label": self.id2label.get(int(label), str(int(label))),
                        "score": float(score),
                        "box": [float(v) for v in box.tolist()],
                    })
                out.append(spans)
        return out

    def predict_one(self, image, *, threshold: float | None = None) -> list[dict]:
        return self.predict([image], threshold=threshold)[0]


def draw_detections(image, detections: list[dict], *, width: int = 3):
    """Render detection boxes (xyxy pixels) onto a copy of *image*."""
    from PIL import ImageDraw

    img = _to_pil(image).copy()
    draw = ImageDraw.Draw(img)
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=width)
        tag = f"{d['label']} {d.get('score', 0):.2f}"
        draw.text((x1 + 2, max(0, y1 - 12)), tag, fill=(255, 0, 0))
    return img


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Object-detection inference")
    parser.add_argument("model_dir")
    parser.add_argument("images", nargs="+", help="image file paths")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Score cutoff. Defaults to the value in preprocessing.json (or 0.3).")
    parser.add_argument("--draw-dir", default=None, help="write annotated images here")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    predictor = ObjectDetectionPredictor(
        args.model_dir, device=args.device, batch_size=args.batch_size, threshold=args.threshold,
    )
    results = predictor.predict(args.images)
    for path, dets in zip(args.images, results):
        print(f"\n{path}")
        print(json.dumps(dets, indent=2))
        if args.draw_dir:
            out_dir = Path(args.draw_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            dst = out_dir / (Path(path).stem + "_pred.png")
            draw_detections(path, dets).save(dst)
            print(f"  annotated -> {dst}")
