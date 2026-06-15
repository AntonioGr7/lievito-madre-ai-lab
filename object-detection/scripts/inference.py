"""Quick local inference for a Trainer-saved detector.

Runs the predictor on a few images, prints detections, and (with --draw-dir)
writes annotated copies.

Usage
-----
python scripts/inference.py outputs/cppe5_dfine/run/final img1.jpg img2.jpg --draw-dir /tmp/preds
python scripts/inference.py <model_dir> img.jpg --threshold 0.5
"""
import argparse
import json
from pathlib import Path

from object_detection import serve as _serve

print(f"[serve loaded from: {_serve.__file__}]")
ObjectDetectionPredictor = _serve.ObjectDetectionPredictor


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir")
    p.add_argument("images", nargs="+")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--draw-dir", default=None)
    args = p.parse_args()

    predictor = ObjectDetectionPredictor(
        args.model_dir, device=args.device, batch_size=args.batch_size, threshold=args.threshold,
    )
    print(f"classes={list(predictor.id2label.values())}\n")

    results = predictor.predict(args.images)
    for path, dets in zip(args.images, results):
        print(f"{path}  ({len(dets)} detections)")
        print(json.dumps(dets, indent=2))
        if args.draw_dir:
            out_dir = Path(args.draw_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            dst = out_dir / (Path(path).stem + "_pred.png")
            _serve.draw_detections(path, dets).save(dst)
            print(f"  annotated -> {dst}")
        print()


if __name__ == "__main__":
    main()
