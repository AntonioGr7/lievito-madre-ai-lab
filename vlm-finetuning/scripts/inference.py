"""Quick local inference for a Trainer-saved VLM grounding model.

Runs the predictor on a couple of images and prints the parsed objects; with
``--draw-dir`` it also writes annotated copies so you can eyeball the boxes.

Usage
-----
python scripts/inference.py outputs/cppe5_qwen/run/final img1.jpg img2.jpg
python scripts/inference.py <model_dir> img.jpg --prompt "Point at every mask." --draw-dir /tmp/preds
"""
import argparse
import json
from pathlib import Path

from vlm_finetuning import serve as _serve

print(f"[serve loaded from: {_serve.__file__}]")
VLMPredictor = _serve.VLMPredictor


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir")
    p.add_argument("images", nargs="+", help="image file paths")
    p.add_argument("--prompt", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--no-merge-lora", action="store_true")
    p.add_argument("--draw-dir", default=None)
    args = p.parse_args()

    predictor = VLMPredictor(
        args.model_dir, device=args.device, batch_size=args.batch_size,
        merge_lora_on_load=not args.no_merge_lora,
    )
    print(f"task={predictor.task}  labels={predictor.labels}")
    print(f"prompt={args.prompt or predictor.default_prompt!r}\n")

    results = predictor.predict(args.images, args.prompt)
    for path, objs in zip(args.images, results):
        print(f"{path}")
        print(json.dumps(objs, indent=2))
        if args.draw_dir:
            out_dir = Path(args.draw_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            dst = out_dir / (Path(path).stem + "_pred.png")
            _serve.draw_objects(path, objs).save(dst)
            print(f"  annotated -> {dst}")
        print()


if __name__ == "__main__":
    main()
