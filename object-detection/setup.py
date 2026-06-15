from setuptools import setup, find_packages

# torch + torchvision are intentionally excluded: CUDA wheels are not on PyPI.
# Install them first with the right index, e.g.:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Then: pip install -e ".[dev]"
#
# Fine-tunes canonical DETR-family detectors (D-FINE, RT-DETRv2, Deformable
# DETR, DETR) via AutoModelForObjectDetection. Needs a recent transformers
# (D-FINE landed in 4.48); incompatible with the GLiNER project's pin. Use its
# own virtualenv.

install_requires = [
    "transformers>=4.49.0",
    "datasets>=2.19.0",
    "accelerate>=0.30.0",
    "albumentations>=1.4.0",   # bbox-aware augmentation
    "torchmetrics>=1.4.0",     # COCO mAP (MeanAveragePrecision)
    "pycocotools>=2.0.7",      # mAP backend
    "timm>=1.0.0",             # some detector backbones load via timm
    "pillow>=10.0.0",
    "pyyaml>=6.0.0",
    "wandb>=0.17.0",
    "python-dotenv>=1.0.0",
]

extras_require = {
    "dev": [
        "pytest>=8.0.0",
        "pytest-cov>=5.0.0",
        "ruff>=0.4.0",
        "black>=24.0.0",
        "mypy>=1.9.0",
    ],
}

setup(
    name="object-detection",
    version="0.1.0",
    description="SOTA fine-tuning of canonical (DETR-family) object detectors with COCO mAP eval, discriminative LR, and weight EMA.",
    author="antonio.grimaldi",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
