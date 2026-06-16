from setuptools import setup, find_packages

# torch is intentionally excluded: CUDA wheels are not on PyPI.
# Install it first with the right index, e.g.:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Then: pip install -e ".[dev]"
#
# This project fine-tunes vision-language models (Qwen2.5-VL, SmolVLM/Idefics3,
# LLaVA, …) with LoRA / QLoRA. It targets `transformers` 5.x (validated on
# 5.12.1; the 5.x Idefics3/Qwen-VL processors are what the collator expects),
# so it is incompatible with the GLiNER project's pinned `transformers<5.2.0`.
# Install it in its own virtualenv.

install_requires = [
    "transformers>=5.0.0",
    "datasets>=2.19.0",
    "accelerate>=0.30.0",
    "peft>=0.11.0",          # LoRA / QLoRA adapters
    "pillow>=10.0.0",        # image decoding + box/point rendering
    "pyyaml>=6.0.0",
    "wandb>=0.17.0",
    "python-dotenv>=1.0.0",
]

extras_require = {
    # 4-bit QLoRA. bitsandbytes ships CUDA kernels (Linux/Windows + NVIDIA);
    # kept optional so the project installs on CPU/macOS for dev + smoke tests.
    "quant": ["bitsandbytes>=0.43.0"],
    "dev": [
        "pytest>=8.0.0",
        "pytest-cov>=5.0.0",
        "ruff>=0.4.0",
        "black>=24.0.0",
        "mypy>=1.9.0",
    ],
}

setup(
    name="vlm-finetuning",
    version="0.1.0",
    description="LoRA/QLoRA fine-tuning of vision-language models for grounded extraction (labels + <box> coordinate tokens).",
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
