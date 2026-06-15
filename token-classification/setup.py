from setuptools import setup, find_packages

# torch is intentionally excluded: CUDA wheels are not on PyPI.
# Install it first with the right index, e.g.:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
# Then: pip install -e ".[dev]"

install_requires = [
    "datasets>=4.8.5",
    "accelerate>=1.13.0",
    "tiktoken==0.12.0",
    "sentencepiece==0.2.1",
    "transformers>=5.8.0",
    "evaluate>=0.4.6",
    "scikit-learn>=1.4.0",
    "seqeval>=1.2.2",        # token-classification (NER/POS) metrics
    "pyyaml>=6.0.0",
    "wandb>=0.26.1",
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
    name="token-classification",
    version="0.1.0",
    description="Encoder fine-tuning for token classification / NER (HuggingFace Trainer).",
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
