from setuptools import setup, find_packages

# torch is intentionally excluded: CUDA wheels are not on PyPI.
# Install it first with the right index, e.g.:
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
# Then: pip install -e ".[encoder]"
# Shared minimal deps. transformers is intentionally NOT here: each extras
# group pins its own transformers version so groups with incompatible
# requirements (e.g. gliner needs transformers<5.2.0) stay isolated.
_base = [
    "datasets>=4.8.5",
    "accelerate>=1.13.0",
    "tiktoken==0.12.0",
    "sentencepiece==0.2.1"
]

extras_require = {
    # ------------------------------------------------------------------ #
    # Encoder models — BERT, RoBERTa, DeBERTa, DistilBERT, …             #
    # Tasks: classification, NER, QA, sentence similarity                 #
    # ------------------------------------------------------------------ #
    "encoder": _base + [
        "transformers>=5.8.0",
        "evaluate>=0.4.6",
        "scikit-learn>=1.4.0",   # metrics (F1, accuracy, confusion matrix)
        "seqeval>=1.2.2",        # token-classification (NER/POS) metrics
        "pyyaml>=6.0.0",         # YAML config files
        "wandb>=0.26.1",         # experiment tracking
        "python-dotenv>=1.0.0",  # load API keys from .env
    ],

    # ------------------------------------------------------------------ #
    # Decoder models — GPT-2, LLaMA, Mistral, Falcon, …                  #
    # Tasks: causal LM, instruction tuning, chat                         #
    # ------------------------------------------------------------------ #
    "decoder": _base + [
        "transformers>=5.8.0",
        "peft>=0.10.0",          # LoRA / adapter fine-tuning
        "trl>=0.8.6",            # SFT / RLHF / DPO trainers
        "bitsandbytes>=0.43.0",  # 4-bit / 8-bit quantization
        "sentencepiece>=0.2.0",  # tokenizer for LLaMA-family
    ],


    # ------------------------------------------------------------------ #
    # Vision & vision-language — ViT, CLIP, BEiT, Swin, …               #
    # Tasks: image classification, contrastive learning, VQA             #
    # ------------------------------------------------------------------ #
    "vision": _base + [
        "transformers>=5.8.0",
        "torchvision>=0.17.0",
        "timm>=0.9.16",
        "Pillow>=10.0.0",
        "evaluate>=0.4.0",
        "scikit-learn>=1.4.0",
    ],

    # ------------------------------------------------------------------ #
    # GLiNER — open-vocabulary entity extraction (urchade/GLiNER)        #
    # Tasks: open-vocab NER, zero-shot entity extraction                 #
    # NOTE: gliner 0.2.x requires transformers<5.2.0, so this group is   #
    # mutually exclusive with the encoder/decoder/vision groups. Install #
    # in its own virtualenv.                                              #
    # ------------------------------------------------------------------ #
    "gliner": _base + [
        "transformers>=4.51.3,<5.2.0",
        "gliner>=0.2.0,<0.3.0",
        "peft>=0.10.0",          # LoRA adapters for parameter-efficient FT
        "evaluate>=0.4.6",
        "seqeval>=1.2.2",
        "pyyaml>=6.0.0",
        "wandb>=0.26.1",
        "python-dotenv>=1.0.0",
    ],

    # ------------------------------------------------------------------ #
    # Development & testing                                               #
    # ------------------------------------------------------------------ #
    "dev": [
        "pytest>=8.0.0",
        "pytest-cov>=5.0.0",
        "ruff>=0.4.0",
        "black>=24.0.0",
        "mypy>=1.9.0",
    ],
}

# Convenience target: install everything compatible. The "gliner" group is
# excluded because its transformers pin conflicts with the encoder/decoder/
# vision groups — install "gliner" in its own virtualenv.
extras_require["all"] = list(
    {dep for name, deps in extras_require.items() if name != "gliner" for dep in deps}
)

setup(
    name="lievito-madre-ai-lab",
    version="0.1.0",
    description="A living lab for fine-tuning encoder, decoder, llms and slms",
    author="antonio.grimaldi",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[],        # no mandatory deps — pick an extra
    extras_require=extras_require,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
