from setuptools import setup, find_packages

# LLM-driven synthetic data generation. Decoupled from any specific downstream
# task: outputs land as HuggingFace DatasetDicts that feed the encoder /
# embedding / decoder training projects. The OpenAI client is OpenAI-compatible,
# so it also drives local vLLM / Ollama servers via base_url.

install_requires = [
    "datasets>=4.8.5",       # output DatasetDicts + vendored preprocessing helper
    "accelerate>=1.13.0",
    "tiktoken==0.12.0",      # token-aware chunking
    "openai>=1.40.0",        # async client; OpenAI-compatible (vLLM, Ollama, ...)
    "pyyaml>=6.0.0",         # YAML configs + prompt templates
    "tqdm>=4.66.0",          # progress bars in the LLM stages
    "python-dotenv>=1.0.0",  # OPENAI_API_KEY from .env
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
    name="bi-encoder-dataset",
    version="0.1.0",
    description="LLM-driven pipeline that synthesises (anchor, positive) bi-encoder training datasets.",
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
