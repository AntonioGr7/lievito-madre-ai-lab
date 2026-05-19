#!/usr/bin/env python
"""Build a tiny synthetic (anchor, positive) dataset for the smoke test.

Generates ~120 paraphrase pairs across 12 topics. The data is procedural
and offline — no HF download required. The goal is "does the pipeline
run end-to-end without crashing"; learning quality is secondary (though
the loss should still decrease over 1 epoch on this).

Usage
-----
python examples/embedding_bi_encoder/prepare_smoke.py
# writes:
#   data/processed/smoke-bi-encoder/{train,validation,test}/...
#   data/processed/smoke-bi-encoder/preprocessing.json
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from datasets import Dataset, DatasetDict

from lievito_madre_ai_lab.embedding.bi_encoder.dataset import (
    save_preprocessing_meta,
    validate_dataset,
)

# Each topic has multiple paraphrase variants. We sample 2 distinct ones
# per row to build (anchor, positive). With 12 topics × C(N,2) combinations
# we comfortably hit > 100 pairs even with deduplication.
TOPICS: dict[str, list[str]] = {
    "vector_db": [
        "How do vector databases work?",
        "What is a vector database?",
        "Explain how vector search systems operate.",
        "How are similarity-search libraries built?",
        "Tell me about embedding-based retrieval systems.",
    ],
    "machine_learning": [
        "What is machine learning?",
        "Explain how ML models are trained.",
        "How does a model learn from data?",
        "Describe supervised learning.",
        "What does it mean to train a neural network?",
    ],
    "cats": [
        "A cat sat on the mat.",
        "The feline was resting on the rug.",
        "A kitten was sleeping on the carpet.",
        "There was a cat on the floor.",
        "The cat curled up on the mat.",
    ],
    "weather": [
        "It is raining heavily today.",
        "Heavy rainfall is occurring.",
        "The weather is wet and stormy.",
        "There's a downpour outside.",
        "It's pouring rain right now.",
    ],
    "cooking": [
        "How do I bake bread at home?",
        "What's the process for making sourdough?",
        "Explain how to prepare fresh bread.",
        "Walk me through baking a loaf.",
        "How is homemade bread made?",
    ],
    "music": [
        "Beethoven composed nine symphonies.",
        "Ludwig van Beethoven wrote nine major symphonic works.",
        "There are nine symphonies in Beethoven's catalog.",
        "Beethoven's symphonic output totals nine pieces.",
        "Nine is the number of symphonies Beethoven created.",
    ],
    "programming": [
        "Python is a popular programming language.",
        "Many developers use Python for software development.",
        "Python is widely adopted in software engineering.",
        "It's a programming language called Python.",
        "Python is one of the most-used coding languages.",
    ],
    "space": [
        "The Sun is a star at the center of our solar system.",
        "Our solar system is centered on a star called the Sun.",
        "The Sun is the central star of the planetary system.",
        "At the heart of the solar system sits the Sun.",
        "The solar system's central body is the Sun.",
    ],
    "physics": [
        "Energy cannot be created or destroyed.",
        "The total energy of an isolated system is conserved.",
        "Conservation of energy is a fundamental law.",
        "Energy is preserved in closed systems.",
        "Total energy stays constant in any isolated system.",
    ],
    "history": [
        "The Roman Empire fell in 476 CE.",
        "476 marked the collapse of the Western Roman Empire.",
        "The Western Roman Empire ended in the year 476.",
        "Rome's western empire dissolved in 476 CE.",
        "The fall of Rome happened in 476.",
    ],
    "geography": [
        "Mount Everest is the tallest mountain on Earth.",
        "Earth's highest peak is Mount Everest.",
        "The highest point on the planet is Everest.",
        "Everest holds the record for tallest mountain.",
        "No peak on Earth is higher than Everest.",
    ],
    "biology": [
        "DNA stores genetic information in living cells.",
        "Genetic data is encoded in DNA within cells.",
        "Living cells carry their genetic instructions in DNA.",
        "DNA is the molecule that holds hereditary information.",
        "Hereditary information lives inside DNA molecules.",
    ],
}


def build_pairs(seed: int, n_train: int, n_val: int, n_test: int) -> DatasetDict:
    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []
    for variants in TOPICS.values():
        for i, anchor in enumerate(variants):
            for j, positive in enumerate(variants):
                if i != j:
                    pairs.append((anchor, positive))

    rng.shuffle(pairs)
    total = n_train + n_val + n_test
    if total > len(pairs):
        raise ValueError(
            f"Requested {total} pairs but only {len(pairs)} unique pairs exist. "
            f"Reduce sizes or extend TOPICS."
        )

    train = pairs[:n_train]
    val = pairs[n_train : n_train + n_val]
    test = pairs[n_train + n_val : n_train + n_val + n_test]

    def _to_dataset(rows: list[tuple[str, str]]) -> Dataset:
        return Dataset.from_dict({
            "anchor":   [a for a, _ in rows],
            "positive": [p for _, p in rows],
        })

    return DatasetDict({
        "train": _to_dataset(train),
        "validation": _to_dataset(val),
        "test": _to_dataset(test),
    })


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir", default="data/processed/smoke-bi-encoder")
    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--n-val", type=int, default=24)
    p.add_argument("--n-test", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[1/3] Building synthetic pairs "
          f"(train={args.n_train}, val={args.n_val}, test={args.n_test}) …")
    datasets = build_pairs(args.seed, args.n_train, args.n_val, args.n_test)
    shape = validate_dataset(datasets)
    print(f"      shape={shape}")
    print(datasets)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[2/3] Saving to {out_dir} …")
    datasets.save_to_disk(str(out_dir))

    print("[3/3] Writing preprocessing.json …")
    save_preprocessing_meta(
        out_dir,
        source="synthetic-smoke",
        max_seq_length=64,
    )
    print(f"Done. Point a YAML's processed_dir at {out_dir} to train on it.")


if __name__ == "__main__":
    main()
