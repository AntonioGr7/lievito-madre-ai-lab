from __future__ import annotations

from datasets import ClassLabel, DatasetDict
from transformers import AutoTokenizer


def tokenize_for_trainer(
    raw: DatasetDict,
    model_name: str,
    text_col: str = "text",
    label_col: str = "label",
    max_length: int = 128,
) -> DatasetDict:
    """Tokenize a text-classification DatasetDict and return it ready for Trainer.

    - Runs AutoTokenizer on `text_col`
    - Renames `label_col` → `labels` (Trainer convention)
    - Drops the original text column
    - Preserves ClassLabel feature metadata so label names survive save_to_disk
    - Does NOT pad here — DataCollatorWithPadding handles per-batch dynamic padding
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _tokenize(batch: dict) -> dict:
        tokens = tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        tokens["labels"] = batch[label_col]
        return tokens

    cols_to_remove = list({text_col, label_col} - {"labels"})

    tokenized = raw.map(
        _tokenize,
        batched=True,
        remove_columns=cols_to_remove,
        desc="Tokenizing",
    )

    # map() infers `labels` as plain int64; cast it back to ClassLabel so that
    # label names (e.g. "joy", "anger") are preserved in the Arrow schema.
    original_feature = raw["train"].features.get(label_col)
    if isinstance(original_feature, ClassLabel):
        tokenized = tokenized.cast_column("labels", original_feature)

    return tokenized
