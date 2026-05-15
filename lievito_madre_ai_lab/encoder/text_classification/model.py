from __future__ import annotations

from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerFast


def load_model_and_tokenizer(
    model_name: str,
    num_labels: int,
    label_names: list[str] | None = None,
    *,
    attn_implementation: str | None = "sdpa",
) -> tuple:
    id2label = {i: (label_names[i] if label_names else str(i)) for i in range(num_labels)}
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        attn_implementation=attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Refuse slow tokenizers: save_pretrained on a slow tokenizer omits
    # tokenizer.json, and reloading the saved dir later silently requires
    # `sentencepiece` to convert slow→fast — a footgun we hit in production.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise RuntimeError(
            f"Tokenizer for {model_name!r} loaded as a slow tokenizer. "
            "This usually means `sentencepiece` (or `tiktoken`) is missing in "
            "the training environment, so HF can't build the fast tokenizer. "
            "Install it (`pip install sentencepiece`) and re-run — otherwise "
            "the saved checkpoint won't contain tokenizer.json and inference "
            "will fail to load without sentencepiece installed too."
        )
    return model, tokenizer
