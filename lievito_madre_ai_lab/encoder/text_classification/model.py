from __future__ import annotations

from transformers import AutoModelForSequenceClassification, AutoTokenizer


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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
