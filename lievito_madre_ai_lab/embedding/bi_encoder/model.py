from __future__ import annotations

from sentence_transformers import SentenceTransformer
from torch import nn


# Common attribute paths to the transformer encoder layer list, by family.
# Add new architectures here when they don't match BERT/ModernBERT/DistilBERT.
_LAYER_ATTR_PATHS = (
    ("encoder", "layer"),    # BERT, RoBERTa, DeBERTa, ModernBERT, XLM-R
    ("encoder", "layers"),   # Some newer architectures
    ("transformer", "layer"),  # DistilBERT
)


def truncate_encoder_layers(model: SentenceTransformer, n: int) -> int:
    """Drop transformer layers beyond the first ``n``.

    Only meaningful for models trained with `Matryoshka2dLoss` /
    `AdaptiveLayerLoss` — pruning a normally-trained model produces near-
    random embeddings. Returns the new layer count.
    """
    if n <= 0:
        raise ValueError(f"truncate_encoder_layers: n must be > 0; got {n}")

    inner = model[0].auto_model
    for path in _LAYER_ATTR_PATHS:
        try:
            obj = inner
            for attr in path[:-1]:
                obj = getattr(obj, attr)
            container = getattr(obj, path[-1])
        except AttributeError:
            continue
        if isinstance(container, nn.ModuleList) and len(container) > 0:
            if n > len(container):
                raise ValueError(
                    f"truncate_encoder_layers: requested n={n} but the model "
                    f"only has {len(container)} layers."
                )
            setattr(obj, path[-1], nn.ModuleList(list(container)[:n]))
            # Update the model's config so downstream code that inspects
            # num_hidden_layers stays consistent.
            if hasattr(inner, "config") and hasattr(inner.config, "num_hidden_layers"):
                inner.config.num_hidden_layers = n
            return n

    raise RuntimeError(
        f"truncate_encoder_layers: could not locate the layer list on "
        f"{type(inner).__name__}. Add the attribute path to _LAYER_ATTR_PATHS."
    )


def load_sentence_transformer(
    model_name: str,
    *,
    max_seq_length: int | None = None,
    trust_remote_code: bool = False,
    attn_implementation: str | None = "sdpa",
    inference_prompts: dict[str, str] | None = None,
) -> SentenceTransformer:
    """Load a SentenceTransformer backbone for fine-tuning.

    Parameters
    ----------
    max_seq_length
        Overrides the model's default truncation cap when set.
    attn_implementation
        Forwarded to the underlying transformer config — `sdpa` routes through
        PyTorch's `F.scaled_dot_product_attention` (Flash Attention 2 on
        Ampere+, fused math kernel on Turing/T4).
    inference_prompts
        Dict ``{prompt_name: prefix}`` saved on the model so callers can do
        ``model.encode(texts, prompt_name="query")`` after re-loading the
        fine-tuned checkpoint. Persists across ``save_pretrained``.
        Required for E5 / Nomic Embed / BGE-M3-style models that were
        pre-trained with query/document prefixes; without them the fine-tuned
        model drifts away from its own protocol.
    """
    model_kwargs: dict = {}
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = SentenceTransformer(
        model_name,
        trust_remote_code=trust_remote_code,
        model_kwargs=model_kwargs or None,
    )
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
    if inference_prompts:
        # SentenceTransformer.prompts is persisted by save_pretrained and read
        # back by from_pretrained, so this round-trips cleanly.
        existing = dict(getattr(model, "prompts", {}) or {})
        existing.update(inference_prompts)
        model.prompts = existing
    return model
