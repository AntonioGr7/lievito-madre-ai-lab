"""Load a vision-language model + processor and configure it for fine-tuning.

Single entry point: :func:`load_vlm`. Responsibilities:

  1. Load the processor (``AutoProcessor``) and the model
     (``AutoModelForImageTextToText`` with a ``Vision2Seq`` fallback), in the
     right compute dtype and attention implementation.
  2. Optionally quantize the base weights to 4-bit (QLoRA) via bitsandbytes.
  3. Optionally freeze the vision tower so only the language model adapts —
     the cheaper, more stable default for grounded-text tasks.
  4. Optionally register ``<box>`` / ``</box>`` as special tokens and resize the
     embedding table (and route them into ``modules_to_save`` so LoRA actually
     trains the new rows).
  5. Wrap the language model with a PEFT LoRA adapter, targeting the LM's
     attention + MLP projections only (never the frozen vision encoder).

Backbone-agnostic by construction: every model-family-specific detail (vision
module name, projection layer names) is *discovered* from the module tree
rather than hard-coded, so Qwen2.5-VL, SmolVLM/Idefics3 and LLaVA-family models
all load through the same path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoraSpec:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Any = "auto"          # "auto" | list[str]
    modules_to_save: list[str] = field(default_factory=list)


@dataclass
class QuantSpec:
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"      # "nf4" | "fp4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


# Language-model projection submodule names shared by the Llama/Qwen-style
# decoders inside every mainstream VLM (Qwen2.5-VL, SmolVLM/Idefics3, LLaVA).
_LM_LINEAR_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "gate_proj", "up_proj", "down_proj",      # MLP
)

# Name fragments that mark the vision encoder / multimodal connector. LoRA
# auto-targeting and the vision freeze both key off these so the visual path
# stays frozen regardless of the family's exact attribute names.
_VISION_NAME_FRAGMENTS = ("visual", "vision_model", "vision_tower", "image_encoder")


def resolve_torch_dtype(precision: str):
    """Map the YAML ``precision`` string → a torch dtype for ``from_pretrained``.

    ``auto`` follows the hardware: bf16 on Ampere+ (sm ≥ 80), fp16 on older
    CUDA, fp32 on CPU. The trainer resolves the *same* string into HF's
    fp16/bf16 mixed-precision flags, so the load dtype and the autocast dtype
    always agree.
    """
    import torch

    if precision == "fp32":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    # auto
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def _is_vision_module(name: str) -> bool:
    return any(frag in name for frag in _VISION_NAME_FRAGMENTS)


def _resolve_lora_targets(model, requested) -> list[str]:
    """Resolve ``lora.target_modules``: ``'auto'`` → explicit LM linear names.

    Returns the *full* module names (not bare suffixes) of every ``nn.Linear``
    in the language model whose name ends in a known projection suffix and is
    **not** inside the vision encoder. Returning full names (rather than
    suffixes peft would match everywhere) is what keeps the vision tower out of
    the adapter set — several VLMs reuse ``q_proj``/``v_proj`` names inside the
    visual blocks too.
    """
    if requested != "auto":
        return list(requested)

    import torch.nn as nn

    names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and name.split(".")[-1] in _LM_LINEAR_SUFFIXES
        and not _is_vision_module(name)
    ]
    if not names:
        raise ValueError(
            "LoRA target auto-resolution found no language-model projection "
            f"layers (looked for {_LM_LINEAR_SUFFIXES} outside the vision "
            "tower). Pass an explicit lora.target_modules list for this backbone."
        )
    return names


def _freeze_vision_tower(model) -> tuple[int, int]:
    """Freeze every parameter inside the vision encoder / connector.

    Returns ``(num_tensors, num_params)`` frozen. Matching by name fragment
    works whether the model is plain, quantized, or already PEFT-wrapped.
    """
    tensors = params = 0
    for name, p in model.named_parameters():
        if _is_vision_module(name) and p.requires_grad:
            p.requires_grad_(False)
            tensors += 1
            params += p.numel()
    return tensors, params


def _add_coord_special_tokens(model, processor) -> list[str]:
    """Register ``<box>``/``</box>`` as special tokens and resize embeddings.

    Returns the ``modules_to_save`` names LoRA must additionally train so the
    freshly-initialised embedding rows for the new tokens actually learn — with
    a pure LoRA adapter the base embedding/`lm_head` stay frozen, and the new
    tokens would never move. No-op-safe if the tokens already exist.
    """
    from vlm_finetuning.dataset import BOX_CLOSE, BOX_OPEN

    tok = getattr(processor, "tokenizer", processor)
    added = tok.add_special_tokens(
        {"additional_special_tokens": [BOX_OPEN, BOX_CLOSE]}
    )
    if added:
        model.resize_token_embeddings(len(tok))
    return ["embed_tokens", "lm_head"]


def _load_auto_model(model_name: str, **kwargs):
    """``AutoModelForImageTextToText`` with a ``Vision2Seq`` fallback for older
    transformers, so the loader works across the version range in setup.py."""
    try:
        from transformers import AutoModelForImageTextToText as _Auto
    except ImportError:  # transformers < 4.45
        from transformers import AutoModelForVision2Seq as _Auto
    return _Auto.from_pretrained(model_name, **kwargs)


def load_vlm(
    model_name: str,
    *,
    precision: str = "auto",
    attn_implementation: str | None = "sdpa",
    lora_cfg: LoraSpec | None = None,
    quant_cfg: QuantSpec | None = None,
    freeze_vision_tower: bool = True,
    gradient_checkpointing: bool = False,
    add_coord_special_tokens: bool = False,
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = None,
    trust_remote_code: bool = False,
):
    """Load a VLM + processor configured for (Q)LoRA or full fine-tuning.

    Passing only ``model_name`` loads a full-FT-ready model with the vision
    tower frozen. Returns ``(model, processor)``.
    """
    import torch
    from transformers import AutoProcessor

    lora_cfg = lora_cfg or LoraSpec(enabled=False)
    quant_cfg = quant_cfg or QuantSpec()

    # Processor. min/max pixels bound Qwen-style dynamic-resolution image-token
    # counts — the dominant lever on VLM memory; harmless kwargs on processors
    # that ignore them.
    proc_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if image_min_pixels is not None:
        proc_kwargs["min_pixels"] = image_min_pixels
    if image_max_pixels is not None:
        proc_kwargs["max_pixels"] = image_max_pixels
    processor = AutoProcessor.from_pretrained(model_name, **proc_kwargs)

    dtype = resolve_torch_dtype(precision)
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "attn_implementation": attn_implementation,
        "trust_remote_code": trust_remote_code,
    }

    if quant_cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=getattr(torch, quant_cfg.bnb_4bit_compute_dtype),
        )
        # 4-bit kernels need the weights resident on the GPU at load time.
        if torch.cuda.is_available():
            load_kwargs["device_map"] = {"": 0}

    model = _load_auto_model(model_name, **load_kwargs)

    if add_coord_special_tokens:
        extra_to_save = _add_coord_special_tokens(model, processor)
        # Merge (preserving caller order) so coord tokens train even under LoRA.
        lora_cfg.modules_to_save = list(
            dict.fromkeys(list(lora_cfg.modules_to_save) + extra_to_save)
        )

    if quant_cfg.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )

    if freeze_vision_tower:
        tensors, params = _freeze_vision_tower(model)
        if tensors:
            print(
                f"      [load_vlm] froze vision tower "
                f"({tensors} tensors, {params:,} params)"
            )

    if lora_cfg.enabled:
        from peft import LoraConfig, get_peft_model

        targets = _resolve_lora_targets(model, lora_cfg.target_modules)
        lora = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=targets,
            modules_to_save=lora_cfg.modules_to_save or None,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    # Gradient checkpointing + LoRA: autograd must still reach the adapters
    # through the frozen embedding table. prepare_model_for_kbit_training
    # already wires this for the 4-bit path; do it explicitly otherwise.
    if gradient_checkpointing and not quant_cfg.load_in_4bit:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if gradient_checkpointing and hasattr(model, "config"):
        model.config.use_cache = False

    return model, processor
