"""High-performance inference for a fine-tuned GLiNER model.

Loads either a full-FT save or a LoRA-adapter save through the same
constructor. Inference uses `batch_predict_entities` so the sorted-by-length
batching above actually pays off.
"""
from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from lievito_madre_ai_lab.encoder.gliner_entity_extraction.dataset import (
    _split_words_with_offsets,
)
from lievito_madre_ai_lab.shared.preprocessing import load_preprocessing_meta

log = logging.getLogger(__name__)

_HAS_COMPILE = hasattr(torch, "compile")

# Sentinel for "caller did not pass a value — pull it from preprocessing.json".
_UNSET: Any = object()


class GLiNERPredictor:
    """Drop-in predictor for a Trainer-saved GLiNER checkpoint (full-FT or LoRA)."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str | None = None,
        batch_size: int = 16,
        use_compile: bool = True,
        compile_mode: str = "default",
        amp_dtype: torch.dtype | None = None,
        quantize_cpu: bool = False,
        warmup_steps: int = 3,
        max_length: int = 512,
        max_words: Any = _UNSET,
        stride: Any = _UNSET,
        default_threshold: float = 0.5,
        merge_lora_on_load: bool = True,
    ) -> None:
        from gliner import GLiNER

        self.batch_size = batch_size
        self.max_length = max_length
        self.default_threshold = default_threshold
        self.device = _resolve_device(device)

        # Discover chunking settings recorded at training time. Caller args
        # override; the sidecar fills in the gap. ``stride=None`` legitimately
        # disables chunking, so we need a sentinel to distinguish "user said
        # None" from "user didn't pass anything."
        meta = load_preprocessing_meta(model_dir) or {}
        if max_words is _UNSET:
            max_words = meta.get("max_words")
        if stride is _UNSET:
            stride = meta.get("stride")
        if not meta:
            log.warning(
                "no preprocessing.json in %s — long-input chunking will fall "
                "back to model.config.max_len / no stride. If the model was "
                "trained with explicit chunking, pass max_words/stride.",
                model_dir,
            )

        model_dir = Path(model_dir)
        adapter_cfg = model_dir / "adapter_config.json"

        if adapter_cfg.exists():
            log.info("loading LoRA adapter from %s", model_dir)
            # First load the base GLiNER pointed at by the adapter directory.
            # The PEFT save layout stores `base_model_name_or_path` in
            # adapter_config.json — but GLiNER.from_pretrained also accepts a
            # directory that contains gliner_config.json. We load from
            # model_dir directly and let GLiNER restore the base, then wrap
            # the underlying encoder with PeftModel.
            from peft import PeftModel
            model = GLiNER.from_pretrained(str(model_dir))
            wrapped = PeftModel.from_pretrained(model.model, str(model_dir))
            model.model = wrapped.merge_and_unload() if merge_lora_on_load else wrapped
        else:
            model = GLiNER.from_pretrained(str(model_dir))

        model.eval()

        if hasattr(model, "model") and isinstance(model.model, nn.Module):
            model.model.to(self.device)
        else:
            model.to(self.device)

        self.train_types: list[str] = list(getattr(model.config, "train_types", []) or [])
        self.holdout_types: list[str] = list(getattr(model.config, "holdout_types", []) or [])
        self.label_aliases: dict[str, str] = dict(getattr(model.config, "label_aliases", {}) or {})
        self._reverse_aliases = {v: k for k, v in self.label_aliases.items()}

        if quantize_cpu and self.device.type == "cpu":
            from torch.ao.quantization import default_dynamic_qconfig, quantize_dynamic
            ffn_qconfig = {
                name: default_dynamic_qconfig
                for name, m in model.model.named_modules()
                if isinstance(m, nn.Linear) and (
                    name.endswith(".intermediate.dense")
                    or (name.endswith(".output.dense") and ".attention." not in name)
                )
            }
            if ffn_qconfig:
                model.model = quantize_dynamic(model.model, ffn_qconfig, dtype=torch.qint8)

        compiled = False
        if use_compile and _HAS_COMPILE and self.device.type in ("cuda", "mps"):
            if hasattr(model, "model"):
                model.model = torch.compile(model.model, mode=compile_mode, fullgraph=False)
                compiled = True

        self._model = model
        self._amp_ctx: AbstractContextManager = _build_amp_ctx(self.device, amp_dtype)

        # Resolve chunking settings now that we have the model config. With
        # stride < 0 (or None) we leave chunking off; otherwise we default
        # max_words to the model's truncation cap so chunks always fit.
        if stride is None or (isinstance(stride, int) and stride < 0):
            self.stride: int | None = None
            self.max_words: int = int(getattr(model.config, "max_len", 384))
        else:
            self.stride = int(stride)
            if max_words is None:
                max_words = int(getattr(model.config, "max_len", 384))
            self.max_words = int(max_words)
            if self.stride >= self.max_words:
                raise ValueError(
                    f"stride={self.stride} must be < max_words={self.max_words}"
                )
        log.info(
            "predictor ready │ device=%s  labels=%d  max_words=%d  stride=%s",
            self.device, len(self.train_types) if self.train_types else 0,
            self.max_words, self.stride if self.stride is not None else "off",
        )

        if compiled and warmup_steps > 0:
            self._warmup(warmup_steps)

    def predict(
        self,
        texts: list[str],
        *,
        labels: list[str] | None = None,
        threshold: float | None = None,
    ) -> list[list[dict]]:
        if not texts:
            return []

        canonical_labels = list(labels) if labels is not None else self.train_types
        if not canonical_labels:
            raise ValueError(
                "no labels available: pass labels=... or load a model with "
                "train_types stamped on its config"
            )
        prompt_labels = [self.label_aliases.get(lbl, lbl) for lbl in canonical_labels]
        thr = self.default_threshold if threshold is None else threshold

        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        sorted_texts = [texts[i] for i in order]

        sorted_results: list[list[dict]] = []
        for start in range(0, len(sorted_texts), self.batch_size):
            chunk = sorted_texts[start : start + self.batch_size]
            sorted_results.extend(self._forward(chunk, prompt_labels, thr))

        out: list[list[dict] | None] = [None] * len(texts)
        for rank, orig_idx in enumerate(order):
            out[orig_idx] = sorted_results[rank]
        return out  # type: ignore[return-value]

    def predict_one(
        self,
        text: str,
        *,
        labels: list[str] | None = None,
        threshold: float | None = None,
    ) -> list[dict]:
        return self.predict([text], labels=labels, threshold=threshold)[0]

    @torch.inference_mode()
    def _forward(self, texts: list[str], labels: list[str], threshold: float) -> list[list[dict]]:
        # If chunking is on, fan out each text into overlapping word-windows
        # — matching the training-time policy. Each chunk's char offsets are
        # shifted back to the original text before merging.
        if self.stride is not None:
            splitter = self._model.data_processor.words_splitter
            chunked_texts: list[str] = []
            # (src_idx_into_texts, char_offset_shift_to_apply_to_results)
            chunk_meta: list[tuple[int, int]] = []
            for src_idx, text in enumerate(texts):
                tokens, offsets = _split_words_with_offsets(text, splitter)
                n = len(tokens)
                if n <= self.max_words:
                    chunked_texts.append(text)
                    chunk_meta.append((src_idx, 0))
                    continue
                step = self.max_words - self.stride
                start = 0
                while True:
                    end = min(start + self.max_words, n)
                    valid = [(s, e) for (s, e) in offsets[start:end] if s >= 0 and e >= 0]
                    cs = valid[0][0] if valid else 0
                    ce = valid[-1][1] if valid else 0
                    chunked_texts.append(text[cs:ce])
                    chunk_meta.append((src_idx, cs))
                    if end >= n:
                        break
                    start += step
            forward_texts = chunked_texts
        else:
            forward_texts = texts
            chunk_meta = [(i, 0) for i in range(len(texts))]

        with self._amp_ctx:
            raw_batches = self._model.batch_predict_entities(
                forward_texts, labels, threshold=threshold,
            )

        # Shift each chunk's spans back into original-text coords and group
        # by source text. Same-label spans that overlap (one entity seen at
        # the tail of one chunk and the head of the next) get merged below.
        per_text: list[list[dict]] = [[] for _ in texts]
        for raw, (src_idx, shift) in zip(raw_batches, chunk_meta):
            for r in raw:
                per_text[src_idx].append({
                    "label": self._reverse_aliases.get(r["label"], r["label"]),
                    "text":  r["text"],
                    "start": int(r["start"]) + shift,
                    "end":   int(r["end"]) + shift,
                    "score": float(r["score"]),
                })

        if self.stride is not None:
            per_text = [_merge_overlapping_spans(spans, texts[i]) for i, spans in enumerate(per_text)]
        return per_text

    def _warmup(self, n: int) -> None:
        dummy_text = "warmup sentence"
        dummy_labels = self.train_types or ["person"]
        for _ in range(n):
            self._forward([dummy_text], dummy_labels, self.default_threshold)


def _merge_overlapping_spans(spans: list[dict], text: str) -> list[dict]:
    """Stitch chunk-level spans back into one consistent set for the source text.

    When a long input is split with stride > 0, an entity that sits in the
    overlap region typically appears twice: once at the tail of chunk N, once
    at the head of chunk N+1. We merge same-label spans whose char ranges
    touch or overlap by taking the outer bounds, keeping the higher score,
    and re-slicing ``text`` from the source string. Different-label overlaps
    are left untouched — that's a real model disagreement, not a chunking
    artefact, and downstream code can resolve it if it wants to.
    """
    if not spans:
        return spans
    spans = sorted(spans, key=lambda s: (s["start"], -s["end"]))
    merged: list[dict] = [dict(spans[0])]
    for s in spans[1:]:
        last = merged[-1]
        if s["label"] == last["label"] and s["start"] <= last["end"]:
            last["end"] = max(last["end"], s["end"])
            last["score"] = max(last["score"], s["score"])
            sliced = text[last["start"]:last["end"]]
            stripped = sliced.strip()
            shift = len(sliced) - len(sliced.lstrip())
            last["start"] += shift
            last["end"] = last["start"] + len(stripped)
            last["text"] = stripped
        else:
            merged.append(dict(s))
    return merged


def _resolve_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_amp_ctx(device: torch.device, dtype: torch.dtype | None) -> AbstractContextManager:
    if device.type != "cuda":
        return torch.autocast("cpu", enabled=False)
    if dtype is None:
        major, _ = torch.cuda.get_device_capability(device)
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    return torch.autocast("cuda", dtype=dtype)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GLiNER entity-extraction inference")
    parser.add_argument("model_dir")
    parser.add_argument("texts", nargs="*")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-merge-lora", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument(
        "--max-words", type=int, default=None,
        help=(
            "Override the word-token window used for long-input chunking. "
            "By default the predictor uses the value recorded at training "
            "time (preprocessing.json), or model.config.max_len as a fallback."
        ),
    )
    parser.add_argument(
        "--stride", type=int, default=None,
        help=(
            "Override the sliding-window overlap (in word tokens). Pass -1 "
            "to disable chunking. By default the predictor uses the value "
            "recorded at training time."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    overrides: dict = {}
    if args.max_words is not None:
        overrides["max_words"] = args.max_words
    if args.stride is not None:
        overrides["stride"] = None if args.stride < 0 else args.stride

    predictor = GLiNERPredictor(
        args.model_dir,
        device=args.device,
        batch_size=args.batch_size,
        use_compile=not args.no_compile,
        quantize_cpu=args.quantize,
        default_threshold=args.threshold,
        merge_lora_on_load=not args.no_merge_lora,
        **overrides,
    )

    texts = args.texts or [
        "Please send the report to Maria Rossi at maria.rossi@example.com by 2024-12-01.",
    ]
    for text, spans in zip(texts, predictor.predict(texts, labels=args.labels)):
        print(f"\n{text}")
        for span in spans or []:
            print(f"  [{span['label']:24s}] {span['text']!r:30s}  score={span['score']:.3f}")
