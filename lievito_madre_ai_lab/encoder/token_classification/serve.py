"""
High-performance inference for TokenClassification (NER / PII detection).

Same performance stack as text_classification/serve.py:

  CUDA │ FP16 weights · SDPA · torch.compile(reduce-overhead) · BF16/FP16 autocast
  CPU  │ FP32 weights (INT8 dynamic quantisation available via quantize_cpu=True,
       │ off by default — degrades accuracy on DeBERTa-V2 and small-margin heads)
  both │ torch.inference_mode · sort-by-length batching · batch-local padding
       │ sliding-window chunking for inputs longer than max_length (stride=128
       │ by default; matches the training preprocessing)

Output — entity spans extracted from BIO predictions::

    [
        {"text": "John Doe", "label": "GIVENNAME", "start": 0, "end": 8, "score": 0.97},
        {"text": "john@example.com", "label": "EMAIL", "start": 20, "end": 36, "score": 0.99},
    ]

Usage
-----
    predictor = TokenClassificationPredictor("outputs/pii_mbert/final")
    spans = predictor.predict("Send it to John Doe at john@example.com please.")

CLI
---
    python -m lievito_madre_ai_lab.encoder.token_classification.serve \\
        outputs/pii_mbert/final "Send it to John Doe at john@example.com."
"""
from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoTokenizer

from lievito_madre_ai_lab.shared.preprocessing import load_preprocessing_meta

log = logging.getLogger(__name__)

_HAS_COMPILE = hasattr(torch, "compile")

# Sentinel for "caller did not pass a value — pull it from preprocessing.json".
# We can't use ``None`` because ``stride=None`` legitimately means "disable
# chunking" and ``max_length=None`` is reserved for tokenizer defaults.
_UNSET: Any = object()

_DEFAULT_MAX_LENGTH = 512
_DEFAULT_STRIDE = 128


class TokenClassificationPredictor:
    """
    Drop-in predictor for a Trainer-saved token classifier (NER / PII).

    Parameters mirror text_classification/serve.py — see that module for
    full parameter documentation.
    """

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
        max_length: Any = _UNSET,
        stride: Any = _UNSET,
        attn_implementation: str = "eager",
    ) -> None:
        self.batch_size = batch_size

        # Discover the preprocessing settings used at training time. Caller-
        # provided max_length / stride still win; the metadata only fills in
        # the gap when the caller didn't ask for anything explicit.
        meta = load_preprocessing_meta(model_dir) or {}
        if max_length is _UNSET:
            max_length = meta.get("max_length", _DEFAULT_MAX_LENGTH)
        if stride is _UNSET:
            stride = meta.get("stride", _DEFAULT_STRIDE)
        if not meta:
            log.warning(
                "no preprocessing.json in %s — falling back to max_length=%d "
                "stride=%s. If the model was trained with different values, "
                "pass them explicitly to TokenClassificationPredictor.",
                model_dir, max_length, stride,
            )

        self.max_length = max_length
        # Sliding-window chunking for long inputs. None or <=0 disables.
        self.stride = stride if (stride is not None and stride > 0) else None
        self.device = _resolve_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

        load_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model = AutoModelForTokenClassification.from_pretrained(
            model_dir,
            attn_implementation=attn_implementation,
            dtype=load_dtype,
            low_cpu_mem_usage=True,
        )
        model.eval().to(self.device)

        self.id2label: dict[int, str] = model.config.id2label
        self.num_labels: int = len(self.id2label)

        if quantize_cpu and self.device.type == "cpu":
            from torch.ao.quantization import default_dynamic_qconfig, quantize_dynamic
            # Quantize FFN Linears only; leave attention projections, embeddings,
            # and the classification head in FP32. Two reasons:
            #   - DeBERTa-V2's disentangled attention uses Q/K/V projections that
            #     interact with relative-position embeddings; INT8 rounding on
            #     those breaks the backbone entirely (every token predicts "O").
            #   - The hidden_size → num_labels head has small class margins for
            #     fine-grained NER (39 PII classes here) that INT8 collapses.
            # FFN layers (intermediate.dense and the layer-level output.dense)
            # account for ~2/3 of the Linear FLOPs and tolerate dynamic INT8.
            ffn_qconfig = {
                name: default_dynamic_qconfig
                for name, m in model.named_modules()
                if isinstance(m, nn.Linear) and (
                    name.endswith(".intermediate.dense")
                    or (name.endswith(".output.dense") and ".attention." not in name)
                )
            }
            if ffn_qconfig:
                model = quantize_dynamic(model, ffn_qconfig, dtype=torch.qint8)
                log.info(
                    "dynamic INT8 quantisation applied to %d FFN Linears (CPU)",
                    len(ffn_qconfig),
                )
            else:
                log.info("dynamic INT8 quantisation skipped (no FFN Linears matched)")

        compiled = False
        if use_compile and _HAS_COMPILE and self.device.type in ("cuda", "mps"):
            model = torch.compile(model, mode=compile_mode, fullgraph=False)
            compiled = True
            log.info("torch.compile(mode=%r) applied", compile_mode)

        self._model = model
        self._amp_ctx: AbstractContextManager = _build_amp_ctx(self.device, amp_dtype)

        if compiled and warmup_steps > 0:
            self._warmup(warmup_steps)

        log.info(
            "predictor ready │ device=%s  dtype=%s  compiled=%s  labels=%d  "
            "max_length=%d  stride=%s",
            self.device, load_dtype, compiled, self.num_labels,
            self.max_length, self.stride if self.stride is not None else "off",
        )

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, texts: list[str]) -> list[list[dict]]:
        """
        Run NER on *texts*. Returns one list of entity spans per input text,
        in the same order.

        Each span::

            {"text": "John", "label": "GIVENNAME", "start": 0, "end": 4, "score": 0.99}
        """
        if not texts:
            return []

        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        sorted_texts = [texts[i] for i in order]

        sorted_results: list[list[dict]] = []
        for start in range(0, len(sorted_texts), self.batch_size):
            chunk = sorted_texts[start : start + self.batch_size]
            sorted_results.extend(self._forward(chunk))

        out: list[list[dict] | None] = [None] * len(texts)
        for rank, orig_idx in enumerate(order):
            out[orig_idx] = sorted_results[rank]
        return out  # type: ignore[return-value]

    def predict_one(self, text: str) -> list[dict]:
        """Convenience wrapper for a single text."""
        return self.predict([text])[0]

    def benchmark(self, texts: list[str], *, repeats: int = 5) -> dict:
        import time
        self.predict(texts)
        latencies: list[float] = []
        for _ in range(repeats):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.predict(texts)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
        mean_s = sum(latencies) / len(latencies)
        return {
            "throughput_texts_per_sec": round(len(texts) / mean_s, 1),
            "mean_latency_ms": round(mean_s * 1_000, 2),
            "min_latency_ms": round(min(latencies) * 1_000, 2),
        }

    # ── internals ─────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def _forward(self, texts: list[str]) -> list[list[dict]]:
        tok_kwargs: dict = dict(
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,   # needed for span extraction
        )
        if self.stride is not None:
            # Long inputs are split into overlapping chunks instead of being
            # silently truncated. Each chunk produces its own (input_ids,
            # attention_mask, offset_mapping); overflow_to_sample_mapping tells
            # us which source text each chunk came from so we can stitch spans
            # back together below.
            tok_kwargs["stride"] = self.stride
            tok_kwargs["return_overflowing_tokens"] = True

        enc = self.tokenizer(texts, **tok_kwargs)

        if self.stride is not None:
            sample_map = enc.pop("overflow_to_sample_mapping").tolist()
        else:
            sample_map = list(range(len(texts)))

        offset_mapping = enc.pop("offset_mapping")          # keep on CPU
        n_chunks = len(sample_map)
        # word_ids is the canonical subword→word grouping from the Fast tokenizer.
        # It must be read BEFORE we convert enc to a plain dict, since BatchEncoding
        # methods don't survive. Robust across WordPiece (mBERT) and SentencePiece
        # (mDeBERTa, XLM-R) — unlike offset contiguity, where the leading "▁" of
        # SentencePiece tokens eats the preceding space and falsely merges words.
        word_ids_per_chunk = [enc.word_ids(batch_index=i) for i in range(n_chunks)]
        enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}

        with self._amp_ctx:
            logits = self._model(**enc).logits              # (n_chunks, T, num_labels)

        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

        # Spans per chunk, in original-text char coords (HF Fast tokenizer
        # preserves source offsets across overflow chunks).
        per_text: list[list[dict]] = [[] for _ in texts]
        for chunk_idx in range(n_chunks):
            src_idx = sample_map[chunk_idx]
            spans = _tokens_to_spans(
                texts[src_idx],
                probs[chunk_idx],
                offset_mapping[chunk_idx].tolist(),
                word_ids_per_chunk[chunk_idx],
                self.id2label,
            )
            per_text[src_idx].extend(spans)

        # Merge adjacent / overlapping chunk outputs. With stride > 0 a span
        # near a chunk boundary often appears twice (once truncated in chunk N,
        # once whole in chunk N+1). _merge_overlapping_spans collapses these.
        if self.stride is not None:
            per_text = [_merge_overlapping_spans(s, texts[i]) for i, s in enumerate(per_text)]
        return per_text

    def _warmup(self, n: int) -> None:
        dummy = ["warmup sentence"] * min(self.batch_size, 4)
        for _ in range(n):
            self._forward(dummy)
        log.debug("warmup complete (%d passes)", n)


# ── span extraction ───────────────────────────────────────────────────────────

def _tokens_to_spans(
    text: str,
    probs: np.ndarray,
    offset_mapping: list[list[int]],
    word_ids: list[int | None],
    id2label: dict[int, str],
) -> list[dict]:
    """Word-aware aggregation of per-subword class probabilities into entity spans.

    Pipeline:
      1. Group subwords into words by tokenizer-provided word_ids
         (None marks special tokens, which are skipped).
      2. For each word, average per-class probs across its subwords. The word's
         label is the argmax of that average; its score is the avg prob of the
         winning class. This is HF's ``aggregation_strategy="average"``.
      3. Merge consecutive words sharing the same non-O entity type into a
         single span, ignoring the B-/I- distinction so models trained with
         first-subword-only labels (where I- tags are never emitted) still
         produce contiguous spans.
    """
    # Step 1: group subwords into words via word_ids
    words: list[dict] = []
    cur_subwords: list[np.ndarray] = []
    cur_offsets: list[tuple[int, int]] = []
    cur_word_id: int | None = None

    for (char_start, char_end), word_id, subword_probs in zip(
        offset_mapping, word_ids, probs
    ):
        if word_id is None:
            continue
        if word_id == cur_word_id:
            cur_subwords.append(subword_probs)
            cur_offsets.append((char_start, char_end))
        else:
            if cur_subwords:
                words.append(_finalize_word(cur_offsets, cur_subwords, id2label))
            cur_subwords = [subword_probs]
            cur_offsets = [(char_start, char_end)]
            cur_word_id = word_id

    if cur_subwords:
        words.append(_finalize_word(cur_offsets, cur_subwords, id2label))

    # Step 3+4: merge consecutive words with the same entity type
    spans: list[dict] = []
    active: dict | None = None
    for w in words:
        if w["entity"] == "O":
            if active is not None:
                spans.append(_finalize_span(active, text))
                active = None
            continue
        if active is None or w["entity"] != active["entity"]:
            if active is not None:
                spans.append(_finalize_span(active, text))
            active = {
                "entity": w["entity"],
                "start": w["start"],
                "end": w["end"],
                "scores": [w["score"]],
            }
        else:
            active["end"] = w["end"]
            active["scores"].append(w["score"])

    if active is not None:
        spans.append(_finalize_span(active, text))

    return spans


def _finalize_word(
    offsets: list[tuple[int, int]],
    subword_probs: list[np.ndarray],
    id2label: dict[int, str],
) -> dict:
    def _base(label: str) -> str:
        return label[2:] if label.startswith(("B-", "I-")) else label

    # Entity type: majority vote across subwords, treating B-X and I-X as one
    # class (X).  Averaging raw probability vectors has three failure modes:
    #   (a) B-X on subword-0 and I-X on subword-1..n each average to ~0.5,
    #       halving apparent confidence (e.g. "Evelyn" → score 0.499).
    #   (b) A leading space-marker subword (▁ alone, word_id shared with the
    #       following content token) votes O and can tip a short word to O.
    #   (c) A trailing O-voting punctuation token co-grouped by the tokenizer
    #       (e.g. "Falls," where "," shares word_id with "Falls") makes O win
    #       by a slim margin, truncating multi-word spans like "Great Falls".
    votes: dict[str, int] = {}
    for sp in subword_probs:
        e = _base(id2label[int(sp.argmax())])
        votes[e] = votes.get(e, 0) + 1

    max_votes = max(votes.values())
    winners = [e for e, v in votes.items() if v == max_votes]
    if len(winners) == 1:
        entity = winners[0]
    else:
        # Break ties with the first subword whose top prediction is not O.
        entity = "O"
        for sp in subword_probs:
            e = _base(id2label[int(sp.argmax())])
            if e != "O":
                entity = e
                break

    # Score: sum of B-X + I-X average probs so the full probability mass
    # assigned to this entity type is reflected in a single number.
    avg = np.mean(subword_probs, axis=0)
    score = sum(float(p) for lid, p in enumerate(avg) if _base(id2label[lid]) == entity)

    # Span: trim to the first and last subword whose top prediction matches the
    # entity type, so co-grouped leading/trailing punctuation (e.g. the "#" in
    # "#AC-…" or the ")" in "XQ)") does not bleed into the extracted text.
    start = offsets[0][0]
    end = offsets[-1][1]
    if entity != "O":
        for i, sp in enumerate(subword_probs):
            if _base(id2label[int(sp.argmax())]) == entity:
                start = offsets[i][0]
                break
        for i in range(len(subword_probs) - 1, -1, -1):
            if _base(id2label[int(subword_probs[i].argmax())]) == entity:
                end = offsets[i][1]
                break

    return {"start": start, "end": end, "entity": entity, "score": score}


def _merge_overlapping_spans(spans: list[dict], text: str) -> list[dict]:
    """Stitch chunk-level spans back into one consistent set for the source text.

    When a long document is split with stride > 0, an entity that sits in the
    overlap region typically appears twice: once truncated at the chunk-1 tail,
    once complete inside chunk 2. We merge same-label spans whose char ranges
    touch or overlap by taking the outer bounds, keeping the higher confidence,
    and re-slicing the merged ``text`` from the source string.

    Different-label overlaps are left untouched — that's a real model
    disagreement, not a chunking artefact, and downstream code can resolve it
    if it wants to.
    """
    if not spans:
        return spans
    spans = sorted(spans, key=lambda s: (s["start"], -s["end"]))
    merged: list[dict] = [dict(spans[0])]
    for s in spans[1:]:
        last = merged[-1]
        same_label = s["label"] == last["label"]
        # "<=" not "<": adjacent spans like [10,15] and [15,20] of the same
        # label are also merged. Real entities don't end at exactly the byte
        # another begins; this is almost always the chunk boundary biting.
        if same_label and s["start"] <= last["end"]:
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


def _finalize_span(active: dict, text: str) -> dict:
    # SentencePiece "▁" tokens carry the preceding space inside their offset,
    # so a raw slice like " John" leaks whitespace into the span. Trim it.
    start, end = active["start"], active["end"]
    raw = text[start:end]
    lstripped = raw.lstrip()
    start += len(raw) - len(lstripped)
    final = lstripped.rstrip()
    end = start + len(final)
    scores = active["scores"]
    return {
        "label": active["entity"],
        "text": final,
        "start": start,
        "end": end,
        "score": round(sum(scores) / len(scores), 6),
    }


# ── helpers ───────────────────────────────────────────────────────────────────

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


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Token-classification (NER/PII) inference")
    parser.add_argument("model_dir", help="Path to saved model directory")
    parser.add_argument("texts", nargs="*", help="Texts to run NER on")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-length", type=int, default=None,
        help=(
            "Override the tokenizer max_length. By default the predictor uses "
            "the value saved at training time (preprocessing.json)."
        ),
    )
    parser.add_argument(
        "--stride", type=int, default=None,
        help=(
            "Override the sliding-window overlap (in tokens) used for inputs "
            "longer than max_length. Pass -1 to disable chunking. By default "
            "the predictor uses the value saved at training time."
        ),
    )
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable dynamic INT8 quantisation of FFN layers (CPU only). "
        "Off by default — known to degrade accuracy on DeBERTa-V2.",
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help='"eager" | "sdpa" | "flash_attention_2". DeBERTa-V2 only supports "eager".',
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    overrides: dict = {}
    if args.max_length is not None:
        overrides["max_length"] = args.max_length
    if args.stride is not None:
        overrides["stride"] = None if args.stride < 0 else args.stride

    predictor = TokenClassificationPredictor(
        args.model_dir,
        device=args.device,
        batch_size=args.batch_size,
        use_compile=not args.no_compile,
        quantize_cpu=args.quantize,
        attn_implementation=args.attn_implementation,
        **overrides,
    )

    texts = args.texts or [
        "Please send the report to Maria Rossi at maria.rossi@example.com by 2024-12-01.",
        "John called from +1-800-555-0199 and left his ID: AB123456.",
    ]

    if args.benchmark:
        print(json.dumps(predictor.benchmark(texts, repeats=args.benchmark_repeats), indent=2))
    else:
        for text, spans in zip(texts, predictor.predict(texts)):
            print(f"\n{text}")
            if spans:
                for span in spans:
                    print(f"  [{span['label']:30s}] {span['text']!r:30s}  score={span['score']:.3f}")
            else:
                print("  (no entities found)")
