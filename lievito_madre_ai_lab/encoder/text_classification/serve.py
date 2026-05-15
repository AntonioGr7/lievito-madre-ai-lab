"""
High-performance inference for BertForSequenceClassification.

Performance stack applied automatically:

  CUDA │ weights loaded in FP16
       │ SDPA attention (Flash Attention 2 on Ampere+, math kernel otherwise)
       │ torch.compile(mode="reduce-overhead") — kernel fusion + shape caching
       │ torch.autocast BF16 (Ampere+) or FP16 (older)
       │ non_blocking GPU transfers
  ─────┤
  CPU  │ weights in FP32
       │ dynamic INT8 quantisation of all Linear layers (~2–4× speedup)

  both │ torch.inference_mode() — zero autograd overhead
       │ sort-by-length batching — minimise intra-batch padding waste
       │ batch-local padding — never pad to global max_length

Usage
-----
  predictor = TextClassificationPredictor("outputs/emotion_bert/final")
  results   = predictor.predict(["I love this!", "I'm furious."])
  # [{"label": "joy", "score": 0.97, "scores": {...}}, ...]

  # single text
  result = predictor.predict_one("What a lovely day.")

  # throughput benchmark
  stats = predictor.benchmark(texts, repeats=10)

CLI
---
  python -m lievito_madre_ai_lab.encoder.text_classification.serve \\
      outputs/emotion_bert/final "I love this" "I'm so angry"

  # benchmark mode
  python -m lievito_madre_ai_lab.encoder.text_classification.serve \\
      outputs/emotion_bert/final --benchmark
"""
from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from lievito_madre_ai_lab.shared.preprocessing import load_preprocessing_meta

log = logging.getLogger(__name__)

_HAS_COMPILE = hasattr(torch, "compile")

# Sentinel for "caller did not pass a value — pull it from preprocessing.json".
_UNSET: Any = object()

_DEFAULT_MAX_LENGTH = 128


class TextClassificationPredictor:
    """
    Drop-in predictor for a Trainer-saved sequence classifier.

    Parameters
    ----------
    model_dir
        Directory containing config.json, model.safetensors, tokenizer files.
    device
        "cuda", "cpu", "mps", or None (auto-detect).
    batch_size
        Max texts per forward pass.
    use_compile
        Wrap model with torch.compile on CUDA/MPS (requires PyTorch ≥ 2).
    compile_mode
        "default" (default) — handles dynamic input shapes; right choice when
        sequence length varies between batches (this predictor uses batch-local
        padding, so shapes do vary).
        "reduce-overhead" — kernel fusion + CUDA graphs; only safe with fixed
        shapes, otherwise recompiles on every new batch shape.
        "max-autotune" — exhaustive search; much slower first call, highest
        steady-state throughput on fixed-shape workloads.
    amp_dtype
        Override autocast dtype. None = auto (BF16 on Ampere+, FP16 older).
    quantize_cpu
        Apply dynamic INT8 quantisation of Linear layers when on CPU.
    warmup_steps
        Dummy forward passes to trigger compilation before real traffic.
        Only runs when torch.compile is active.
    max_length
        Tokeniser truncation limit — should match the value used at training.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str | None = None,
        batch_size: int = 32,
        use_compile: bool = True,
        compile_mode: str = "default",
        amp_dtype: torch.dtype | None = None,
        quantize_cpu: bool = True,
        warmup_steps: int = 3,
        max_length: Any = _UNSET,
    ) -> None:
        self.batch_size = batch_size

        # Discover the tokenizer settings used at training time. Caller-provided
        # max_length still wins; the metadata only fills in the gap when the
        # caller didn't ask for anything explicit.
        meta = load_preprocessing_meta(model_dir) or {}
        if max_length is _UNSET:
            max_length = meta.get("max_length", _DEFAULT_MAX_LENGTH)
        if not meta:
            log.warning(
                "no preprocessing.json in %s — falling back to max_length=%d. "
                "If the model was trained with a different value, pass it "
                "explicitly to TextClassificationPredictor.",
                model_dir, max_length,
            )

        self.max_length = max_length
        self.device = _resolve_device(device)

        # ── tokenizer (fast Rust-backed by default) ───────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

        # ── model ─────────────────────────────────────────────────────────────
        # attn_implementation="sdpa" routes through F.scaled_dot_product_attention,
        # which dispatches to Flash Attention 2 on sm_80+ or a fused math kernel
        # elsewhere — both faster than the default unfused implementation.
        load_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            attn_implementation="sdpa",
            dtype=load_dtype,
            low_cpu_mem_usage=True,  # stream weights directly; avoids peak-RAM spike
        )
        model.eval()
        model.to(self.device)

        # Read config before any wrapping that hides attributes
        self.id2label: dict[int, str] = model.config.id2label
        self.num_labels: int = len(self.id2label)

        # ── CPU: dynamic INT8 quantisation ────────────────────────────────────
        if quantize_cpu and self.device.type == "cpu":
            from torch.ao.quantization import quantize_dynamic
            model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
            log.info("dynamic INT8 quantisation applied (CPU)")

        # ── CUDA / MPS: torch.compile ─────────────────────────────────────────
        # fullgraph=False lets the compiler handle data-dependent control flow
        # in HF models without breaking (graph breaks are silently skipped).
        compiled = False
        if use_compile and _HAS_COMPILE and self.device.type in ("cuda", "mps"):
            model = torch.compile(model, mode=compile_mode, fullgraph=False)
            compiled = True
            log.info("torch.compile(mode=%r) applied", compile_mode)

        self._model = model

        # ── autocast context ──────────────────────────────────────────────────
        self._amp_ctx: AbstractContextManager = _build_amp_ctx(self.device, amp_dtype)

        # ── warmup: amortise compilation over dummy passes ────────────────────
        if compiled and warmup_steps > 0:
            self._warmup(warmup_steps)

        log.info(
            "predictor ready │ device=%s  dtype=%s  compiled=%s  labels=%d  "
            "max_length=%d",
            self.device,
            load_dtype,
            compiled,
            self.num_labels,
            self.max_length,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, texts: list[str]) -> list[dict]:
        """
        Classify *texts*. Returns one result dict per input, in the same order.

        Output shape per item::

            {
                "label":  "joy",          # predicted class name
                "score":  0.9721,         # softmax confidence [0, 1]
                "scores": {               # all class probabilities
                    "joy": 0.9721,
                    "anger": 0.0091,
                    ...
                }
            }
        """
        if not texts:
            return []

        # Sort longest-first → shorter sequences cluster together → less padding
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        sorted_texts = [texts[i] for i in order]

        sorted_results: list[dict] = []
        for start in range(0, len(sorted_texts), self.batch_size):
            chunk = sorted_texts[start : start + self.batch_size]
            sorted_results.extend(self._forward(chunk))

        # Restore original input order
        out: list[dict | None] = [None] * len(texts)
        for rank, orig_idx in enumerate(order):
            out[orig_idx] = sorted_results[rank]
        return out  # type: ignore[return-value]

    def predict_one(self, text: str) -> dict:
        """Convenience wrapper for a single text."""
        return self.predict([text])[0]

    def benchmark(self, texts: list[str], *, repeats: int = 5) -> dict:
        """
        Measure steady-state throughput and latency over *repeats* runs.

        Returns::

            {
                "throughput_texts_per_sec": 1234.5,
                "mean_latency_ms": 81.2,
                "min_latency_ms":  79.3,
            }
        """
        import time

        self.predict(texts)  # discard first call (may include deferred compilation)

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

    @torch.inference_mode()  # faster than no_grad: disables all autograd tracking
    def _forward(self, texts: list[str]) -> list[dict]:
        enc = self.tokenizer(
            texts,
            padding=True,       # pad to this batch's max length — not global max
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # non_blocking lets the CPU keep working while DMA transfer happens
        enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}

        with self._amp_ctx:
            logits = self._model(**enc).logits  # (B, num_labels)

        # Cast to float32 before softmax — avoids precision loss with FP16 logits
        probs = torch.softmax(logits.float(), dim=-1).cpu()

        results: list[dict] = []
        for row in probs:
            scores = {self.id2label[i]: round(float(row[i]), 6) for i in range(self.num_labels)}
            best = int(row.argmax())
            results.append(
                {
                    "label": self.id2label[best],
                    "score": float(row[best]),
                    "scores": scores,
                }
            )
        return results

    def _warmup(self, n: int) -> None:
        dummy = ["warmup sentence"] * min(self.batch_size, 8)
        for _ in range(n):
            self._forward(dummy)
        log.debug("warmup complete (%d passes)", n)


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
        # BF16 on Ampere (sm_80+): wider dynamic range, no loss scaling needed
        major, _ = torch.cuda.get_device_capability(device)
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    return torch.autocast("cuda", dtype=dtype)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Text-classification inference")
    parser.add_argument("model_dir", help="Path to saved model directory")
    parser.add_argument("texts", nargs="*", help="Texts to classify")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-length", type=int, default=None,
        help=(
            "Override the tokenizer max_length. By default the predictor uses "
            "the value saved at training time (preprocessing.json)."
        ),
    )
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    overrides: dict = {}
    if args.max_length is not None:
        overrides["max_length"] = args.max_length

    predictor = TextClassificationPredictor(
        args.model_dir,
        device=args.device,
        batch_size=args.batch_size,
        use_compile=not args.no_compile,
        compile_mode=args.compile_mode,
        quantize_cpu=not args.no_quantize,
        **overrides,
    )

    texts = args.texts or ["I am so happy today!", "This makes me furious.", "I feel nothing."]

    if args.benchmark:
        stats = predictor.benchmark(texts, repeats=args.benchmark_repeats)
        print(json.dumps(stats, indent=2))
    else:
        for text, result in zip(texts, predictor.predict(texts)):
            top_scores = sorted(result["scores"].items(), key=lambda x: -x[1])
            scores_str = "  ".join(f"{k}={v:.3f}" for k, v in top_scores[:3])
            print(f"{text!r:50s}  →  {result['label']} ({result['score']:.1%})  [{scores_str}]")
