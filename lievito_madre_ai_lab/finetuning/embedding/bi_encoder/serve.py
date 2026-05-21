"""
High-performance inference for SentenceTransformer bi-encoders.

Performance stack applied automatically:

  CUDA │ weights loaded in FP16
       │ SDPA attention (Flash Attention 2 on Ampere+, math kernel otherwise)
       │ torch.compile(mode="default") — kernel fusion + shape caching
       │ torch.autocast BF16 (Ampere+) or FP16 (older)
       │ non_blocking GPU transfers
  ─────┤
  CPU  │ weights in FP32
       │ dynamic INT8 quantisation of all Linear layers (~2–4× speedup)

  both │ torch.inference_mode() — zero autograd overhead

Two features specific to the lab's training pipeline:

- **Matryoshka truncation**: pass ``truncate_dim=N`` to ``encode`` / the CLI
  to slice the first N components. Trade quality for speed/storage at zero
  retraining cost — the dim must be one the model was trained on (see
  ``bi_encoder.matryoshka.dims`` in the training YAML).
- **Instruction prompts**: instruction-tuned backbones (E5, Nomic, BGE-M3,
  …) expect prefixes like ``search_query: `` / ``search_document: ``. Use
  ``prompt_name="query"`` (recommended — reads from the saved model) or
  ``prompt="..."`` to inline a one-off.

Usage
-----
  predictor = BiEncoderPredictor("outputs/run/final")
  emb = predictor.encode(["semantic search"], prompt_name="query")

  # Matryoshka: trade dim for speed/storage at inference
  emb_64 = predictor.encode(texts, truncate_dim=64)

  # Query-vs-corpus retrieval with asymmetric prompts
  hits = predictor.search(
      queries=["how do vector DBs work?"],
      corpus=["FAISS is a library.", "PyTorch is a tensor framework."],
      top_k=5,
      query_prompt_name="query",
      corpus_prompt_name="document",
  )

CLI
---
  python -m lievito_madre_ai_lab.finetuning.embedding.bi_encoder.serve \\
      outputs/run/final "semantic search" --prompt-name query

  # Matryoshka truncation
  python -m lievito_madre_ai_lab.finetuning.embedding.bi_encoder.serve \\
      outputs/run/final "text" --truncate-dim 64
"""
from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from lievito_madre_ai_lab.finetuning.embedding.bi_encoder.model import truncate_encoder_layers
from lievito_madre_ai_lab.shared.preprocessing import load_preprocessing_meta

log = logging.getLogger(__name__)

_HAS_COMPILE = hasattr(torch, "compile")

# Sentinel for "caller didn't pass a value — pull it from preprocessing.json".
_UNSET: Any = object()


class BiEncoderPredictor:
    """
    Drop-in predictor for a Trainer-saved SentenceTransformer bi-encoder.

    Parameters
    ----------
    model_dir
        Directory containing the saved SentenceTransformer model.
    device
        "cuda", "cpu", "mps", or None (auto-detect).
    batch_size
        Max texts per forward pass.
    use_compile
        Wrap the underlying transformer with torch.compile on CUDA/MPS.
    compile_mode
        "default" handles dynamic input shapes (sort-by-length batching).
    amp_dtype
        Override autocast dtype. None = auto (BF16 on Ampere+, FP16 older).
    quantize_cpu
        Apply dynamic INT8 quantisation of Linear layers when on CPU.
    warmup_steps
        Dummy forward passes to trigger compilation before real traffic.
    max_seq_length
        Tokeniser truncation cap. Defaults to ``preprocessing.json`` value
        or the model's own.
    normalize_embeddings
        L2-normalise outputs so dot product == cosine similarity.
    default_truncate_dim
        If set, all ``encode`` calls truncate to this dim by default. The
        model must have been trained with Matryoshka over this dim.
    truncate_layers
        Drop transformer layers beyond the first N at load time. Only safe
        on Matryoshka2d / AdaptiveLayer-trained models; on a normal model
        this produces near-random embeddings. Applied **before**
        torch.compile so the compiled graph reflects the smaller model.
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
        max_seq_length: Any = _UNSET,
        normalize_embeddings: bool = True,
        default_truncate_dim: int | None = None,
        truncate_layers: int | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.default_truncate_dim = default_truncate_dim

        meta = load_preprocessing_meta(model_dir) or {}
        if max_seq_length is _UNSET:
            max_seq_length = meta.get("max_seq_length")
        if default_truncate_dim is None:
            default_truncate_dim = meta.get("default_truncate_dim")
            self.default_truncate_dim = default_truncate_dim

        self.device = _resolve_device(device)

        load_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model = SentenceTransformer(
            str(model_dir),
            device=str(self.device),
            model_kwargs={"dtype": load_dtype, "attn_implementation": "sdpa"},
        )
        if max_seq_length is not None:
            model.max_seq_length = int(max_seq_length)
        if truncate_layers is not None:
            n_kept = truncate_encoder_layers(model, truncate_layers)
            log.info("encoder truncated to %d layers", n_kept)
        model.eval()

        # CPU: dynamic INT8 quantisation of Linear layers (~2–4× speedup).
        if quantize_cpu and self.device.type == "cpu":
            from torch.ao.quantization import quantize_dynamic
            quantized = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
            model[0].auto_model = quantized[0].auto_model
            log.info("dynamic INT8 quantisation applied (CPU)")

        compiled = False
        if use_compile and _HAS_COMPILE and self.device.type in ("cuda", "mps"):
            model[0].auto_model = torch.compile(
                model[0].auto_model, mode=compile_mode, fullgraph=False
            )
            compiled = True
            log.info("torch.compile(mode=%r) applied", compile_mode)

        self._model = model
        self._amp_ctx: AbstractContextManager = _build_amp_ctx(self.device, amp_dtype)

        if compiled and warmup_steps > 0:
            self._warmup(warmup_steps)

        # `prompts` is a dict[name → prefix] that survives save/load on the
        # SentenceTransformer object. Exposed for callers that want to list
        # available prompt names.
        self.prompts: dict[str, str] = dict(getattr(self._model, "prompts", {}) or {})

        log.info(
            "bi-encoder predictor ready │ device=%s  dtype=%s  compiled=%s  "
            "max_seq_length=%d  normalize=%s  truncate_dim=%s  prompts=%s",
            self.device,
            load_dtype,
            compiled,
            self._model.max_seq_length,
            self.normalize_embeddings,
            self.default_truncate_dim,
            list(self.prompts) or "—",
        )

    # ── public API ────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def encode(
        self,
        texts: list[str],
        *,
        prompt: str | None = None,
        prompt_name: str | None = None,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        """Encode `texts` and return an (N, D) float32 tensor on CPU.

        `prompt_name` looks up the prefix on the model (saved at training
        time). `prompt` inlines a raw prefix. `truncate_dim` slices the
        first N components — only meaningful for Matryoshka-trained models.
        """
        if not texts:
            dim = self._effective_dim(truncate_dim)
            return torch.empty((0, dim))

        tdim = self.default_truncate_dim if truncate_dim is None else truncate_dim

        with self._amp_ctx:
            emb = self._model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_tensor=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
                prompt=prompt,
                prompt_name=prompt_name,
            )
        emb = emb.detach().to(torch.float32).cpu()
        if tdim is not None:
            emb = emb[:, :tdim]
            if self.normalize_embeddings:
                # Truncation breaks the L2 norm — renormalise so dot product
                # still equals cosine similarity at the truncated dim.
                emb = nn.functional.normalize(emb, p=2, dim=-1)
        return emb

    @torch.inference_mode()
    def similarity(
        self,
        a: list[str],
        b: list[str],
        *,
        a_prompt_name: str | None = None,
        b_prompt_name: str | None = None,
        truncate_dim: int | None = None,
    ) -> torch.Tensor:
        """Return the (len(a), len(b)) cosine-similarity matrix.

        ``a_prompt_name`` / ``b_prompt_name`` let asymmetric encoders (E5,
        Nomic, BGE-M3) apply ``query`` and ``document`` prompts on the right
        sides.
        """
        emb_a = self.encode(a, prompt_name=a_prompt_name, truncate_dim=truncate_dim)
        emb_b = self.encode(b, prompt_name=b_prompt_name, truncate_dim=truncate_dim)
        if self.normalize_embeddings:
            return emb_a @ emb_b.T
        return nn.functional.cosine_similarity(
            emb_a.unsqueeze(1), emb_b.unsqueeze(0), dim=-1
        )

    @torch.inference_mode()
    def search(
        self,
        queries: list[str],
        corpus: list[str],
        *,
        top_k: int = 10,
        query_prompt_name: str | None = None,
        corpus_prompt_name: str | None = None,
        truncate_dim: int | None = None,
    ) -> list[list[dict]]:
        """Rank `corpus` against each `queries` entry. Returns top-k hits per query."""
        if not queries or not corpus:
            return [[] for _ in queries]

        sims = self.similarity(
            queries, corpus,
            a_prompt_name=query_prompt_name,
            b_prompt_name=corpus_prompt_name,
            truncate_dim=truncate_dim,
        )
        k = min(top_k, len(corpus))
        scores, indices = sims.topk(k, dim=-1)
        return [
            [
                {"corpus_id": int(indices[i, j]), "score": float(scores[i, j])}
                for j in range(k)
            ]
            for i in range(len(queries))
        ]

    def benchmark(
        self,
        texts: list[str],
        *,
        repeats: int = 5,
        prompt_name: str | None = None,
        truncate_dim: int | None = None,
    ) -> dict:
        """Measure steady-state throughput and latency over *repeats* runs."""
        import time

        self.encode(texts, prompt_name=prompt_name, truncate_dim=truncate_dim)

        latencies: list[float] = []
        for _ in range(repeats):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.encode(texts, prompt_name=prompt_name, truncate_dim=truncate_dim)
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

    def _effective_dim(self, truncate_dim: int | None) -> int:
        native = self._model.get_sentence_embedding_dimension()
        tdim = self.default_truncate_dim if truncate_dim is None else truncate_dim
        return tdim if tdim is not None else native

    def _warmup(self, n: int) -> None:
        dummy = ["warmup sentence"] * min(self.batch_size, 8)
        for _ in range(n):
            self.encode(dummy)
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
        major, _ = torch.cuda.get_device_capability(device)
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    return torch.autocast("cuda", dtype=dtype)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Bi-encoder inference")
    parser.add_argument("model_dir", help="Path to saved SentenceTransformer directory")
    parser.add_argument("texts", nargs="*", help="Texts to encode")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument(
        "--truncate-dim", type=int, default=None,
        help="Matryoshka truncation: slice the first N components (renormalised).",
    )
    parser.add_argument(
        "--truncate-layers", type=int, default=None,
        help="Drop encoder layers beyond the first N. Matryoshka2d/Adaptive only.",
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Raw prompt prefix (e.g. 'search_query: '). Overrides --prompt-name.",
    )
    parser.add_argument(
        "--prompt-name", default=None,
        help="Named prompt saved on the model (e.g. 'query' or 'document').",
    )
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument(
        "--compile-mode", default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    parser.add_argument(
        "--similarity", action="store_true",
        help="Print the pairwise similarity matrix instead of raw embeddings.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    overrides: dict = {}
    if args.max_seq_length is not None:
        overrides["max_seq_length"] = args.max_seq_length

    predictor = BiEncoderPredictor(
        args.model_dir,
        device=args.device,
        batch_size=args.batch_size,
        use_compile=not args.no_compile,
        compile_mode=args.compile_mode,
        quantize_cpu=not args.no_quantize,
        normalize_embeddings=not args.no_normalize,
        truncate_layers=args.truncate_layers,
        **overrides,
    )

    texts = args.texts or [
        "A man is eating food.",
        "Someone is having a meal.",
        "The weather is sunny today.",
    ]

    if args.benchmark:
        stats = predictor.benchmark(
            texts,
            repeats=args.benchmark_repeats,
            prompt_name=args.prompt_name,
            truncate_dim=args.truncate_dim,
        )
        print(json.dumps(stats, indent=2))
    elif args.similarity:
        sims = predictor.similarity(
            texts, texts,
            a_prompt_name=args.prompt_name,
            b_prompt_name=args.prompt_name,
            truncate_dim=args.truncate_dim,
        )
        print(json.dumps(sims.tolist(), indent=2))
    else:
        emb = predictor.encode(
            texts,
            prompt=args.prompt,
            prompt_name=args.prompt_name,
            truncate_dim=args.truncate_dim,
        )
        for text, vec in zip(texts, emb):
            print(f"{text!r:50s}  →  dim={vec.shape[0]}  norm={vec.norm().item():.4f}")
