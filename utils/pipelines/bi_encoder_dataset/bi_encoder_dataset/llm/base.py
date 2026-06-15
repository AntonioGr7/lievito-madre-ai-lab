"""Provider-agnostic LLM client contract.

The synthetic-data pipelines speak to LLMs through this thin protocol so the
generation and filtering modules stay agnostic to the backend. v1 ships with
an OpenAI-compatible adapter (also covers vLLM / Ollama via ``base_url``);
swapping in Anthropic or any other SDK later is a single new file in
``providers.py``.

The contract is async-first because synthetic-data jobs are I/O bound — a
10k-document corpus × 5 queries × 2 LLM calls (gen + judge) is ~100k
requests, and a synchronous client wastes most of the wall-clock waiting on
the network.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class LLMRequest:
    """One LLM call. Kept minimal — provider-specific knobs live on the client."""
    system: str | None
    user: str
    response_format: str | None = None  # "json_object" to force JSON mode where supported
    metadata: dict = field(default_factory=dict)  # echoed back unchanged; useful for joining


@dataclass
class LLMResponse:
    """Result of a single :class:`LLMRequest`.

    ``error`` is set when the request failed after all retries; callers
    decide whether to drop the row or surface the error. Keeping failures
    in-band (rather than raising) lets a batch of 50k calls finish even if
    a handful of rows hit content filters or transient 5xx.
    """
    text: str
    metadata: dict = field(default_factory=dict)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None


class LLMClient(Protocol):
    """The surface every provider adapter implements.

    Implementations are responsible for retries, rate-limit backoff, and
    bounding concurrency — the pipeline calls :meth:`generate_batch` with
    a flat list and expects a list back in the same order.
    """

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Run a single request. Used by tests and small ad-hoc calls."""
        ...

    async def generate_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Run a batch concurrently, preserving order.

        Order preservation matters because the pipeline joins responses back
        to source chunks by index. Implementations should bound concurrency
        with a semaphore (default ~16) — higher and you'll trip per-minute
        rate limits on most providers, lower and a 10k-row job takes hours.
        """
        ...
