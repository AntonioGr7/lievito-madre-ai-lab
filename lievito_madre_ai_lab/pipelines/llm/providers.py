"""OpenAI-compatible async client for synthetic-data pipelines.

OpenAI's SDK is OpenAI-compatible for vLLM, Ollama, Together, Groq, Fireworks,
and any other server that exposes the ``/v1/chat/completions`` shape — point
``base_url`` at a local vLLM and the same client serves a 70B model on a
single H100. So shipping one adapter covers cloud + local on day one.

Concurrency is bounded by an asyncio.Semaphore. The SDK already handles
retries and rate-limit backoff via ``max_retries``; we only add a per-request
exception-to-error conversion so a partial batch failure doesn't take the
whole job down — at 50k calls some are *going* to fail (content filter,
transient 5xx, finish_reason=length on a too-tight max_tokens), and we'd
rather drop those rows than crash after 90% of the work is done.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from lievito_madre_ai_lab.pipelines.llm.base import LLMRequest, LLMResponse


@dataclass
class OpenAIClient:
    """Async OpenAI-compatible client.

    Pass ``base_url`` to redirect at a local vLLM / Ollama instance — the
    rest of the interface is unchanged. Default concurrency of 16 is a
    safe starting point for the OpenAI cloud (tier-1 rate limits) and
    well below what vLLM saturates with continuous batching, so callers
    on local backends should bump ``max_concurrency`` significantly.
    """
    model: str
    api_key: str | None = None  # falls back to OPENAI_API_KEY
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024
    max_concurrency: int = 16
    timeout: float = 60.0
    max_retries: int = 4

    def __post_init__(self) -> None:
        # Imported lazily so the package import doesn't require openai
        # installed unless a pipeline actually instantiates a client.
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=self.api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        self._sem = asyncio.Semaphore(self.max_concurrency)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        async with self._sem:
            messages = []
            if request.system:
                messages.append({"role": "system", "content": request.system})
            messages.append({"role": "user", "content": request.user})

            kwargs: dict = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if request.response_format == "json_object":
                kwargs["response_format"] = {"type": "json_object"}

            try:
                resp = await self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 — surface as in-band error
                return LLMResponse(text="", metadata=request.metadata, error=repr(exc))

            choice = resp.choices[0]
            usage = getattr(resp, "usage", None)
            return LLMResponse(
                text=choice.message.content or "",
                metadata=request.metadata,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
            )

    async def generate_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        return await asyncio.gather(*(self.generate(r) for r in requests))
