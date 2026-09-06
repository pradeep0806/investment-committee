"""Provider-agnostic LLM wrapper.

This is the ONLY place in the codebase that imports a provider SDK. Every agent
calls `LLMClient.call(...)`, never `anthropic`/`openai`/`litellm` directly, so
switching providers is a `.env` edit (`LLM_PROVIDER` / `LLM_MODEL` / `LLM_API_KEY`),
never an orchestrator or agent code change (CLAUDE.md §1.4).
"""

from __future__ import annotations

import time
from typing import TypeVar

from pydantic import BaseModel

from committee.config import Settings
from committee.llm.structured_output import LLMValidationError, call_structured
from committee.observability.metrics import llm_call_errors_total, llm_call_latency_seconds

ModelT = TypeVar("ModelT", bound=BaseModel)

__all__ = ["LLMClient", "LLMValidationError", "build_llm_client_from_settings"]


class LLMClient:
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        timeout_seconds: int = 60,
        max_retries: int = 3,
        vertex_project: str | None = None,
        vertex_location: str | None = None,
        temperature: float | None = None,
        thinking_budget: int | None = None,
        base_url: str | None = None,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.vertex_project = vertex_project
        self.vertex_location = vertex_location
        # temperature: passed to every provider's actual API call, not
        # previously wired at all (every call used the provider's own
        # default). thinking_budget: only meaningful for Gemini via
        # litellm — see LiteLLMRawCaller. base_url: only meaningful for a
        # local/self-hosted model server (e.g. Ollama's default
        # http://localhost:11434) — ignored by hosted providers, which
        # resolve their own endpoint from the model string / auth.
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.base_url = base_url
        self._raw_caller = self._build_raw_caller()

    def _build_raw_caller(self):
        if self.provider == "anthropic":
            from committee.llm.providers.anthropic_provider import AnthropicRawCaller

            return AnthropicRawCaller(
                model=self.model,
                api_key=self.api_key,
                timeout_seconds=self.timeout_seconds,
                temperature=self.temperature,
            )
        if self.provider == "openai":
            from committee.llm.providers.openai_provider import OpenAIRawCaller

            return OpenAIRawCaller(
                model=self.model,
                api_key=self.api_key,
                timeout_seconds=self.timeout_seconds,
                temperature=self.temperature,
            )
        if self.provider == "litellm":
            from committee.llm.providers.litellm_provider import LiteLLMRawCaller

            return LiteLLMRawCaller(
                model=self.model,
                api_key=self.api_key,
                timeout_seconds=self.timeout_seconds,
                vertex_project=self.vertex_project,
                vertex_location=self.vertex_location,
                temperature=self.temperature,
                thinking_budget=self.thinking_budget,
                base_url=self.base_url,
            )
        raise ValueError(f"Unknown LLM_PROVIDER: {self.provider!r} (expected anthropic|openai|litellm)")

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelT],
        max_retries: int | None = None,
        max_tokens: int | None = None,
    ) -> tuple[ModelT, int]:
        """Returns (validated_instance, tokens_used). Raises LLMValidationError
        after exhausting `max_retries` (defaults to the client's configured value).

        `max_tokens`, when given, becomes a real API-level output-token cap
        on the underlying provider call (clamped to a sane floor inside
        call_structured) — not advisory. Pass the caller's token_budget here
        to actually enforce it, rather than relying on the model to respect
        a number it's never told about.

        `llm_call_latency_seconds` times the whole call (including any
        retries-on-invalid-output, since those are still calls to the same
        provider). `llm_call_errors_total` counts only *unexpected* failures —
        network errors, auth failures, provider outages — not
        LLMValidationError, which is an expected, already-handled outcome the
        orchestrator excludes the agent for, not a call failure."""
        start_time = time.monotonic()
        try:
            result = await call_structured(
                raw_caller=self._raw_caller,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=response_model,
                max_retries=max_retries if max_retries is not None else self.max_retries,
                max_tokens=max_tokens,
            )
        except LLMValidationError:
            raise
        except Exception:
            llm_call_errors_total.labels(provider=self.provider).inc()
            raise
        finally:
            llm_call_latency_seconds.labels(provider=self.provider).observe(
                time.monotonic() - start_time
            )
        return result


def build_llm_client_from_settings(
    settings: Settings,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    thinking_budget: int | None = None,
) -> LLMClient:
    """`provider`/`model`/`temperature`/`thinking_budget`, when given,
    override the server-configured `.env` defaults for this one client —
    the per-debate override path (DebateConfig.llm_provider etc.), so a
    caller can try a different model or enable Ollama without a server
    restart. API key, Vertex project/location, timeout, and base_url stay
    server-only: switching to Ollama resolves base_url from
    settings.ollama_base_url, not a client-supplied value, since letting a
    request point the server at an arbitrary URL is not something to accept
    from outside.
    """
    resolved_provider = provider or settings.llm_provider
    base_url = settings.ollama_base_url if resolved_provider == "litellm" and (
        model or settings.llm_model
    ).startswith("ollama/") else None

    return LLMClient(
        provider=resolved_provider,
        model=model or settings.llm_model,
        api_key=settings.llm_api_key,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        vertex_project=settings.llm_vertex_project,
        vertex_location=settings.llm_vertex_location,
        temperature=temperature,
        thinking_budget=thinking_budget,
        base_url=base_url,
    )
