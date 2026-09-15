"""Provider-agnostic LLM wrapper.

This is the ONLY place in the codebase that imports a provider SDK. Every agent
calls `LLMClient.call(...)`, never `anthropic`/`openai`/`litellm` directly, so
switching providers is a `.env` edit (`LLM_PROVIDER` / `LLM_MODEL` / `LLM_API_KEY`),
never an orchestrator or agent code change (CLAUDE.md §1.4).
"""

from __future__ import annotations

import asyncio
import time
from typing import TypeVar

from pydantic import BaseModel

from committee.config import Settings
from committee.llm.structured_output import (
    TRANSPORT_ERRORS,
    LLMValidationError,
    call_structured,
    is_transient_error,
)
from committee.observability.metrics import (
    llm_call_errors_total,
    llm_call_latency_seconds,
    llm_fallback_invocations_total,
)

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
        retry_backoff_seconds: float = 1.0,
        fallback_provider: str | None = None,
        fallback_model: str | None = None,
        fallback_api_key: str | None = None,
        fallback_max_retries: int = 2,
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
        # Base delay for exponential backoff between retries of the primary
        # provider on a transient (429/503) error — attempt N sleeps
        # retry_backoff_seconds * 2**(N-1) before the next attempt.
        self.retry_backoff_seconds = retry_backoff_seconds
        self._raw_caller = self._build_raw_caller(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
        )

        # Fallback is entirely optional — fallback_provider is None unless
        # explicitly configured (Settings.llm_fallback_provider), in which
        # case a second, independent RawCaller is built up front (same
        # construction path as the primary, just parameterized) so it's
        # ready to use the moment retries on the primary exhaust, with no
        # per-call construction cost or partial-init risk mid-debate.
        self.fallback_provider = fallback_provider
        self.fallback_model = fallback_model
        self.fallback_max_retries = fallback_max_retries
        self._fallback_raw_caller = None
        if fallback_provider is not None:
            if not fallback_model:
                raise ValueError("LLM_FALLBACK_PROVIDER is set but LLM_FALLBACK_MODEL is missing")
            self._fallback_raw_caller = self._build_raw_caller(
                provider=fallback_provider,
                model=fallback_model,
                api_key=fallback_api_key or "",
            )

    def _build_raw_caller(self, provider: str, model: str, api_key: str):
        """Builds a RawCaller for `provider`/`model`/`api_key` — used for
        both the primary client construction and (when configured) the
        fallback, so the two share identical provider-selection logic
        rather than duplicating it. vertex_project/vertex_location/
        temperature/thinking_budget/base_url are shared client-level
        settings (not distinct per primary/fallback) since the fallback is
        expected to be a plain hosted model, not another Vertex/Ollama
        setup — the litellm path below still accepts them for parity, they
        just won't be populated unless also relevant to the fallback."""
        if provider == "anthropic":
            from committee.llm.providers.anthropic_provider import AnthropicRawCaller

            return AnthropicRawCaller(
                model=model,
                api_key=api_key,
                timeout_seconds=self.timeout_seconds,
                temperature=self.temperature,
            )
        if provider == "openai":
            from committee.llm.providers.openai_provider import OpenAIRawCaller

            return OpenAIRawCaller(
                model=model,
                api_key=api_key,
                timeout_seconds=self.timeout_seconds,
                temperature=self.temperature,
            )
        if provider == "litellm":
            from committee.llm.providers.litellm_provider import LiteLLMRawCaller

            return LiteLLMRawCaller(
                model=model,
                api_key=api_key,
                timeout_seconds=self.timeout_seconds,
                vertex_project=self.vertex_project,
                vertex_location=self.vertex_location,
                temperature=self.temperature,
                thinking_budget=self.thinking_budget,
                base_url=self.base_url,
            )
        raise ValueError(f"Unknown LLM provider: {provider!r} (expected anthropic|openai|litellm)")

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelT],
        max_retries: int | None = None,
        max_tokens: int | None = None,
    ) -> tuple[ModelT, int, str]:
        """Returns (validated_instance, tokens_used, provider_used). Raises
        LLMValidationError after exhausting every attempt against both the
        primary and (if configured) the fallback provider.

        `max_tokens`, when given, becomes a real API-level output-token cap
        on the underlying provider call (clamped to a sane floor inside
        call_structured) — not advisory. Pass the caller's token_budget here
        to actually enforce it, rather than relying on the model to respect
        a number it's never told about.

        Two independent retry layers are stacked here, deliberately kept
        separate rather than merged into one loop:
          - call_structured's own inner loop (unchanged) retries immediately,
            no backoff, for validation failures AND any TRANSPORT_ERRORS —
            it has no concept of "transient vs. not," it just keeps trying
            the same provider up to max_retries total attempts, then raises
            LLMValidationError wrapping whichever error was last seen.
          - This method's outer loop only fires when that LLMValidationError
            wraps a *transient* error (429/503, see is_transient_error) as
            opposed to a validation failure or a non-transient transport
            error (auth, bad request) — those are NOT retried again here or
            escalated to the fallback, since another attempt (on this
            provider or a different one) has no reason to succeed where the
            first didn't. On a transient exhaustion, this method itself
            waits with exponential backoff and retries the *primary* call
            fresh (a full new call_structured attempt sequence) up to
            fallback_max_retries times; if every one of those also exhausts
            on a transient error, and a fallback provider is configured, one
            final attempt is made against the fallback — still through
            call_structured, so validation/retry-on-invalid-output behaves
            identically regardless of which provider ultimately serves it.

        `llm_call_latency_seconds`/`llm_call_errors_total` are recorded once
        per provider actually attempted (so a fallback that ends up serving
        the call is measured under its own provider label, not folded into
        the primary's) — not once per outer retry attempt against the same
        provider, since those aren't independent calls worth double-counting
        as separate latency samples for the *provider* metric (call_structured
        already retries those internally)."""
        primary_result = await self._call_one(
            raw_caller=self._raw_caller,
            provider=self.provider,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
            max_retries=max_retries if max_retries is not None else self.max_retries,
            max_tokens=max_tokens,
        )
        if not isinstance(primary_result, LLMValidationError):
            result, tokens_used = primary_result
            return result, tokens_used, self.provider

        last_error = primary_result
        for attempt in range(1, self.fallback_max_retries + 1):
            if not is_transient_error(last_error.last_error):
                raise last_error

            await asyncio.sleep(self.retry_backoff_seconds * (2 ** (attempt - 1)))
            outcome = await self._call_one(
                raw_caller=self._raw_caller,
                provider=self.provider,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=response_model,
                max_retries=max_retries if max_retries is not None else self.max_retries,
                max_tokens=max_tokens,
            )
            if not isinstance(outcome, LLMValidationError):
                result, tokens_used = outcome
                return result, tokens_used, self.provider
            last_error = outcome

        if (
            not is_transient_error(last_error.last_error)
            or self._fallback_raw_caller is None
            or self.fallback_provider is None
        ):
            raise last_error
        fallback_provider = self.fallback_provider

        llm_fallback_invocations_total.labels(
            from_provider=self.provider, to_provider=fallback_provider
        ).inc()
        fallback_outcome = await self._call_one(
            raw_caller=self._fallback_raw_caller,
            provider=fallback_provider,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
            max_retries=max_retries if max_retries is not None else self.max_retries,
            max_tokens=max_tokens,
        )
        if isinstance(fallback_outcome, LLMValidationError):
            raise fallback_outcome
        result, tokens_used = fallback_outcome
        return result, tokens_used, fallback_provider

    async def _call_one(
        self,
        raw_caller,
        provider: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelT],
        max_retries: int,
        max_tokens: int | None,
    ) -> tuple[ModelT, int] | LLMValidationError:
        """One full call_structured attempt sequence against `raw_caller`,
        with latency/error metrics recorded under `provider`'s label.
        Returns the (result, tokens_used) tuple on success, or the caught
        LLMValidationError on failure (returned, not raised, so the caller
        can inspect `.last_error` to decide whether a transient retry/
        fallback applies without relying on exception-based control flow for
        an expected, already-classified outcome)."""
        start_time = time.monotonic()
        try:
            result, tokens_used = await call_structured(
                raw_caller=raw_caller,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=response_model,
                max_retries=max_retries,
                max_tokens=max_tokens,
            )
        except LLMValidationError as exc:
            if not is_transient_error(exc.last_error):
                llm_call_errors_total.labels(provider=provider).inc()
            return exc
        except Exception:
            llm_call_errors_total.labels(provider=provider).inc()
            raise
        finally:
            llm_call_latency_seconds.labels(provider=provider).observe(
                time.monotonic() - start_time
            )
        return result, tokens_used


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
        retry_backoff_seconds=settings.llm_retry_backoff_seconds,
        # Fallback is server-only (no per-debate override) — same reasoning
        # as API key/Vertex project/base_url above: it's operational
        # resilience config, not something a per-request caller should be
        # able to redirect.
        fallback_provider=settings.llm_fallback_provider,
        fallback_model=settings.llm_fallback_model,
        fallback_api_key=settings.llm_fallback_api_key,
        fallback_max_retries=settings.llm_fallback_max_retries,
    )
