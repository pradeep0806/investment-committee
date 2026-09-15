"""Validated structured-output call with a bounded retry-on-invalid loop.

Never trust raw JSON parsing of free text: every call goes through tool-calling
so the provider returns arguments matching a JSON schema, and the result is
still re-validated through Pydantic before it's trusted. If validation fails,
the validation error is fed back to the model verbatim and it gets another
attempt, up to `max_retries` total attempts.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)

# Transport-level failures a provider call can raise — timeouts, dropped
# connections, rate limits, transient 5xxs — that are retryable in exactly
# the same sense as a validation failure: the call didn't produce usable
# output, and another attempt might. Real bug found via live testing: a
# 60s-timeout APIConnectionError from litellm (Ollama under Docker taking
# longer than LLM_TIMEOUT_SECONDS for one slow agent call) propagated
# straight through call_structured uncaught, crashing the entire debate
# with a raw 500 instead of retrying/excluding just that one agent. Every
# provider here ultimately raises exceptions rooted in openai's hierarchy
# (litellm mirrors it; the Anthropic SDK has its own APIError but litellm's
# is what's actually reachable from LiteLLMRawCaller) — importing both
# covers every RawCaller implementation without a blanket `except Exception`
# that would also swallow real bugs (TypeError, AttributeError, etc.).
try:
    import anthropic

    _ANTHROPIC_TRANSPORT_ERRORS: tuple[type[Exception], ...] = (anthropic.APIError,)
except ImportError:  # pragma: no cover - anthropic is a core dependency here
    _ANTHROPIC_TRANSPORT_ERRORS = ()

try:
    import openai

    _OPENAI_TRANSPORT_ERRORS: tuple[type[Exception], ...] = (openai.APIError,)
except ImportError:  # pragma: no cover - openai is a core dependency here
    _OPENAI_TRANSPORT_ERRORS = ()

TRANSPORT_ERRORS: tuple[type[Exception], ...] = _ANTHROPIC_TRANSPORT_ERRORS + _OPENAI_TRANSPORT_ERRORS

# Status codes worth a bounded retry-with-backoff (and, on exhaustion,
# a fallback-provider attempt) rather than immediate agent exclusion: 429
# (rate limited) and 503 (overloaded/service unavailable) commonly clear on
# retry and don't indicate anything wrong with the request itself. Anything
# else (401 auth, 400 bad request, 404, etc.) is not retried this way —
# retrying or falling back on a malformed/unauthorized request would just
# repeat the same failure against a second provider for no benefit.
#
# Every provider SDK reachable from this codebase (anthropic, openai, and
# litellm — which subclasses openai's exception hierarchy, confirmed via
# litellm.RateLimitError.__mro__ including openai.APIStatusError) exposes
# the HTTP status on a `.status_code` attribute of its APIStatusError-rooted
# exceptions, which is exactly the TRANSPORT_ERRORS set above — so checking
# status_code generically here covers all three without enumerating each
# SDK's own RateLimitError/OverloadedError/ServiceUnavailableError class by
# name (litellm alone would need its own distinct classes tracked, since
# litellm.RateLimitError is not the same class as openai.RateLimitError
# despite subclassing it).
TRANSIENT_STATUS_CODES = frozenset({429, 503})


def is_transient_error(exc: Exception) -> bool:
    """True for a rate-limit/overload signal worth retrying (and, on
    exhaustion, falling back to a different provider for) — see
    TRANSIENT_STATUS_CODES above. False for anything else, including
    TRANSPORT_ERRORS with no status_code at all (e.g. a connection error
    that never got an HTTP response) — those still get *validation-style*
    retries from call_structured's existing loop, just not a provider
    fallback, since there's no evidence retrying elsewhere would help."""
    return getattr(exc, "status_code", None) in TRANSIENT_STATUS_CODES

# A structured tool-call response (stance/confidence/key_factors/top_risk,
# occasionally rebuttals) needs a few hundred tokens at minimum to come back
# well-formed; capping max_tokens below this floor risks a truncated/invalid
# tool call rather than a shorter valid one, defeating the point of a cap.
MIN_MAX_TOKENS = 256


class RawCaller(Protocol):
    """Abstraction over a single provider call that returns raw tool-call arguments.

    Implemented per-provider inside LLMClient; structured_output.py never talks
    to a provider SDK directly, only to this narrow shape.
    """

    async def __call__(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: dict,
        retry_note: str | None,
        max_tokens: int | None,
    ) -> tuple[dict, int]:
        """Returns (raw_tool_call_arguments, tokens_used_for_this_attempt).

        `max_tokens`, when given, is the real API-level output-token cap for
        this call — not advisory. `None` means "use the provider's own
        default," for callers that don't have a token_budget to enforce.

        Important asymmetry: `max_tokens` bounds *output* generation only
        (and, for Gemini, thinking tokens separately — see
        litellm_provider.py). `tokens_used_for_this_attempt` is the
        provider's `total_tokens` (input/prompt + output), which `max_tokens`
        does not — and cannot — bound. A call can still legitimately report
        more tokens_used than the max_tokens it was given, by roughly the
        size of the prompt itself. This is expected: the fix here converts
        an unbounded overrun (agents historically using 3-10x their
        allocation, driven by uncapped output/thinking) into one bounded by
        prompt length, which is small and fairly fixed per agent — not a
        residual bug to chase further.
        """
        ...


class LLMValidationError(Exception):
    """Raised when structured output still fails validation — or the
    underlying call itself never produced a response — after all retries.
    `last_error` covers both cases (a Pydantic ValidationError, or the
    transport exception from the final attempt) since from the caller's
    perspective (the orchestrator excluding this agent for the round)
    they're the same outcome: no usable output after max_retries."""

    def __init__(self, attempts: int, last_error: ValidationError | Exception):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Structured output failed validation after {attempts} attempt(s): {last_error}")


async def call_structured(
    raw_caller: RawCaller,
    system_prompt: str,
    user_prompt: str,
    response_model: type[ModelT],
    max_retries: int,
    max_tokens: int | None = None,
) -> tuple[ModelT, int]:
    """Calls `raw_caller` up to `max_retries` times, validating the result against
    `response_model` each time. On validation failure, the next attempt's prompt
    is augmented with the validation error so the model can self-correct.

    `max_tokens`, when given, is clamped to at least MIN_MAX_TOKENS before
    being passed to the provider — a real API-level cap, not advisory — so a
    small token_budget allocation still gets enough headroom to produce a
    well-formed tool call rather than a truncated/invalid one.

    Returns (validated_instance, total_tokens_used_across_all_attempts).
    """
    schema = response_model.model_json_schema()
    total_tokens_used = 0
    retry_note: str | None = None
    last_error: ValidationError | Exception | None = None
    effective_max_tokens = max(max_tokens, MIN_MAX_TOKENS) if max_tokens is not None else None

    for attempt in range(1, max_retries + 1):
        try:
            raw_args, tokens_used = await raw_caller(
                system_prompt, user_prompt, schema, retry_note, effective_max_tokens
            )
        except TRANSPORT_ERRORS as exc:
            # The call itself failed (timeout, dropped connection, rate
            # limit, transient 5xx) rather than returning bad output —
            # nothing was generated, so there's no tokens_used to add and
            # no retry_note to give the model (there's no response to point
            # at). Retry with the same prompt; if every attempt fails this
            # way, the agent gets excluded exactly like a validation
            # failure, not a crash.
            last_error = exc
            continue
        total_tokens_used += tokens_used
        try:
            validated = response_model.model_validate(raw_args)
        except ValidationError as exc:
            last_error = exc
            retry_note = (
                f"Your previous response failed schema validation: {exc}. "
                "Correct the response to match the required schema exactly."
            )
            continue
        return validated, total_tokens_used

    assert last_error is not None
    raise LLMValidationError(attempts=max_retries, last_error=last_error)
