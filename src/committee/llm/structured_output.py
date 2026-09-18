"""Validated structured-output call with a bounded retry-on-invalid loop.

Never trust raw JSON parsing of free text: every call goes through tool-calling
so the provider returns arguments matching a JSON schema, and the result is
still re-validated through Pydantic before it's trusted. If validation fails,
the validation error is fed back to the model verbatim and it gets another
attempt, up to `max_retries` total attempts.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

import structlog
from pydantic import BaseModel, ValidationError

logger = structlog.get_logger()

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

# Suffix appended after truncating an over-long string field, so a
# truncated executive_summary is visibly not the model's complete thought
# rather than looking like a naturally short one.
_TRUNCATION_SUFFIX = "..."


def _truncate_over_long_strings(raw_args: dict, exc: ValidationError) -> dict | None:
    """Best-effort mechanical repair for the one validation failure mode that
    doesn't reflect anything wrong with the model's actual answer: a
    top-level string field that is otherwise well-formed but exceeds its
    schema's `max_length`.

    Real bug found via a live debate against Gemini: `executive_summary`
    (a free-text digest with `max_length=280`) came back a few characters
    over the cap, with every other field — stance, confidence, key_factors,
    top_risk — fully valid. `call_structured`'s retry loop fed the error
    back verbatim for all `max_retries` attempts and the model kept
    regenerating a similarly-length summary rather than reliably
    self-truncating (models don't count characters as they generate), so
    the *entire* agent output was discarded over one cosmetic field.

    Returns a repaired copy of `raw_args` only when every error in `exc` is
    a `string_too_long` on a top-level string field — i.e. the fix is a
    lossless, mechanical truncation of exactly the field pydantic flagged,
    never a guess applied to a field that failed for some other reason
    (missing, wrong type, nested error, etc.). Returns None otherwise, so
    the caller falls through to the normal retry-with-feedback path.
    """
    errors = exc.errors()
    if not errors or not all(err["type"] == "string_too_long" and len(err["loc"]) == 1 for err in errors):
        return None
    repaired = dict(raw_args)
    for err in errors:
        field_name = err["loc"][0]
        max_length = err["ctx"]["max_length"]
        value = err["input"]
        if not isinstance(value, str) or max_length <= len(_TRUNCATION_SUFFIX):
            return None
        repaired[field_name] = value[: max_length - len(_TRUNCATION_SUFFIX)] + _TRUNCATION_SUFFIX
    return repaired


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
    they're the same outcome: no usable output after max_retries.

    `total_tokens_used` is the real cumulative spend across every attempt
    made before giving up — not discarded on failure. A real bug found via
    a live debate against a weak local model (qwen3.5:9b via Ollama):
    BudgetGate.call() previously released the *full* reservation on any
    exception, including this one, silently crediting back budget for
    tokens that were genuinely spent against the real provider across
    several failed attempts. Callers (BudgetGate, orchestrator ledger
    accounting) must reconcile against this value instead of assuming zero
    spend on an excluded agent."""

    def __init__(self, attempts: int, last_error: ValidationError | Exception, total_tokens_used: int = 0):
        self.attempts = attempts
        self.last_error = last_error
        self.total_tokens_used = total_tokens_used
        super().__init__(f"Structured output failed validation after {attempts} attempt(s): {last_error}")


# Ceiling on cumulative spend across every retry attempt combined, as a
# multiple of the caller's original max_tokens allocation. Real bug found
# via a live debate against qwen3.5:9b (a weak local model via Ollama):
# each individual attempt's own max_tokens cap was correctly honored by the
# provider every time (verified directly against both Ollama and Vertex AI
# Gemini — this was never a per-call cap-enforcement problem), but the
# retry loop re-issued the *same* max_tokens on every attempt with no
# ceiling on the running total, so a model that needed 3 attempts to
# produce a valid tool call could spend roughly 3x its allocation (observed
# live: tokens_used=7044 against tokens_allocated=1853, a ~3.8x overrun) —
# an unbounded-in-practice accumulation, not the "small, bounded" prompt-
# size gap this module previously (and incorrectly) claimed was the only
# residual asymmetry. RETRY_BUDGET_MULTIPLIER caps total spend across all
# attempts combined at this multiple of the original allocation; once the
# remaining allowance can't support another attempt at MIN_MAX_TOKENS, the
# loop stops retrying rather than force a doomed attempt that would only
# add to the overrun.
RETRY_BUDGET_MULTIPLIER = 2


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
    well-formed tool call rather than a truncated/invalid one. Cumulative
    spend across every attempt is separately capped at
    RETRY_BUDGET_MULTIPLIER times the original `max_tokens` (see that
    constant's docstring) — each subsequent attempt's own cap shrinks by
    what prior attempts already spent, and the loop stops retrying early
    (raising LLMValidationError, exactly as if validation had failed) once
    the remaining allowance can no longer support a viable attempt.

    Returns (validated_instance, total_tokens_used_across_all_attempts) on
    success. On failure, raises LLMValidationError carrying
    `total_tokens_used` — real spend across every attempt made is never
    discarded, even though no valid result was produced (see that
    exception's docstring for why this matters to callers).
    """
    schema = response_model.model_json_schema()
    total_tokens_used = 0
    retry_note: str | None = None
    last_error: ValidationError | Exception | None = None
    effective_max_tokens = max(max_tokens, MIN_MAX_TOKENS) if max_tokens is not None else None
    retry_budget = (
        effective_max_tokens * RETRY_BUDGET_MULTIPLIER if effective_max_tokens is not None else None
    )

    for attempt in range(1, max_retries + 1):
        attempt_max_tokens = effective_max_tokens
        if retry_budget is not None and effective_max_tokens is not None:
            remaining_retry_budget = retry_budget - total_tokens_used
            if remaining_retry_budget < MIN_MAX_TOKENS:
                # Even a floor-sized attempt would blow past the cumulative
                # ceiling — stop here rather than force one more doomed,
                # budget-compounding attempt. Same outcome as exhausting
                # max_retries: the agent is excluded, with whatever was
                # genuinely spent already reported via total_tokens_used.
                logger.warning(
                    "retry_budget_exhausted",
                    attempt=attempt,
                    total_tokens_used=total_tokens_used,
                    retry_budget=retry_budget,
                )
                break
            attempt_max_tokens = min(effective_max_tokens, remaining_retry_budget)

        try:
            raw_args, tokens_used = await raw_caller(
                system_prompt, user_prompt, schema, retry_note, attempt_max_tokens
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
            repaired_args = _truncate_over_long_strings(raw_args, exc)
            if repaired_args is not None:
                try:
                    validated = response_model.model_validate(repaired_args)
                except ValidationError:
                    pass
                else:
                    logger.warning(
                        "truncated_over_long_field",
                        fields=[err["loc"][0] for err in exc.errors()],
                    )
                    return validated, total_tokens_used
            last_error = exc
            retry_note = (
                f"Your previous response failed schema validation: {exc}. "
                "Correct the response to match the required schema exactly."
            )
            continue
        return validated, total_tokens_used

    if last_error is None:
        # Reached only by the retry_budget_exhausted break above, before any
        # attempt in this call ever ran (e.g. the very first attempt's own
        # cost already left no room) — LLMValidationError still requires a
        # last_error, so synthesize one that names the actual cause rather
        # than asserting on a None that would otherwise crash here.
        last_error = RuntimeError(
            f"Retry budget of {retry_budget} tokens exhausted after {total_tokens_used} "
            "tokens spent, before another attempt could be made."
        )
    raise LLMValidationError(attempts=max_retries, last_error=last_error, total_tokens_used=total_tokens_used)
