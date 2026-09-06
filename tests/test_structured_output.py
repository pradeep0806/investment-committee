import openai
import pytest
from pydantic import BaseModel

from committee.llm.structured_output import MIN_MAX_TOKENS, LLMValidationError, call_structured


class _SimpleSchema(BaseModel):
    value: str


class _CapturingRawCaller:
    def __init__(self):
        self.seen_max_tokens: list[int | None] = []

    async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens):
        self.seen_max_tokens.append(max_tokens)
        return {"value": "ok"}, 10


async def test_max_tokens_none_passes_through_as_none():
    """No token_budget to enforce (a caller that doesn't have one) means the
    provider's own default is used — not a manufactured cap."""
    raw_caller = _CapturingRawCaller()
    await call_structured(
        raw_caller=raw_caller,
        system_prompt="sys",
        user_prompt="user",
        response_model=_SimpleSchema,
        max_retries=1,
        max_tokens=None,
    )
    assert raw_caller.seen_max_tokens == [None]


async def test_max_tokens_above_floor_passes_through_unchanged():
    raw_caller = _CapturingRawCaller()
    await call_structured(
        raw_caller=raw_caller,
        system_prompt="sys",
        user_prompt="user",
        response_model=_SimpleSchema,
        max_retries=1,
        max_tokens=1000,
    )
    assert raw_caller.seen_max_tokens == [1000]


async def test_max_tokens_below_floor_is_clamped_up():
    """A tiny token_budget allocation (e.g. a heavily-degraded later round)
    must not produce a cap so small the tool call itself can't come back
    well-formed — clamped to MIN_MAX_TOKENS instead of passed through raw."""
    raw_caller = _CapturingRawCaller()
    await call_structured(
        raw_caller=raw_caller,
        system_prompt="sys",
        user_prompt="user",
        response_model=_SimpleSchema,
        max_retries=1,
        max_tokens=1,
    )
    assert raw_caller.seen_max_tokens == [MIN_MAX_TOKENS]


async def test_max_tokens_clamp_applies_consistently_across_retries():
    class _AlwaysInvalidRawCaller:
        def __init__(self):
            self.seen_max_tokens: list[int | None] = []

        async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens):
            self.seen_max_tokens.append(max_tokens)
            return {"not_value": "wrong shape"}, 10

    raw_caller = _AlwaysInvalidRawCaller()
    from committee.llm.structured_output import LLMValidationError
    import pytest

    with pytest.raises(LLMValidationError):
        await call_structured(
            raw_caller=raw_caller,
            system_prompt="sys",
            user_prompt="user",
            response_model=_SimpleSchema,
            max_retries=3,
            max_tokens=1,
        )

    assert raw_caller.seen_max_tokens == [MIN_MAX_TOKENS, MIN_MAX_TOKENS, MIN_MAX_TOKENS]


def _fake_api_connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=None, message="connection timed out")


async def test_transport_error_is_retried_like_a_validation_failure():
    """Real bug found via live testing: a raw_caller can fail at the
    transport level (timeout, dropped connection) rather than returning bad
    output — e.g. litellm raised an APIConnectionError after a 60s timeout
    talking to Ollama through Docker. This used to propagate straight
    through call_structured uncaught, crashing the entire debate with a raw
    500 instead of retrying (and, if every attempt fails this way,
    excluding just that one agent like any other structured-output
    failure)."""

    class _FailsOnceThenSucceedsRawCaller:
        def __init__(self):
            self.call_count = 0

        async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens):
            self.call_count += 1
            if self.call_count == 1:
                raise _fake_api_connection_error()
            return {"value": "ok"}, 10

    raw_caller = _FailsOnceThenSucceedsRawCaller()
    result, tokens_used = await call_structured(
        raw_caller=raw_caller,
        system_prompt="sys",
        user_prompt="user",
        response_model=_SimpleSchema,
        max_retries=3,
        max_tokens=500,
    )

    assert result.value == "ok"
    assert tokens_used == 10  # the failed first attempt contributed no tokens
    assert raw_caller.call_count == 2


async def test_transport_error_on_every_attempt_raises_llm_validation_error():
    class _AlwaysFailsRawCaller:
        def __init__(self):
            self.call_count = 0

        async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens):
            self.call_count += 1
            raise _fake_api_connection_error()

    raw_caller = _AlwaysFailsRawCaller()

    with pytest.raises(LLMValidationError) as exc_info:
        await call_structured(
            raw_caller=raw_caller,
            system_prompt="sys",
            user_prompt="user",
            response_model=_SimpleSchema,
            max_retries=3,
            max_tokens=500,
        )

    assert raw_caller.call_count == 3
    assert isinstance(exc_info.value.last_error, openai.APIConnectionError)
