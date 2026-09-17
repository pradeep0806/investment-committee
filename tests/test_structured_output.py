import openai
import pytest
from pydantic import BaseModel

from committee.llm.structured_output import (
    MIN_MAX_TOKENS,
    RETRY_BUDGET_MULTIPLIER,
    LLMValidationError,
    call_structured,
)


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


class _NonEnforcingRawCaller:
    """Real bug found via a live debate against qwen3.5:9b on Ollama: this
    is NOT a provider that ignores the requested max_tokens itself (that
    part was verified correct against both Ollama and Vertex AI Gemini via
    real calls) — every individual attempt's own cap is honored. The actual
    gap was call_structured re-issuing the *same* max_tokens on every
    retry with no ceiling on the running total, so a model needing several
    attempts to produce a valid tool call could spend roughly
    max_retries x max_tokens. This stub always fails validation and always
    reports using exactly the max_tokens it was given (i.e. it "would
    otherwise generate unbounded output" across repeated full-cost
    attempts if nothing capped the cumulative total) — modeling that
    scenario precisely."""

    def __init__(self):
        self.seen_max_tokens: list[int] = []

    async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens):
        self.seen_max_tokens.append(max_tokens)
        return {"not_value": "always invalid"}, max_tokens


class _IgnoresLimitsRawCaller:
    """A raw_caller that truly ignores the max_tokens it's given — reports
    generating a fixed, large amount of output regardless of what was
    requested, modeling a provider/model that doesn't respect the limit at
    all (the task's literal framing, distinct from _NonEnforcingRawCaller
    above which respects the per-attempt cap exactly but still needed the
    retry-total ceiling)."""

    def __init__(self, fixed_output: int):
        self._fixed_output = fixed_output
        self.seen_max_tokens: list[int] = []

    async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens):
        self.seen_max_tokens.append(max_tokens)
        return {"not_value": "always invalid"}, self._fixed_output


class _WellBehavedRawCaller:
    """A stub that produces valid output on the first attempt, well within
    its allocation — the "normal" case that must be completely unaffected
    by the retry-budget cap, since it never needs a second attempt."""

    def __init__(self, tokens_used: int = 50):
        self._tokens_used = tokens_used
        self.call_count = 0

    async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens):
        self.call_count += 1
        return {"value": "ok"}, self._tokens_used


async def test_cumulative_spend_is_capped_when_provider_ignores_limits_across_retries():
    """The actual fix: a raw_caller whose every attempt reports using its
    full max_tokens (never producing valid output, so every attempt is a
    full-cost retry) must not be allowed to accumulate unbounded spend
    across max_retries attempts — cumulative total_tokens_used, whether the
    call ultimately succeeds or exhausts retries, is capped at
    RETRY_BUDGET_MULTIPLIER x the original max_tokens allocation."""
    raw_caller = _NonEnforcingRawCaller()

    with pytest.raises(LLMValidationError) as exc_info:
        await call_structured(
            raw_caller=raw_caller,
            system_prompt="sys",
            user_prompt="user",
            response_model=_SimpleSchema,
            max_retries=10,  # would have compounded to 10x allocation pre-fix
            max_tokens=1000,
        )

    assert exc_info.value.total_tokens_used <= 1000 * RETRY_BUDGET_MULTIPLIER
    # This raw_caller reports using exactly the max_tokens it's given each
    # time, so with a 2x retry budget exactly two full-cost attempts fit
    # (1000 + 1000 == the 2000 ceiling) before a third would exceed it —
    # proving the loop stopped itself early rather than proceeding to all
    # 10 max_retries, which is what would have produced the old, unbounded
    # ~10x overrun this fix closes.
    assert raw_caller.seen_max_tokens == [1000, 1000]
    assert len(raw_caller.seen_max_tokens) < 10


async def test_cumulative_spend_capped_when_provider_generates_far_more_than_requested():
    """A provider that generates unbounded output regardless of what it's
    asked for (the task's literal "would otherwise generate unbounded
    output" framing): every attempt reports 5000 tokens used no matter
    what max_tokens says. Cumulative spend across the whole retry sequence
    must still be bounded — application-side, since the provider itself
    cannot be trusted to enforce anything here — even though each
    individual attempt's real cost already exceeds the entire retry
    budget on its own."""
    raw_caller = _IgnoresLimitsRawCaller(fixed_output=5000)

    with pytest.raises(LLMValidationError) as exc_info:
        await call_structured(
            raw_caller=raw_caller,
            system_prompt="sys",
            user_prompt="user",
            response_model=_SimpleSchema,
            max_retries=10,
            max_tokens=1000,
        )

    # Only one attempt is ever made: its real cost (5000) alone already
    # exceeds the 2000 retry budget, so there's no room left for a second
    # attempt regardless of what max_retries allows.
    assert raw_caller.seen_max_tokens == [1000]
    assert exc_info.value.total_tokens_used == 5000


async def test_well_behaved_provider_unaffected_by_retry_budget_cap():
    """A raw_caller that succeeds on the first attempt within its
    allocation must see exactly its requested max_tokens, completely
    unaffected by the new retry-budget machinery — normal behavior for a
    hard-enforcing/well-behaved provider is unchanged (hard constraint:
    "do not change behavior for hard-enforcing providers... beyond adding
    the capability flag and verification")."""
    raw_caller = _WellBehavedRawCaller(tokens_used=50)

    result, tokens_used = await call_structured(
        raw_caller=raw_caller,
        system_prompt="sys",
        user_prompt="user",
        response_model=_SimpleSchema,
        max_retries=3,
        max_tokens=1000,
    )

    assert result.value == "ok"
    assert tokens_used == 50
    assert raw_caller.call_count == 1


async def test_llm_validation_error_carries_real_cumulative_spend_on_failure():
    """Real accounting bug found in the same session: LLMValidationError
    previously carried no token usage at all, so a caller reconciling
    budget against a failed call (BudgetGate.call()) had no way to know
    real tokens were spent across the failed attempts and silently
    credited back the full reservation as if nothing had happened. Every
    attempt here reports a small, fixed cost so the total is exactly
    predictable."""

    class _AlwaysInvalidFixedCostRawCaller:
        def __init__(self):
            self.call_count = 0

        async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens):
            self.call_count += 1
            return {"not_value": "wrong shape"}, 30

    raw_caller = _AlwaysInvalidFixedCostRawCaller()

    with pytest.raises(LLMValidationError) as exc_info:
        await call_structured(
            raw_caller=raw_caller,
            system_prompt="sys",
            user_prompt="user",
            response_model=_SimpleSchema,
            max_retries=3,
            max_tokens=1000,
        )

    assert raw_caller.call_count == 3
    assert exc_info.value.total_tokens_used == 90  # 3 attempts x 30 tokens each


async def test_retry_loop_stops_early_once_retry_budget_cannot_support_another_attempt():
    """Once the remaining retry-budget allowance drops below MIN_MAX_TOKENS,
    the loop must stop retrying immediately rather than force one more
    doomed, budget-compounding attempt — even if max_retries hasn't been
    reached yet. A tiny max_tokens (clamped to MIN_MAX_TOKENS) combined with
    a raw_caller that always spends the full MIN_MAX_TOKENS per attempt
    leaves no room for a second attempt within a 2x retry budget."""
    raw_caller = _NonEnforcingRawCaller()

    with pytest.raises(LLMValidationError):
        await call_structured(
            raw_caller=raw_caller,
            system_prompt="sys",
            user_prompt="user",
            response_model=_SimpleSchema,
            max_retries=10,
            max_tokens=1,  # clamped to MIN_MAX_TOKENS=256; retry budget = 512
        )

    # 512 / 256 == 2 exactly, so exactly 2 attempts fit before the
    # remaining allowance drops below MIN_MAX_TOKENS for a third.
    assert len(raw_caller.seen_max_tokens) == 2


async def test_synthesize_never_receives_an_excluded_agents_output():
    """Phase 1-C's structured-output-corruption concern, proven at the
    orchestrator boundary rather than re-testing call_structured's own
    exclusion logic: an agent whose structured output never validated
    (including one that hit the retry-budget cap and stopped early) must
    never reach synthesis as if it were a valid AgentOutput. This is
    already structurally guaranteed by the orchestrator's control flow
    (the `continue` on LLMValidationError skips agent_outputs.append), and
    is exercised end-to-end in test_orchestrator.py's
    test_orchestrator_excludes_agent_that_fails_structured_output_validation
    — this test instead asserts the same guarantee directly against
    call_structured's own contract: a failure always raises, never returns
    a value, so there is no code path back to a caller that could
    mistake a truncated/invalid response for a validated one."""
    raw_caller = _NonEnforcingRawCaller()

    with pytest.raises(LLMValidationError):
        result = await call_structured(
            raw_caller=raw_caller,
            system_prompt="sys",
            user_prompt="user",
            response_model=_SimpleSchema,
            max_retries=3,
            max_tokens=1000,
        )
        # Unreachable — call_structured must raise, never return a tuple
        # whose first element is anything but a validated response_model
        # instance, on a failure path.
        assert False, f"call_structured returned {result!r} instead of raising"
