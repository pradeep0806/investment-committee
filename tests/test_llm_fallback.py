"""Primary/fallback provider on transient (429/503) failure — see CLAUDE.md's
task brief. Retry-with-backoff and fallback both live inside LLMClient.call()
(client.py); BudgetGate is untouched structurally — it just receives one more
value in the returned tuple (provider_used) alongside (result, tokens_used).

Real anthropic.APIStatusError/subclass instances are used as the fake raw
callers' raised errors (rather than a bespoke fake exception) so
is_transient_error's `getattr(exc, "status_code", None)` check is exercised
against the exact shape every provider SDK reachable from this codebase
actually raises (see structured_output.py's TRANSPORT_ERRORS discussion).
"""

from __future__ import annotations

import httpx
import pytest
from anthropic import APIStatusError, AuthenticationError

from committee.agents._base_impl import _LLMAgentOutputSchema
from committee.llm.client import LLMClient
from committee.llm.structured_output import LLMValidationError
from committee.orchestration.budget_gate import BudgetGate
from committee.orchestration.budget_manager import BudgetExhaustedError

pytestmark = pytest.mark.asyncio


_REQUEST = httpx.Request("POST", "https://example.invalid/v1/messages")


def _status_error(status_code: int, cls=APIStatusError):
    response = httpx.Response(status_code, request=_REQUEST, json={"error": {"message": "boom"}})
    return cls(f"error {status_code}", response=response, body=None)


class _ScriptedRawCaller:
    """Returns/raises whatever `script` says, in order, one entry per call.
    An entry that's an Exception instance is raised; otherwise it's returned
    as (raw_args, tokens_used)."""

    def __init__(self, script: list):
        self._script = list(script)
        self.call_count = 0

    async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens=None):
        self.call_count += 1
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item, 100


_VALID_RESPONSE = {
    "stance": "Buy",
    "confidence": 70,
    "key_factors": ["growth"],
    "evidence": ["Q3 revenue up 22% YoY"],
    "top_risk": "x",
    "executive_summary": "Growth justifies a Buy.",
}


def _make_client(
    primary_script: list,
    fallback_script: list | None = None,
    max_retries: int = 1,
    fallback_max_retries: int = 2,
    retry_backoff_seconds: float = 0,
) -> tuple[LLMClient, _ScriptedRawCaller, _ScriptedRawCaller | None]:
    client = LLMClient.__new__(LLMClient)
    client.provider = "primary-fake"
    client.model = "primary-model"
    client.api_key = "fake-key"
    client.timeout_seconds = 60
    client.max_retries = max_retries
    client.retry_backoff_seconds = retry_backoff_seconds
    client.fallback_max_retries = fallback_max_retries

    primary_caller = _ScriptedRawCaller(primary_script)
    client._raw_caller = primary_caller

    fallback_caller = None
    if fallback_script is not None:
        client.fallback_provider = "fallback-fake"
        client.fallback_model = "fallback-model"
        fallback_caller = _ScriptedRawCaller(fallback_script)
        client._fallback_raw_caller = fallback_caller
    else:
        client.fallback_provider = None
        client.fallback_model = None
        client._fallback_raw_caller = None

    return client, primary_caller, fallback_caller


async def test_transient_error_then_successful_retry_never_triggers_fallback():
    """503 on the first attempt, success on the second — call_structured's
    own inner retry loop already handles this (no backoff, same provider);
    the outer fallback loop must never even engage."""
    client, primary_caller, fallback_caller = _make_client(
        primary_script=[_status_error(503), _VALID_RESPONSE],
        fallback_script=[_VALID_RESPONSE],
        max_retries=2,
    )

    result, _tokens_used, provider_used = await client.call(
        system_prompt="sys", user_prompt="user", response_model=_LLMAgentOutputSchema
    )

    assert provider_used == "primary-fake"
    assert primary_caller.call_count == 2
    assert fallback_caller.call_count == 0
    assert result.stance.value == "Buy"


async def test_primary_retries_exhausted_falls_back_and_budget_still_deducted():
    """Every attempt against the primary (across both call_structured's
    inner loop and this client's own outer retry-with-backoff loop) returns
    503; the fallback provider is then tried and succeeds. Routed through
    BudgetGate exactly like any other call — same reservation, same
    deduction — proving the fallback cannot bypass budget enforcement."""
    # max_retries=1 (call_structured makes exactly 1 attempt per outer try);
    # fallback_max_retries=2 -> 2 outer retries against the primary, all 503,
    # before falling through to the fallback.
    client, primary_caller, fallback_caller = _make_client(
        primary_script=[_status_error(503), _status_error(503), _status_error(503)],
        fallback_script=[_VALID_RESPONSE],
        max_retries=1,
        fallback_max_retries=2,
        retry_backoff_seconds=0,
    )
    gate = BudgetGate(llm_client=client, total_budget=10_000)

    result, tokens_used, provider_used = await gate.call(
        system_prompt="sys",
        user_prompt="user",
        response_model=_LLMAgentOutputSchema,
        max_tokens=500,
    )

    assert provider_used == "fallback-fake"
    assert primary_caller.call_count == 3
    assert fallback_caller.call_count == 1
    assert result.stance.value == "Buy"
    # Budget was actually deducted for the fallback's real usage (100 tokens
    # from _ScriptedRawCaller), not skipped/bypassed because the provider
    # changed mid-call.
    assert tokens_used == 100
    assert gate.remaining == 10_000 - 100


async def test_fallback_cannot_bypass_budget_gate_enforcement():
    """Same enforcement path regardless of provider: a max_tokens request
    that would overspend the remaining gate budget is refused *before* any
    network call — including before the primary is ever attempted — whether
    or not a fallback is configured. Proves there's no separate/looser gate
    for the fallback path."""
    client, primary_caller, fallback_caller = _make_client(
        primary_script=[_status_error(503)] * 5,
        fallback_script=[_VALID_RESPONSE],
        max_retries=1,
        fallback_max_retries=2,
    )
    gate = BudgetGate(llm_client=client, total_budget=100)

    with pytest.raises(BudgetExhaustedError):
        await gate.call(
            system_prompt="sys",
            user_prompt="user",
            response_model=_LLMAgentOutputSchema,
            max_tokens=500,
        )

    # Refused at reservation time — neither the primary nor the fallback
    # raw caller was ever invoked.
    assert primary_caller.call_count == 0
    assert fallback_caller.call_count == 0


async def test_non_transient_error_does_not_trigger_fallback():
    """An auth failure (401) is not a transient signal — no amount of
    retrying the primary or switching to the fallback would fix a bad API
    key, so neither path should be attempted. The error propagates as
    LLMValidationError (call_structured's normal wrapping of any
    TRANSPORT_ERRORS-exhausted call), not silently swallowed."""
    client, primary_caller, fallback_caller = _make_client(
        primary_script=[_status_error(401, cls=AuthenticationError)],
        fallback_script=[_VALID_RESPONSE],
        max_retries=1,
        fallback_max_retries=2,
    )

    with pytest.raises(LLMValidationError):
        await client.call(system_prompt="sys", user_prompt="user", response_model=_LLMAgentOutputSchema)

    assert primary_caller.call_count == 1
    assert fallback_caller.call_count == 0


async def test_no_fallback_configured_raises_after_primary_exhausted():
    """Without a fallback provider configured (the default, matching every
    existing debate today), a persistent transient error still eventually
    raises — this is the pre-existing, unchanged behavior when no fallback
    is set up, proving the new code path is fully opt-in."""
    client, primary_caller, fallback_caller = _make_client(
        primary_script=[_status_error(503)] * 5,
        fallback_script=None,
        max_retries=1,
        fallback_max_retries=2,
    )

    with pytest.raises(LLMValidationError):
        await client.call(system_prompt="sys", user_prompt="user", response_model=_LLMAgentOutputSchema)

    assert fallback_caller is None
    assert primary_caller.call_count == 3  # 1 initial + 2 fallback_max_retries retries
