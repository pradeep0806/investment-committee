import pytest

from committee.agents.fundamentals import FundamentalsAgent
from committee.llm.client import LLMClient
from committee.models.requests import ThesisRequest
from committee.orchestration.budget_gate import BudgetGate
from committee.orchestration.budget_manager import BudgetExhaustedError


class _CountingRawCaller:
    """Counts how many times the underlying provider was actually invoked —
    the test oracle for 'no network call was made'."""

    def __init__(self):
        self.call_count = 0

    async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens=None):
        self.call_count += 1
        return (
            {
                "stance": "Buy",
                "confidence": 65,
                "key_factors": ["growth"],
                "evidence": ["Q3 revenue up 22% YoY"],
                "top_risk": "x",
                "executive_summary": "test summary",
            },
            500,
        )


def _make_client(raw_caller) -> LLMClient:
    client = LLMClient.__new__(LLMClient)
    client.provider = "fake"
    client.model = "fake-model"
    client.api_key = "fake-key"
    client.timeout_seconds = 60
    client.max_retries = 3
    client.retry_backoff_seconds = 0
    client.fallback_provider = None
    client.fallback_model = None
    client.fallback_max_retries = 0
    client._fallback_raw_caller = None
    client._raw_caller = raw_caller
    return client


class TestGateIsTheOnlyPathToTheLLM:
    def test_base_analyst_agent_holds_no_llm_client_reference(self):
        """Structural proof, not just behavioral: the agent's instance
        attributes never include anything LLMClient-shaped — the only
        object it holds is the gate, so there's no `._llm_client`-style
        attribute a future call site could reach into and call directly."""
        raw_caller = _CountingRawCaller()
        gate = BudgetGate(llm_client=_make_client(raw_caller), total_budget=10_000)
        agent = FundamentalsAgent(budget_gate=gate)

        assert not hasattr(agent, "_llm_client")
        assert not hasattr(agent, "llm_client")
        assert isinstance(agent._budget_gate, BudgetGate)

    async def test_agent_analyze_only_reaches_the_llm_through_the_gate(self):
        """End-to-end: calling the public agent API results in exactly one
        call reaching the raw provider caller, and it went through the
        gate's reservation accounting (remaining budget actually dropped)."""
        raw_caller = _CountingRawCaller()
        gate = BudgetGate(llm_client=_make_client(raw_caller), total_budget=10_000)
        agent = FundamentalsAgent(budget_gate=gate)

        before_remaining = gate.remaining
        await agent.analyze(
            request=ThesisRequest(thesis="Test thesis"),
            round=1,
            token_budget=2000,
            prior_round_outputs=None,
        )

        assert raw_caller.call_count == 1
        # Surplus (max_tokens reserved minus actual tokens_used) is credited
        # back, but the reservation still happened — remaining must have
        # moved, not stayed frozen at the starting value.
        assert gate.remaining != before_remaining
        assert gate.remaining <= before_remaining


class TestOverBudgetRequestNeverCallsTheAPI:
    async def test_gate_refuses_before_any_network_call_when_over_budget(self):
        raw_caller = _CountingRawCaller()
        gate = BudgetGate(llm_client=_make_client(raw_caller), total_budget=100)

        with pytest.raises(BudgetExhaustedError):
            await gate.call(
                system_prompt="sys",
                user_prompt="user",
                response_model=None,  # never reached if the gate refuses correctly
                max_tokens=500,
            )

        assert raw_caller.call_count == 0

    async def test_gate_refuses_second_call_once_first_call_exhausts_budget(self):
        raw_caller = _CountingRawCaller()
        gate = BudgetGate(llm_client=_make_client(raw_caller), total_budget=600)

        from pydantic import BaseModel

        class _Schema(BaseModel):
            stance: str
            confidence: int
            key_factors: list[str]
            evidence: list[str]
            top_risk: str

        await gate.call(
            system_prompt="sys", user_prompt="user", response_model=_Schema, max_tokens=500
        )
        assert raw_caller.call_count == 1

        with pytest.raises(BudgetExhaustedError):
            await gate.call(
                system_prompt="sys", user_prompt="user", response_model=_Schema, max_tokens=500
            )
        # The second, refused call must not have reached the provider.
        assert raw_caller.call_count == 1

    async def test_gate_requires_an_explicit_max_tokens(self):
        """max_tokens is a required positional/keyword parameter on
        BudgetGate.call — there is no None-means-uncapped escape hatch the
        way the old raw LLMClient.call allowed."""
        import inspect

        signature = inspect.signature(BudgetGate.call)
        assert signature.parameters["max_tokens"].default is inspect._empty

    async def test_failed_call_releases_its_reservation(self):
        """If the underlying call raises, the reserved tokens must be
        released back — otherwise a string of transient failures would
        permanently burn budget without ever reaching the API."""

        class _AlwaysRaisingRawCaller:
            call_count = 0

            async def __call__(self, *args, **kwargs):
                self.call_count += 1
                raise ConnectionError("simulated transport failure")

        raw_caller = _AlwaysRaisingRawCaller()
        gate = BudgetGate(llm_client=_make_client(raw_caller), total_budget=500)

        from pydantic import BaseModel

        class _Schema(BaseModel):
            stance: str

        with pytest.raises(ConnectionError):
            await gate.call(
                system_prompt="sys", user_prompt="user", response_model=_Schema, max_tokens=500
            )

        assert gate.remaining == 500


class TestFallbackProviderUsesTheSameGate:
    """See tests/test_llm_fallback.py for the retry/backoff/fallback
    decision logic itself (owned by LLMClient). This class checks the
    narrower BudgetGate-level guarantee the task brief calls out
    explicitly: whichever provider ends up serving a call, it went through
    exactly one reservation and one deduction on this gate — never a
    second, looser path."""

    async def test_gate_reservation_and_deduction_identical_regardless_of_provider(self):
        import httpx
        from anthropic import APIStatusError

        class _FallbackRawCaller:
            def __init__(self):
                self.call_count = 0

            async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens=None):
                self.call_count += 1
                return (
                    {
                        "stance": "Buy",
                        "confidence": 65,
                        "key_factors": ["growth"],
                        "evidence": ["Q3 revenue up 22% YoY"],
                        "top_risk": "x",
                        "executive_summary": "test summary",
                    },
                    300,
                )

        class _AlwaysOverloadedRawCaller:
            def __init__(self):
                self.call_count = 0

            async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens=None):
                self.call_count += 1
                request = httpx.Request("POST", "https://example.invalid/v1/messages")
                response = httpx.Response(503, request=request, json={"error": {"message": "overloaded"}})
                raise APIStatusError("overloaded", response=response, body=None)

        primary = _AlwaysOverloadedRawCaller()
        fallback = _FallbackRawCaller()

        client = LLMClient.__new__(LLMClient)
        client.provider = "primary-fake"
        client.model = "primary-model"
        client.api_key = "fake-key"
        client.timeout_seconds = 60
        client.max_retries = 1
        client.retry_backoff_seconds = 0
        client.fallback_provider = "fallback-fake"
        client.fallback_model = "fallback-model"
        client.fallback_max_retries = 1
        client._raw_caller = primary
        client._fallback_raw_caller = fallback

        gate = BudgetGate(llm_client=client, total_budget=1_000)

        from pydantic import BaseModel

        class _Schema(BaseModel):
            stance: str
            confidence: int
            key_factors: list[str]
            evidence: list[str]
            top_risk: str

        before_remaining = gate.remaining
        _result, tokens_used, provider_used = await gate.call(
            system_prompt="sys", user_prompt="user", response_model=_Schema, max_tokens=500
        )

        assert provider_used == "fallback-fake"
        assert tokens_used == 300
        # Exactly one reservation/deduction cycle happened on this gate — the
        # remaining budget dropped by exactly the fallback's real usage
        # (500 reserved, 300 used, 200 credited back), never twice and never
        # skipped because the serving provider changed mid-call.
        assert before_remaining - gate.remaining == 300
        assert fallback.call_count == 1


class TestOverrunAccounting:
    """Two real accounting bugs found via a live debate against qwen3.5:9b
    on Ollama, fixed in the same session as structured_output.py's
    retry-budget cap: (1) BudgetGate.call() released the *full* reservation
    on any exception, including LLMValidationError, silently crediting
    back budget for tokens genuinely spent across failed attempts; (2)
    `remaining` was only ever decremented by the *reserved* max_tokens,
    never by real usage beyond it, so a call that overran its own
    reservation (the documented prompt-size asymmetry, or the new bounded
    retry overrun) left `remaining` overstating what was actually left."""

    async def test_failed_call_releases_only_the_unused_portion_of_the_reservation(self):
        from pydantic import BaseModel

        class _Schema(BaseModel):
            value: str

        class _PartiallySpendingThenFailingRawCaller:
            def __init__(self):
                self.call_count = 0

            async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens=None):
                self.call_count += 1
                return {"wrong_field": "always invalid"}, 40

        raw_caller = _PartiallySpendingThenFailingRawCaller()
        client = LLMClient.__new__(LLMClient)
        client.provider = "fake"
        client.model = "fake-model"
        client.api_key = "fake-key"
        client.timeout_seconds = 60
        client.max_retries = 3
        client.retry_backoff_seconds = 0
        client.fallback_provider = None
        client.fallback_model = None
        client.fallback_max_retries = 0
        client._fallback_raw_caller = None
        client._raw_caller = raw_caller

        gate = BudgetGate(llm_client=client, total_budget=1000)
        before_remaining = gate.remaining

        from committee.llm.structured_output import LLMValidationError

        with pytest.raises(LLMValidationError) as exc_info:
            await gate.call(
                system_prompt="sys", user_prompt="user", response_model=_Schema, max_tokens=500
            )

        # 3 attempts x 40 tokens each = 120 genuinely spent — must be
        # debited from remaining, not silently credited back as if the
        # whole 500 reservation had gone unused.
        assert exc_info.value.total_tokens_used == 120
        assert before_remaining - gate.remaining == 120

    async def test_overrun_beyond_reservation_is_debited_not_silently_absorbed(self):
        from pydantic import BaseModel

        class _Schema(BaseModel):
            stance: str
            confidence: int
            key_factors: list[str]
            evidence: list[str]
            top_risk: str
            executive_summary: str

        class _OverrunRawCaller:
            async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens=None):
                return (
                    {
                        "stance": "Buy",
                        "confidence": 50,
                        "key_factors": ["x"],
                        "evidence": ["y"],
                        "top_risk": "z",
                        "executive_summary": "s",
                    },
                    900,  # far more than the 500 reserved below
                )

        client = LLMClient.__new__(LLMClient)
        client.provider = "fake"
        client.model = "fake-model"
        client.api_key = "fake-key"
        client.timeout_seconds = 60
        client.max_retries = 1
        client.retry_backoff_seconds = 0
        client.fallback_provider = None
        client.fallback_model = None
        client.fallback_max_retries = 0
        client._fallback_raw_caller = None
        client._raw_caller = _OverrunRawCaller()

        gate = BudgetGate(llm_client=client, total_budget=1000)
        before_remaining = gate.remaining

        _result, tokens_used, _provider = await gate.call(
            system_prompt="sys", user_prompt="user", response_model=_Schema, max_tokens=500
        )

        assert tokens_used == 900
        # remaining must reflect the real 900 spent, not just the 500
        # reserved — this is the accounting bug: before the fix, remaining
        # would have stayed at before_remaining - 500 (the surplus branch
        # never fires since tokens_used > max_tokens, and nothing else
        # touched remaining for the excess).
        assert before_remaining - gate.remaining == 900

    async def test_a_later_call_is_correctly_refused_after_an_earlier_overrun(self):
        """The real-world consequence of the accounting bug: without
        debiting the overrun, a later call could be admitted past what was
        genuinely left, since `remaining` overstated the true balance."""
        from pydantic import BaseModel

        class _Schema(BaseModel):
            value: str

        class _OverrunRawCaller:
            async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens=None):
                return {"value": "ok"}, 900

        client = LLMClient.__new__(LLMClient)
        client.provider = "fake"
        client.model = "fake-model"
        client.api_key = "fake-key"
        client.timeout_seconds = 60
        client.max_retries = 1
        client.retry_backoff_seconds = 0
        client.fallback_provider = None
        client.fallback_model = None
        client.fallback_max_retries = 0
        client._fallback_raw_caller = None
        client._raw_caller = _OverrunRawCaller()

        gate = BudgetGate(llm_client=client, total_budget=1000)

        # First call: reserves 500, actually spends 900 -> remaining should
        # drop to 100 (1000 - 900), not 500 (1000 - 500 reserved).
        await gate.call(system_prompt="sys", user_prompt="user", response_model=_Schema, max_tokens=500)
        assert gate.remaining == 100

        # A second call requesting more than what's genuinely left (100)
        # must be refused — this would have wrongly succeeded pre-fix,
        # since remaining would still have shown 500.
        with pytest.raises(BudgetExhaustedError):
            await gate.call(
                system_prompt="sys", user_prompt="user", response_model=_Schema, max_tokens=200
            )
