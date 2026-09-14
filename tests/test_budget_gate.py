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
