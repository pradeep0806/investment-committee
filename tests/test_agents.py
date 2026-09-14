import pytest

from committee.agents.fundamentals import FundamentalsAgent
from committee.agents.registry import build_agents
from committee.llm.client import LLMClient
from committee.llm.structured_output import LLMValidationError
from committee.models.agent_output import Stance
from committee.models.requests import ThesisRequest
from committee.orchestration.budget_gate import BudgetGate


class _FakeRawCaller:
    """Stands in for a provider call: returns canned tool-call arguments without
    touching the network, so agent-mapping logic is tested deterministically."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str, dict, str | None]] = []

    async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens=None):
        self.calls.append((system_prompt, user_prompt, schema, retry_note))
        response = self._responses.pop(0)
        return response, 1234


def _make_client(responses: list[dict], max_retries: int = 3) -> LLMClient:
    client = LLMClient.__new__(LLMClient)
    client.provider = "fake"
    client.model = "fake-model"
    client.api_key = "fake-key"
    client.timeout_seconds = 60
    client.max_retries = max_retries
    client._raw_caller = _FakeRawCaller(responses)
    return client


def _make_gate(responses: list[dict], max_retries: int = 3, total_budget: int = 1_000_000) -> BudgetGate:
    return BudgetGate(llm_client=_make_client(responses, max_retries), total_budget=total_budget)


async def test_fundamentals_agent_maps_valid_response_to_agent_output():
    canned = {
        "stance": "Buy",
        "confidence": 78,
        "key_factors": ["revenue growth", "margin expansion"],
        "evidence": ["Q3 revenue up 22% YoY", "gross margin expanded 3pts"],
        "top_risk": "customer concentration",
    }
    gate = _make_gate([canned])
    agent = FundamentalsAgent(budget_gate=gate)

    output = await agent.analyze(
        request=ThesisRequest(thesis="NovaTech is undervalued given enterprise AI adoption"),
        round=1,
        token_budget=2000,
        prior_round_outputs=None,
    )

    assert output.agent_id == "fundamentals"
    assert output.round == 1
    assert output.stance == Stance.BUY
    assert output.confidence == 78
    assert output.key_factors == ["revenue growth", "margin expansion"]
    assert output.top_risk == "customer concentration"
    assert output.tokens_used == 1234


async def test_fundamentals_agent_retries_on_invalid_response_then_succeeds():
    invalid = {"stance": "Strong Buy", "confidence": 200, "key_factors": [], "top_risk": "x"}
    valid = {
        "stance": "Hold",
        "confidence": 50,
        "key_factors": ["valuation"],
        "evidence": ["EV/EBITDA at 18x vs sector median 12x"],
        "top_risk": "growth deceleration",
    }
    gate = _make_gate([invalid, valid], max_retries=3)
    agent = FundamentalsAgent(budget_gate=gate)

    output = await agent.analyze(
        request=ThesisRequest(thesis="Test thesis"),
        round=1,
        token_budget=2000,
        prior_round_outputs=None,
    )

    assert output.stance == Stance.HOLD
    # tokens accumulate across both the failed and the successful attempt
    assert output.tokens_used == 2468
    raw_caller = gate._llm_client._raw_caller
    assert len(raw_caller.calls) == 2
    assert raw_caller.calls[1][3] is not None  # retry_note populated on 2nd call


async def test_fundamentals_agent_raises_after_exhausting_retries():
    always_invalid = {"stance": "Strong Buy", "confidence": 200, "key_factors": [], "top_risk": "x"}
    gate = _make_gate([always_invalid] * 3, max_retries=3)
    agent = FundamentalsAgent(budget_gate=gate)

    with pytest.raises(LLMValidationError):
        await agent.analyze(
            request=ThesisRequest(thesis="Test thesis"),
            round=1,
            token_budget=2000,
            prior_round_outputs=None,
        )


def test_registry_builds_all_four_default_agents():
    gate = _make_gate([])
    agents = build_agents(budget_gate=gate)
    assert len(agents) == 4
    assert [a.agent_id for a in agents] == [
        "fundamentals",
        "market_sentiment",
        "risk_contrarian",
        "macro_context",
    ]
    assert agents[0].lens_name == "Fundamentals/Valuation"


def test_registry_can_build_a_subset_of_agents_by_role():
    gate = _make_gate([])
    agents = build_agents(budget_gate=gate, agent_roles=["risk_contrarian"])
    assert len(agents) == 1
    assert agents[0].agent_id == "risk_contrarian"
