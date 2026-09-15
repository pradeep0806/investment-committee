"""End-to-end: a debate including one custom (persona-backed) agent runs
alongside the built-in four and produces a normal structured trace — proves
DynamicAnalystAgent is a fully interchangeable AnalystAgent from the
orchestrator's point of view, with no special-casing required."""

from committee.agents.registry import build_agents
from committee.llm.client import LLMClient
from committee.models.persona import AgentPersona
from committee.models.requests import DebateConfig, ThesisRequest
from committee.observability.logging import configure_logging
from committee.orchestration.budget_gate import BudgetGate
from committee.orchestration.orchestrator import DebateOrchestrator

configure_logging()


class _FakeRawCaller:
    def __init__(self):
        self.call_count = 0

    async def __call__(self, system_prompt, user_prompt, schema, retry_note, max_tokens=None):
        self.call_count += 1
        return (
            {
                "stance": "Buy",
                "confidence": 65,
                "key_factors": ["revenue growth", "margin expansion"],
                "evidence": ["Q3 revenue up 22% YoY", "gross margin expanded 3pts"],
                "top_risk": "customer concentration",
                "executive_summary": "test summary",
            },
            500,
        )


def _make_gate(raw_caller=None, total_budget: int = 1_000_000) -> BudgetGate:
    client = LLMClient.__new__(LLMClient)
    client.provider = "fake"
    client.model = "fake-model"
    client.api_key = "fake-key"
    client.timeout_seconds = 60
    client.max_retries = 3
    client._raw_caller = raw_caller if raw_caller is not None else _FakeRawCaller()
    return BudgetGate(llm_client=client, total_budget=total_budget)


def _custom_persona() -> AgentPersona:
    return AgentPersona(
        name="ESG Screener",
        role="ESG/Sustainability analyst",
        responsibility="Assess environmental, social, and governance exposure and controversy risk.",
        thinking_style="Weighs long-horizon reputational and regulatory risk over near-term multiples.",
        priorities=["Governance red flags", "Regulatory/ESG controversy exposure"],
        blind_spots=["Ignores short-term price action entirely"],
    )


async def test_debate_with_one_custom_persona_produces_normal_structured_trace():
    gate = _make_gate()
    persona = _custom_persona()
    agents = build_agents(budget_gate=gate, custom_personas=[persona])
    assert len(agents) == 5  # core 4 + 1 custom

    config = DebateConfig(total_token_budget=12_000, num_rounds=2)
    orchestrator = DebateOrchestrator(config=config, agents=agents)

    trace = await orchestrator.run(ThesisRequest(thesis="Test thesis with a custom persona"))

    assert len(trace.rounds) == 2
    assert all(len(round_record.agent_outputs) == 5 for round_record in trace.rounds)

    round1_ids = {output.agent_id for output in trace.rounds[0].agent_outputs}
    assert round1_ids == {
        "fundamentals",
        "market_sentiment",
        "risk_contrarian",
        "macro_context",
        persona.id,
    }

    custom_outputs = [o for o in trace.all_agent_outputs if o.agent_id == persona.id]
    assert custom_outputs
    assert all(o.agent_name == persona.name for o in custom_outputs)
    assert all(o.stance is not None for o in custom_outputs)

    # Budget ledger entries exist for the custom agent too — dynamic
    # allocation treats it exactly like a built-in.
    ledger_agent_ids = {entry.agent_id for entry in trace.budget_ledger}
    assert persona.id in ledger_agent_ids
