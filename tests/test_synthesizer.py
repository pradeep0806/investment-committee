import pytest

from committee.models.agent_output import AgentOutput, Stance
from committee.orchestration.conflict_resolution.flag_unresolved import FlagUnresolvedStrategy
from committee.synthesis.synthesizer import synthesize

pytestmark = pytest.mark.asyncio


def _output(agent_id: str, stance: Stance, confidence: int, executive_summary: str = "") -> AgentOutput:
    return AgentOutput(
        agent_id=agent_id,
        round=2,
        stance=stance,
        confidence=confidence,
        key_factors=["factor"],
        top_risk="risk",
        executive_summary=executive_summary,
        tokens_used=100,
    )


async def test_synthesize_produces_clean_consensus_when_all_agents_agree():
    outputs = [
        _output("fundamentals", Stance.BUY, 70, executive_summary="Revenue growth is strong."),
        _output("market_sentiment", Stance.BUY, 80, executive_summary="Momentum is building."),
    ]

    memo, resolved = await synthesize(
        final_round_outputs=outputs,
        final_round_disagreements=[],
        strategy=FlagUnresolvedStrategy(),
    )

    assert resolved == []
    assert memo.recommendation == Stance.BUY
    assert memo.confidence == 75
    assert memo.supporting_agents == ["fundamentals", "market_sentiment"]
    assert memo.dissent_appendix is None
    assert memo.agent_summaries == {
        "fundamentals": "Revenue growth is strong.",
        "market_sentiment": "Momentum is building.",
    }


async def test_synthesize_returns_pass_memo_when_final_round_has_no_agent_outputs():
    """Real bug found via live testing against a weak local model: every
    agent can legitimately be excluded from the final round (structured-
    output retries exhausted for all of them), leaving final_round_outputs
    empty. This must produce a degraded-but-valid PASS memo, not crash with
    an IndexError from Counter(...).most_common(1)[0] on an empty sequence."""
    memo, resolved = await synthesize(
        final_round_outputs=[],
        final_round_disagreements=[],
        strategy=FlagUnresolvedStrategy(),
    )

    assert resolved == []
    assert memo.recommendation == Stance.PASS
    assert memo.confidence == 0
    assert memo.supporting_agents == []
    assert memo.dissenting_agents == []
    assert memo.reasoning_trace_refs == []
    assert memo.dissent_appendix is not None
    assert "could not reach a recommendation" in memo.dissent_appendix.lower()
