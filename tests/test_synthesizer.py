import pytest

from committee.models.agent_output import AgentOutput, Stance
from committee.models.synthesis import DisagreementRecord, DissentEntry
from committee.orchestration.conflict_resolution.confidence_weighted import (
    ConfidenceWeightedStrategy,
)
from committee.orchestration.conflict_resolution.flag_unresolved import (
    FlagUnresolvedStrategy,
)
from committee.synthesis.synthesizer import synthesize

pytestmark = pytest.mark.asyncio


def _output(
    agent_id: str,
    stance: Stance,
    confidence: int,
    executive_summary: str = "",
    top_risk: str = "risk",
    agent_name: str | None = None,
) -> AgentOutput:
    return AgentOutput(
        agent_id=agent_id,
        agent_name=agent_name,
        round=2,
        stance=stance,
        confidence=confidence,
        key_factors=["factor"],
        top_risk=top_risk,
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
    assert memo.dissenting_view == []
    assert memo.dissenting_view_note == "No dissent — all agents converged on Buy."


async def test_synthesize_clean_consensus_still_names_a_minority_dissenter():
    """Counter.most_common(1) only guarantees a plurality, not unanimity — a
    minority agent can still differ from the majority stance even on the
    'clean consensus' (no DisagreementRecord raised) path. That agent must
    still be named in dissenting_view, not silently dropped."""
    outputs = [
        _output("fundamentals", Stance.BUY, 70, executive_summary="Revenue growth is strong."),
        _output("market_sentiment", Stance.BUY, 80, executive_summary="Momentum is building."),
        _output(
            "risk_contrarian",
            Stance.SELL,
            60,
            executive_summary="Downside risk is underpriced.",
            agent_name="Risk Contrarian",
        ),
    ]

    memo, resolved = await synthesize(
        final_round_outputs=outputs,
        final_round_disagreements=[],
        strategy=FlagUnresolvedStrategy(),
    )

    assert resolved == []
    assert memo.recommendation == Stance.BUY
    assert memo.dissenting_agents == ["risk_contrarian"]
    assert memo.dissenting_view == [
        DissentEntry(
            agent_id="risk_contrarian",
            agent_name="Risk Contrarian",
            stance=Stance.SELL,
            reason="Downside risk is underpriced.",
        )
    ]
    assert "risk_contrarian (Sell)" in memo.dissenting_view_note


async def test_synthesize_dissenting_view_falls_back_to_top_risk_without_executive_summary():
    outputs = [
        _output("fundamentals", Stance.BUY, 70),
        _output("risk_contrarian", Stance.SELL, 60, top_risk="valuation is stretched"),
    ]

    memo, _ = await synthesize(
        final_round_outputs=outputs,
        final_round_disagreements=[],
        strategy=FlagUnresolvedStrategy(),
    )

    assert memo.dissenting_view[0].reason == "valuation is stretched"


async def test_synthesize_disagreement_path_populates_dissenting_view_from_losing_side():
    """confidence_weighted resolves the disagreement (picks a winner), but
    the synthesizer must still separately name the losing side in
    dissenting_view — the Core requirement that disagreement is surfaced
    explicitly, never blended away, regardless of which strategy ran."""
    outputs = [
        _output(
            "fundamentals", Stance.BUY, 85, executive_summary="Strong fundamentals justify a Buy."
        ),
        _output(
            "risk_contrarian",
            Stance.SELL,
            60,
            executive_summary="Tail risk is being ignored.",
            agent_name="Risk Contrarian",
        ),
    ]
    disagreement = DisagreementRecord(
        round_detected=2,
        agents_involved=["fundamentals", "risk_contrarian"],
        opposing_stances={"fundamentals": Stance.BUY, "risk_contrarian": Stance.SELL},
        contested_factors=["valuation"],
    )

    memo, resolved = await synthesize(
        final_round_outputs=outputs,
        final_round_disagreements=[disagreement],
        strategy=ConfidenceWeightedStrategy(),
    )

    assert memo.recommendation == Stance.BUY
    assert memo.dissenting_view == [
        DissentEntry(
            agent_id="risk_contrarian",
            agent_name="Risk Contrarian",
            stance=Stance.SELL,
            reason="Tail risk is being ignored.",
        )
    ]
    assert memo.dissenting_view_note != ""
    assert "risk_contrarian" in memo.dissenting_view_note
    assert resolved[0].resolution_strategy_applied == "confidence_weighted"


async def test_synthesize_never_makes_an_additional_llm_call():
    """dissenting_view must be pure aggregation over final_round_outputs
    already produced by agents earlier in the debate. synthesize()'s own
    signature takes no LLM client/budget gate at all — proven here by
    spawn_agent_fn (the *only* parameter through which synthesize() can ever
    reach an LLM, used exclusively by the tie_breaker strategy) being passed
    as a function that fails the test if invoked. Since this test's strategy
    never raises a disagreement, that spawn path is provably never reached,
    and no other parameter offers an LLM entry point at all."""

    async def _fail_if_called(*args, **kwargs):
        raise AssertionError("synthesize() must never invoke spawn_agent_fn without a disagreement")

    outputs = [
        _output("fundamentals", Stance.BUY, 85, executive_summary="Strong fundamentals."),
        _output("risk_contrarian", Stance.SELL, 60, executive_summary="Tail risk is high."),
    ]

    memo, _ = await synthesize(
        final_round_outputs=outputs,
        final_round_disagreements=[],
        strategy=FlagUnresolvedStrategy(),
        spawn_agent_fn=_fail_if_called,
    )

    assert len(memo.dissenting_view) == 1


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
