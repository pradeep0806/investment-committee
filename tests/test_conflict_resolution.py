import pytest

from committee.models.agent_output import AgentOutput, Stance
from committee.models.synthesis import DisagreementRecord
from committee.orchestration.budget_manager import BudgetExhaustedError
from committee.orchestration.conflict_resolution.confidence_weighted import ConfidenceWeightedStrategy
from committee.orchestration.conflict_resolution.flag_unresolved import FlagUnresolvedStrategy
from committee.orchestration.conflict_resolution.registry import build_strategy
from committee.orchestration.conflict_resolution.tie_breaker import TieBreakerStrategy


def _output(agent_id: str, stance: Stance, confidence: int, key_factors: list[str], round: int = 3) -> AgentOutput:
    return AgentOutput(
        agent_id=agent_id,
        round=round,
        stance=stance,
        confidence=confidence,
        key_factors=key_factors,
        top_risk="some risk",
        tokens_used=500,
    )


def _fixed_disagreement() -> tuple[list[AgentOutput], DisagreementRecord]:
    """A single fixed disagreement fixture reused across strategies to prove
    the same input produces genuinely different synthesis output."""
    outputs = [
        _output("fundamentals", Stance.BUY, 85, ["valuation", "growth"]),
        _output("market_sentiment", Stance.BUY, 60, ["momentum"]),
        _output("risk_contrarian", Stance.SELL, 80, ["valuation", "customer concentration"]),
        _output("macro_context", Stance.HOLD, 55, ["macro"]),
    ]
    disagreement = DisagreementRecord(
        round_detected=3,
        agents_involved=["fundamentals", "risk_contrarian"],
        opposing_stances={"fundamentals": Stance.BUY, "risk_contrarian": Stance.SELL},
        contested_factors=["valuation"],
    )
    return outputs, disagreement


async def test_flag_unresolved_picks_no_winner():
    outputs, disagreement = _fixed_disagreement()
    strategy = FlagUnresolvedStrategy()

    memo, resolved = await strategy.resolve(
        disagreement=disagreement, final_round_outputs=outputs, remaining_budget=0
    )

    assert memo.recommendation == Stance.PASS
    assert memo.supporting_agents == []
    assert set(memo.dissenting_agents) == {"fundamentals", "risk_contrarian"}
    assert memo.dissent_appendix is not None
    assert "did not reach consensus" in memo.dissent_appendix
    assert resolved.resolved is False
    assert resolved.resolution_strategy_applied == "flag_unresolved"


async def test_confidence_weighted_picks_higher_confidence_side():
    outputs, disagreement = _fixed_disagreement()
    strategy = ConfidenceWeightedStrategy()

    memo, resolved = await strategy.resolve(
        disagreement=disagreement, final_round_outputs=outputs, remaining_budget=0
    )

    # fundamentals (Buy, 85) outweighs risk_contrarian (Sell, 80)
    assert memo.recommendation == Stance.BUY
    assert "fundamentals" in memo.supporting_agents
    assert "risk_contrarian" in memo.dissenting_agents
    assert memo.dissent_appendix is not None
    assert resolved.resolved is True
    assert resolved.resolution_strategy_applied == "confidence_weighted"


async def test_confidence_weighted_flips_when_confidences_flip():
    """Same structure, opposite confidence balance -> opposite winner. Proves
    the strategy genuinely weighs confidence rather than any fixed rule like
    "first agent wins" or "alphabetical"."""
    outputs = [
        _output("fundamentals", Stance.BUY, 50, ["valuation"]),
        _output("risk_contrarian", Stance.SELL, 90, ["valuation"]),
    ]
    disagreement = DisagreementRecord(
        round_detected=3,
        agents_involved=["fundamentals", "risk_contrarian"],
        opposing_stances={"fundamentals": Stance.BUY, "risk_contrarian": Stance.SELL},
        contested_factors=["valuation"],
    )
    strategy = ConfidenceWeightedStrategy()

    memo, _ = await strategy.resolve(
        disagreement=disagreement, final_round_outputs=outputs, remaining_budget=0
    )

    assert memo.recommendation == Stance.SELL


async def test_same_disagreement_different_strategy_different_synthesis():
    """The explicit CLAUDE.md §9 requirement: same disagreement, different
    strategy -> different synthesis."""
    outputs, disagreement = _fixed_disagreement()

    flag_memo, _ = await FlagUnresolvedStrategy().resolve(
        disagreement=disagreement, final_round_outputs=outputs, remaining_budget=0
    )
    weighted_memo, _ = await ConfidenceWeightedStrategy().resolve(
        disagreement=disagreement, final_round_outputs=outputs, remaining_budget=0
    )

    assert flag_memo.recommendation != weighted_memo.recommendation
    assert flag_memo.recommendation == Stance.PASS
    assert weighted_memo.recommendation == Stance.BUY


async def test_tie_breaker_verdict_is_dispositive_by_construction():
    outputs, disagreement = _fixed_disagreement()
    strategy = TieBreakerStrategy()

    async def fake_spawn(opposing_outputs, contested_factors):
        return _output("tie_breaker", Stance.SELL, 90, ["valuation"], round=3)

    memo, resolved = await strategy.resolve(
        disagreement=disagreement,
        final_round_outputs=outputs,
        remaining_budget=1000,
        spawn_agent_fn=fake_spawn,
    )

    assert memo.recommendation == Stance.SELL
    assert "tie_breaker" in memo.supporting_agents
    assert "fundamentals" in memo.dissenting_agents
    assert resolved.resolved is True
    assert resolved.resolution_strategy_applied == "tie_breaker"


async def test_tie_breaker_falls_back_to_flag_unresolved_when_reserve_cant_cover_actual_spend():
    """Real bug found via live testing: max_tokens caps the tie-breaker's
    output only — its actual cost (prompt + output) can still exceed the
    reserve pool by roughly the size of its own prompt (the two opposing
    AgentOutputs + contested factors), the same asymmetry already accepted
    for ordinary agents. There the fix is "exclude one agent"; for the
    tie-breaker specifically, draw_from_reserve raising BudgetExhaustedError
    used to propagate uncaught and crash the *entire debate* over a
    single-digit-percent overrun. It must instead fall back to
    flag_unresolved's memo shape (explicit unresolved split, no averaging)
    so one unaffordable tie-breaker never loses a whole completed debate."""
    outputs, disagreement = _fixed_disagreement()
    strategy = TieBreakerStrategy()

    async def fake_spawn_that_overruns_reserve(opposing_outputs, contested_factors):
        raise BudgetExhaustedError("Tie-breaker spend 3187 exceeds remaining reserve 3000.")

    memo, resolved = await strategy.resolve(
        disagreement=disagreement,
        final_round_outputs=outputs,
        remaining_budget=3000,
        spawn_agent_fn=fake_spawn_that_overruns_reserve,
    )

    assert memo.recommendation == Stance.PASS
    assert memo.supporting_agents == []
    assert set(memo.dissenting_agents) == {"fundamentals", "risk_contrarian"}
    assert resolved.resolved is False
    assert resolved.resolution_strategy_applied == "flag_unresolved"


async def test_tie_breaker_requires_spawn_agent_fn():
    outputs, disagreement = _fixed_disagreement()
    strategy = TieBreakerStrategy()

    with pytest.raises(ValueError):
        await strategy.resolve(disagreement=disagreement, final_round_outputs=outputs, remaining_budget=1000)


def test_registry_builds_all_three_strategies_by_id():
    assert build_strategy("flag_unresolved").strategy_id == "flag_unresolved"
    assert build_strategy("confidence_weighted").strategy_id == "confidence_weighted"
    assert build_strategy("tie_breaker").strategy_id == "tie_breaker"
