"""confidence_weighted: majority wins, weighted by stated confidence — but the
losing side's reasoning still appears as a dissent appendix, never silently
dropped (CLAUDE.md §5 step 5)."""

from __future__ import annotations

from collections import defaultdict

from committee.models.agent_output import AgentOutput
from committee.models.synthesis import DisagreementRecord, SynthesisMemo
from committee.orchestration.conflict_resolution.base import SpawnAgentFn
from committee.orchestration.conflict_resolution.registry import register


@register("confidence_weighted")
class ConfidenceWeightedStrategy:
    strategy_id = "confidence_weighted"

    async def resolve(
        self,
        disagreement: DisagreementRecord,
        final_round_outputs: list[AgentOutput],
        remaining_budget: int,
        spawn_agent_fn: SpawnAgentFn | None = None,
    ) -> tuple[SynthesisMemo, DisagreementRecord]:
        involved_outputs = [
            output for output in final_round_outputs if output.agent_id in disagreement.opposing_stances
        ]

        confidence_by_stance: dict = defaultdict(int)
        for output in involved_outputs:
            confidence_by_stance[output.stance] += output.confidence

        winning_stance = max(confidence_by_stance, key=confidence_by_stance.get)
        winners = [o for o in involved_outputs if o.stance == winning_stance]
        losers = [o for o in involved_outputs if o.stance != winning_stance]

        total_confidence = sum(confidence_by_stance.values())
        winning_confidence = confidence_by_stance[winning_stance]
        aggregate_confidence = (
            round(100 * winning_confidence / total_confidence) if total_confidence else 50
        )

        dissent_appendix = None
        if losers:
            dissent_summary = "; ".join(
                f"{o.agent_id} ({o.stance.value}, confidence={o.confidence}): {o.top_risk}"
                for o in losers
            )
            dissent_appendix = (
                f"Resolved via confidence-weighted majority in favor of {winning_stance.value} "
                f"(aggregate confidence {aggregate_confidence}). Dissenting view(s) retained: "
                f"{dissent_summary}."
            )

        memo = SynthesisMemo(
            recommendation=winning_stance,
            confidence=aggregate_confidence,
            supporting_agents=sorted(o.agent_id for o in winners),
            dissenting_agents=sorted(o.agent_id for o in losers),
            dissent_appendix=dissent_appendix,
            reasoning_trace_refs=[
                f"round{output.round}:{output.agent_id}" for output in final_round_outputs
            ],
        )

        resolved_disagreement = disagreement.model_copy(
            update={"resolution_strategy_applied": self.strategy_id, "resolved": True}
        )
        return memo, resolved_disagreement
