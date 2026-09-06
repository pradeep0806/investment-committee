"""Default conflict resolution strategy: state the split explicitly, pick no
winner. Never averages the opposing stances into a blended recommendation
(CLAUDE.md §5 step 5)."""

from __future__ import annotations

from committee.models.agent_output import AgentOutput, Stance
from committee.models.synthesis import DisagreementRecord, SynthesisMemo
from committee.orchestration.conflict_resolution.base import SpawnAgentFn
from committee.orchestration.conflict_resolution.registry import register


@register("flag_unresolved")
class FlagUnresolvedStrategy:
    strategy_id = "flag_unresolved"

    async def resolve(
        self,
        disagreement: DisagreementRecord,
        final_round_outputs: list[AgentOutput],
        remaining_budget: int,
        spawn_agent_fn: SpawnAgentFn | None = None,
    ) -> tuple[SynthesisMemo, DisagreementRecord]:
        involved = {output.agent_id for output in final_round_outputs if output.agent_id in disagreement.opposing_stances}
        stance_summary = ", ".join(
            f"{agent_id}={stance.value}" for agent_id, stance in disagreement.opposing_stances.items()
        )

        memo = SynthesisMemo(
            recommendation=Stance.PASS,
            confidence=50,
            supporting_agents=[],
            dissenting_agents=sorted(involved),
            dissent_appendix=(
                f"The committee did not reach consensus. Opposing positions: {stance_summary}. "
                f"Contested factors: {', '.join(disagreement.contested_factors) or 'none identified'}. "
                "No single recommendation is issued; both positions are presented as-is."
            ),
            reasoning_trace_refs=[
                f"round{output.round}:{output.agent_id}" for output in final_round_outputs
            ],
        )

        resolved_disagreement = disagreement.model_copy(
            update={"resolution_strategy_applied": self.strategy_id, "resolved": False}
        )
        return memo, resolved_disagreement
