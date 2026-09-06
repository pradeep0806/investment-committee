"""tie_breaker (stretch, CLAUDE.md §5 step 5): spawns one additional agent
with the two opposing arguments as context, extra budget drawn from the
BudgetManager's reserve pool.

Resolved design decisions (confirmed with the project owner during planning):
- Context payload is the opposing AgentOutputs + their contested factors only
  — not the full multi-round transcript — to keep the spawned agent's prompt
  and cost bounded.
- The tie-breaker's verdict is dispositive by construction: whichever side it
  agrees with simply wins. It cannot itself cascade into a second
  disagreement, bounding worst-case runtime/budget.
"""

from __future__ import annotations

import structlog

from committee.models.agent_output import AgentOutput
from committee.models.synthesis import DisagreementRecord, SynthesisMemo
from committee.orchestration.budget_manager import BudgetExhaustedError
from committee.orchestration.conflict_resolution.base import SpawnAgentFn
from committee.orchestration.conflict_resolution.flag_unresolved import FlagUnresolvedStrategy
from committee.orchestration.conflict_resolution.registry import register

logger = structlog.get_logger()


@register("tie_breaker")
class TieBreakerStrategy:
    strategy_id = "tie_breaker"

    async def resolve(
        self,
        disagreement: DisagreementRecord,
        final_round_outputs: list[AgentOutput],
        remaining_budget: int,
        spawn_agent_fn: SpawnAgentFn | None = None,
    ) -> tuple[SynthesisMemo, DisagreementRecord]:
        if spawn_agent_fn is None:
            raise ValueError("tie_breaker strategy requires a spawn_agent_fn")

        involved_outputs = [
            output for output in final_round_outputs if output.agent_id in disagreement.opposing_stances
        ]

        try:
            tie_breaker_output = await spawn_agent_fn(involved_outputs, disagreement.contested_factors)
        except BudgetExhaustedError as exc:
            # Real bug found via live testing: max_tokens caps the
            # tie-breaker's *output* only — its actual cost
            # (spawn_agent_fn's tokens_used, i.e. prompt+output) can still
            # exceed the reserve by roughly the size of its own prompt
            # (the two opposing AgentOutputs + contested factors), the same
            # documented asymmetry accepted for ordinary agents. But there
            # the fix is "exclude one agent"; here it used to crash the
            # *entire debate* with an unhandled BudgetExhaustedError over a
            # single-digit-percent overrun — wildly disproportionate, and
            # the tie-breaker call's real tokens/latency were already spent
            # by the time the reserve check fires (draw_from_reserve is
            # necessarily post-hoc, like the budget check everywhere else
            # in this system). Falling back to flag_unresolved's memo
            # shape still produces a spec-correct synthesis (explicit
            # unresolved split, no averaging) instead of losing the whole
            # debate over an unaffordable tie-breaker.
            logger.warning(
                "tie_breaker_unaffordable_falling_back_to_flag_unresolved",
                error=str(exc),
            )
            return await FlagUnresolvedStrategy().resolve(
                disagreement=disagreement,
                final_round_outputs=final_round_outputs,
                remaining_budget=remaining_budget,
                spawn_agent_fn=None,
            )

        # Dispositive by construction: the tie-breaker's own stance settles it.
        winning_stance = tie_breaker_output.stance
        winners = [o for o in involved_outputs if o.stance == winning_stance]
        losers = [o for o in involved_outputs if o.stance != winning_stance]

        dissent_appendix = None
        if losers:
            dissent_summary = "; ".join(
                f"{o.agent_id} ({o.stance.value}, confidence={o.confidence}): {o.top_risk}"
                for o in losers
            )
            dissent_appendix = (
                f"A tie-breaker agent was spawned to resolve a contested split and sided with "
                f"{winning_stance.value} (confidence {tie_breaker_output.confidence}). "
                f"Dissenting view(s) retained: {dissent_summary}."
            )

        memo = SynthesisMemo(
            recommendation=winning_stance,
            confidence=tie_breaker_output.confidence,
            supporting_agents=sorted([o.agent_id for o in winners] + ["tie_breaker"]),
            dissenting_agents=sorted(o.agent_id for o in losers),
            dissent_appendix=dissent_appendix,
            reasoning_trace_refs=[
                f"round{output.round}:{output.agent_id}" for output in final_round_outputs
            ]
            + ["tie_breaker:verdict"],
        )

        resolved_disagreement = disagreement.model_copy(
            update={"resolution_strategy_applied": self.strategy_id, "resolved": True}
        )
        return memo, resolved_disagreement
