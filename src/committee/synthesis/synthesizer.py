"""Synthesizer: produces the final committee memo.

Clean-consensus path (no final-round disagreements): the memo is built
directly from the agreeing agents, no conflict resolution strategy involved,
`dissent_appendix` stays None. Disagreement path: delegates entirely to the
configured ConflictResolutionStrategy (CLAUDE.md §5 step 6) — this module
never branches on which strategy is active.
"""

from __future__ import annotations

from committee.models.agent_output import AgentOutput, Stance
from committee.models.synthesis import DisagreementRecord, SynthesisMemo
from committee.orchestration.conflict_resolution.base import ConflictResolutionStrategy, SpawnAgentFn


async def synthesize(
    final_round_outputs: list[AgentOutput],
    final_round_disagreements: list[DisagreementRecord],
    strategy: ConflictResolutionStrategy,
    remaining_budget: int = 0,
    spawn_agent_fn: SpawnAgentFn | None = None,
) -> tuple[SynthesisMemo, list[DisagreementRecord]]:
    """Returns (synthesis_memo, resolved_disagreement_records).

    If there are no final-round disagreements, produces a clean-consensus
    memo directly. Otherwise delegates each disagreement to `strategy` — in
    practice there is normally at most one disagreement record per final
    round, but the loop handles multiple defensively; the *last* resolved
    memo is what's returned as the top-level synthesis, since a debate
    produces exactly one committee recommendation.
    """
    if not final_round_disagreements:
        return _clean_consensus_memo(final_round_outputs), []

    resolved_records: list[DisagreementRecord] = []
    memo: SynthesisMemo | None = None
    for disagreement in final_round_disagreements:
        memo, resolved = await strategy.resolve(
            disagreement=disagreement,
            final_round_outputs=final_round_outputs,
            remaining_budget=remaining_budget,
            spawn_agent_fn=spawn_agent_fn,
        )
        resolved_records.append(resolved)

    assert memo is not None
    return memo, resolved_records


def _clean_consensus_memo(final_round_outputs: list[AgentOutput]) -> SynthesisMemo:
    from collections import Counter

    if not final_round_outputs:
        # Every agent was excluded from the final round (structured-output
        # retries exhausted for all of them — seen in practice with a weak
        # local model). There's no agent output left to synthesize from, so
        # this is a legitimate "the committee reached no conclusion" result,
        # not a bug to crash on: Stance.PASS with zero confidence and an
        # explanatory dissent_appendix, rather than a raw 500.
        return SynthesisMemo(
            recommendation=Stance.PASS,
            confidence=0,
            supporting_agents=[],
            dissenting_agents=[],
            dissent_appendix=(
                "No agent produced a valid output in the final round (all were "
                "excluded after exhausting structured-output retries) — the "
                "committee could not reach a recommendation."
            ),
            reasoning_trace_refs=[],
        )

    majority_stance, _ = Counter(output.stance for output in final_round_outputs).most_common(1)[0]
    supporting = [output for output in final_round_outputs if output.stance == majority_stance]
    average_confidence = round(sum(o.confidence for o in supporting) / len(supporting))

    return SynthesisMemo(
        recommendation=majority_stance,
        confidence=average_confidence,
        supporting_agents=sorted(o.agent_id for o in supporting),
        dissenting_agents=[],
        dissent_appendix=None,
        reasoning_trace_refs=[
            f"round{output.round}:{output.agent_id}" for output in final_round_outputs
        ],
    )
