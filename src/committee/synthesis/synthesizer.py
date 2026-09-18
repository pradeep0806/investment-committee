"""Synthesizer: produces the final committee memo.

Clean-consensus path (no final-round disagreements): the memo is built
directly from the agreeing agents, no conflict resolution strategy involved,
`dissent_appendix` stays None. Disagreement path: delegates entirely to the
configured ConflictResolutionStrategy (CLAUDE.md §5 step 6) — this module
never branches on which strategy is active.
"""

from __future__ import annotations

from committee.models.agent_output import AgentOutput, Stance
from committee.models.synthesis import (
    ConvergenceType,
    DisagreementRecord,
    DissentEntry,
    SynthesisMemo,
)
from committee.orchestration.conflict_resolution.base import (
    ConflictResolutionStrategy,
    SpawnAgentFn,
)


def _build_dissenting_view(
    final_round_outputs: list[AgentOutput], recommendation: Stance, has_real_winner: bool = True
) -> tuple[list[DissentEntry], str]:
    """Diffs each final-round agent's own stance against the committee's
    final `recommendation` — pure aggregation over data already produced,
    no new LLM call. `reason` reuses executive_summary when the agent set
    one, otherwise falls back to top_risk (always populated).

    `has_real_winner` is False exactly when `recommendation` is
    flag_unresolved's placeholder (Stance.PASS with no agent behind it,
    signaled by that strategy's own `resolved=False` — see
    ConflictResolutionStrategy.resolve()) rather than a stance any agent
    actually holds. Real bug found via a live run: diffing every agent's
    stance against a placeholder nobody voted for flagged all of them as
    "dissenting," a technically-true but meaningless result that misrepresents
    a genuine no-consensus outcome (already correctly stated in
    dissent_appendix) as if it were 100% disagreement with a real decision.
    When False, skip the diff entirely — there is no real recommendation to
    dissent from — and return a distinct "no consensus" note instead of
    either the "no dissent" note (a different, opposite situation) or a
    fabricated dissenter list."""
    if not has_real_winner:
        return [], "No majority reached — see dissent appendix for each agent's position."

    dissenters = [output for output in final_round_outputs if output.stance != recommendation]

    if not dissenters:
        return [], f"No dissent — all agents converged on {recommendation.value}."

    view = [
        DissentEntry(
            agent_id=output.agent_id,
            agent_name=output.agent_name,
            stance=output.stance,
            reason=output.executive_summary or output.top_risk,
        )
        for output in dissenters
    ]
    agent_label = "agent" if len(dissenters) == 1 else "agents"
    note = (
        f"{len(dissenters)} {agent_label} dissented from {recommendation.value}: "
        + ", ".join(f"{entry.agent_id} ({entry.stance.value})" for entry in view)
    )
    return view, note


async def synthesize(
    final_round_outputs: list[AgentOutput],
    final_round_disagreements: list[DisagreementRecord],
    strategy: ConflictResolutionStrategy,
    remaining_budget: int = 0,
    spawn_agent_fn: SpawnAgentFn | None = None,
    convergence_types: dict[str, ConvergenceType] | None = None,
) -> tuple[SynthesisMemo, list[DisagreementRecord]]:
    """Returns (synthesis_memo, resolved_disagreement_records).

    If there are no final-round disagreements, produces a clean-consensus
    memo directly. Otherwise delegates each disagreement to `strategy` — in
    practice there is normally at most one disagreement record per final
    round, but the loop handles multiple defensively; the *last* resolved
    memo is what's returned as the top-level synthesis, since a debate
    produces exactly one committee recommendation.

    `convergence_types` (agent_id -> ConvergenceType for the *final* round,
    from ExploreExploitController.score/convergence_classifier.classify_round
    — see orchestrator.py's `final_round.convergence_signal.convergence_types`)
    is what lets the clean-consensus path down-weight an ECHO'd argument
    instead of letting it vote in the majority on equal footing with a
    genuine one. Optional/default-None so every existing caller that doesn't
    have this classification handy keeps working with the prior, unfiltered
    behavior — this is additive, not a required new argument. Only consulted
    on the clean-consensus path: the disagreement/conflict-resolution path
    is out of scope for this echo-down-weighting task (see CLAUDE.md) and is
    untouched.
    """
    agent_summaries = {
        output.agent_id: output.executive_summary
        for output in final_round_outputs
        if output.executive_summary
    }

    if not final_round_disagreements:
        return _clean_consensus_memo(final_round_outputs, convergence_types or {}), []

    resolved_records: list[DisagreementRecord] = []
    memo: SynthesisMemo | None = None
    last_resolved: DisagreementRecord | None = None
    for disagreement in final_round_disagreements:
        memo, last_resolved = await strategy.resolve(
            disagreement=disagreement,
            final_round_outputs=final_round_outputs,
            remaining_budget=remaining_budget,
            spawn_agent_fn=spawn_agent_fn,
        )
        resolved_records.append(last_resolved)

    assert memo is not None
    assert last_resolved is not None
    dissenting_view, dissenting_view_note = _build_dissenting_view(
        final_round_outputs, memo.recommendation, has_real_winner=last_resolved.resolved
    )
    memo = memo.model_copy(
        update={
            "agent_summaries": agent_summaries,
            "dissenting_view": dissenting_view,
            "dissenting_view_note": dissenting_view_note,
        }
    )
    return memo, resolved_records


def _clean_consensus_memo(
    final_round_outputs: list[AgentOutput],
    convergence_types: dict[str, ConvergenceType],
) -> SynthesisMemo:
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
            dissenting_view_note="No dissent — no agent produced a valid final-round output.",
        )

    echoed_agents = sorted(
        output.agent_id
        for output in final_round_outputs
        if convergence_types.get(output.agent_id) == ConvergenceType.ECHO
    )
    # An echoed argument is down-weighted out of the vote entirely — it
    # doesn't count toward majority_stance, supporting_agents, or the
    # average confidence, since it contributes nothing beyond what a prior
    # argument already put on the table (see convergence_classifier.py).
    # Guarded against the degenerate all-echo case: if literally every
    # final-round output is an echo, there is nothing non-echoed left to
    # vote — falling back to the full (unfiltered) set here is what keeps
    # this a "down-weight genuine vs. echo" rule rather than a "the debate
    # produces no recommendation at all" rule, which is a different failure
    # mode this task doesn't ask for and would be a much bigger behavior
    # change for an edge case that's already unlikely (an all-echo final
    # round would also have scored near-zero genuine convergence upstream).
    voting_outputs = [
        output for output in final_round_outputs if output.agent_id not in echoed_agents
    ] or final_round_outputs

    majority_stance, _ = Counter(output.stance for output in voting_outputs).most_common(1)[0]
    supporting = [output for output in voting_outputs if output.stance == majority_stance]
    average_confidence = round(sum(o.confidence for o in supporting) / len(supporting))
    dissenting_view, dissenting_view_note = _build_dissenting_view(voting_outputs, majority_stance)

    return SynthesisMemo(
        recommendation=majority_stance,
        confidence=average_confidence,
        supporting_agents=sorted(o.agent_id for o in supporting),
        dissenting_agents=sorted(entry.agent_id for entry in dissenting_view),
        dissent_appendix=None,
        reasoning_trace_refs=[
            f"round{output.round}:{output.agent_id}" for output in final_round_outputs
        ],
        agent_summaries={
            output.agent_id: output.executive_summary
            for output in final_round_outputs
            if output.executive_summary
        },
        dissenting_view=dissenting_view,
        dissenting_view_note=dissenting_view_note,
        echoed_agents=echoed_agents,
    )
