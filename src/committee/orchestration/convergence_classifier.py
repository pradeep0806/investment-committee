"""Evidence-aware convergence classification.

Two agents reaching the same stance is not, by itself, trustworthy
convergence — an agent that just restates another agent's (or its own
prior-round) conclusion with no new supporting evidence looks identical to
genuine independent corroboration under stance/factor-tag comparison alone.
This module tells the two apart by comparing each current-round output's
`evidence` against everything already said in prior rounds, so
`ExploreExploitController` can exclude echoes from the convergence signal
that drives the explore->exploit switch (see explore_exploit.py).

Classification, per agent, per round:
  - NONE:    no prior-round output shares this agent's stance to compare
             against (e.g. round 1, or this agent just changed its mind).
  - GENUINE: same stance as some prior-round output, and this round's
             evidence contains at least one item not already stated by any
             prior-round output (its own or another agent's).
  - ECHO:    same stance as some prior-round output, and every evidence item
             this round is a normalized subset of what was already stated —
             nothing new backs the repeated conclusion.
"""

from __future__ import annotations

from committee.models.agent_output import AgentOutput
from committee.models.synthesis import ConvergenceType


def _normalize(items: list[str]) -> set[str]:
    return {item.strip().lower() for item in items if item.strip()}


def classify_round(
    current_round_outputs: list[AgentOutput],
    all_prior_outputs: list[AgentOutput],
) -> dict[str, ConvergenceType]:
    """Returns agent_id -> ConvergenceType for `current_round_outputs`.

    `all_prior_outputs` is every AgentOutput from every round strictly
    before the current one (own and others') — not just the immediately
    preceding round — so an echo of something said two rounds ago is still
    caught, not just an echo of the last round.
    """
    if not all_prior_outputs:
        return {output.agent_id: ConvergenceType.NONE for output in current_round_outputs}

    # Evidence already on the table, per stance — an agent's claim only
    # needs to be checked against prior arguments that reached the *same*
    # conclusion; evidence backing a different stance doesn't make this
    # round's repetition of its own conclusion any less of an echo.
    prior_evidence_by_stance: dict[str, set[str]] = {}
    for prior in all_prior_outputs:
        bucket = prior_evidence_by_stance.setdefault(prior.stance.value, set())
        bucket |= _normalize(prior.evidence)

    results: dict[str, ConvergenceType] = {}
    for output in current_round_outputs:
        prior_evidence = prior_evidence_by_stance.get(output.stance.value)
        if not prior_evidence:
            results[output.agent_id] = ConvergenceType.NONE
            continue

        current_evidence = _normalize(output.evidence)
        has_new_evidence = bool(current_evidence - prior_evidence)
        results[output.agent_id] = (
            ConvergenceType.GENUINE if has_new_evidence else ConvergenceType.ECHO
        )

    return results
