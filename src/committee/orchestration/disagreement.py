"""Disagreement detection: never averages or blends opposing stances into a
single number (CLAUDE.md §5 step 4). Flags explicitly instead, for the active
ConflictResolutionStrategy to handle.

Computed every round (feeds the live `disagreements_detected_total` metric —
CLAUDE.md's wording supports a per-round read for observability purposes),
but only the *final* round's result is what the orchestrator feeds into
synthesis/conflict resolution — resolved by the plan's explicit sign-off.

Excluded agents (failed structured-output validation after all retries, per
step 2's resolution) never reach this function's `agent_outputs` list at all
— they're filtered out by the orchestrator before disagreement detection
runs, so they never count toward the majority-fraction denominator.
"""

from __future__ import annotations

from collections import Counter

from committee.models.agent_output import AgentOutput, Stance
from committee.models.synthesis import DisagreementRecord

DEFAULT_MAJORITY_THRESHOLD = 0.75
DEFAULT_HIGH_CONFIDENCE_THRESHOLD = 70


def detect(
    round: int,
    agent_outputs: list[AgentOutput],
    majority_threshold: float = DEFAULT_MAJORITY_THRESHOLD,
    high_confidence_threshold: int = DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
) -> list[DisagreementRecord]:
    if not agent_outputs:
        return []

    records: list[DisagreementRecord] = []

    stance_counts = Counter(output.stance for output in agent_outputs)
    majority_stance, majority_count = stance_counts.most_common(1)[0]
    has_majority = (majority_count / len(agent_outputs)) >= majority_threshold

    high_confidence_conflict = _find_high_confidence_buy_sell_conflict(
        agent_outputs, high_confidence_threshold
    )

    if not has_majority or high_confidence_conflict:
        involved = high_confidence_conflict if high_confidence_conflict else agent_outputs
        opposing_stances = {output.agent_id: output.stance for output in involved}
        contested_factors = _contested_factors(involved)
        records.append(
            DisagreementRecord(
                round_detected=round,
                agents_involved=[output.agent_id for output in involved],
                opposing_stances=opposing_stances,
                contested_factors=contested_factors,
            )
        )

    return records


def _find_high_confidence_buy_sell_conflict(
    agent_outputs: list[AgentOutput], high_confidence_threshold: int
) -> list[AgentOutput] | None:
    """Any Buy-vs-Sell pair where both sides hold confidence >= threshold is a
    disagreement regardless of whether a majority otherwise exists — a
    confident direct conflict is never something to average away."""
    buys = [o for o in agent_outputs if o.stance == Stance.BUY and o.confidence >= high_confidence_threshold]
    sells = [o for o in agent_outputs if o.stance == Stance.SELL and o.confidence >= high_confidence_threshold]
    if buys and sells:
        return buys + sells
    return None


def _contested_factors(agent_outputs: list[AgentOutput]) -> list[str]:
    """Factors cited by more than one agent among the disagreeing set — these
    are the specific points of contention, not each agent's full factor list.

    Normalized (.strip().lower()) before counting, matching
    explore_exploit.py's _factor_overlap() exactly — otherwise two agents
    citing the same factor with different casing/whitespace (e.g.
    "Valuation" vs "valuation ") never register as contested here even
    though they'd count as overlapping for the convergence score. The
    first original-cased spelling seen for each matched factor is what's
    returned, not the lowercased key, so contested_factors reads naturally
    in the synthesis memo."""
    first_seen_spelling: dict[str, str] = {}
    normalized_counts: Counter[str] = Counter()

    for output in agent_outputs:
        for factor in output.key_factors:
            normalized = factor.strip().lower()
            normalized_counts[normalized] += 1
            first_seen_spelling.setdefault(normalized, factor)

    return sorted(
        (first_seen_spelling[normalized] for normalized, count in normalized_counts.items() if count > 1)
    )
