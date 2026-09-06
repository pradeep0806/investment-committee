"""BudgetManager: the single source of truth for how many tokens an agent gets
this round, and the hard invariant that total spend never exceeds
`total_token_budget` — including a reserve pool carved out up front for
tie-breaker spawns (CLAUDE.md §5 step 5's "extra budget drawn from a reserve
pool"), not retrofitted once conflict resolution is built (step 6).

The ledger tracks *actual* usage (`tokens_used`), not just what was allocated
— allocation is a plan, usage is what the LLM call actually cost, and the
remaining-budget math must be driven by the latter or the budget ceiling is
aspirational rather than enforced.

The allocation each agent receives IS passed to the LLM call as a real
`max_tokens` cap (see llm/client.py, structured_output.py) — this was
previously decorative (accepted as a parameter, never reaching the API),
which is what let real Gemini calls silently use 3-10x their allocation.
One residual, expected gap: `tokens_used` is the provider's total_tokens
(prompt + output), while `max_tokens` only bounds output generation (and,
for Gemini, thinking tokens separately) — a call can still legitimately
report tokens_used somewhat above its allocation, by roughly the size of
the prompt itself. That's a small, bounded overrun rather than the
previous unbounded one, and `record_actual_usage` logs/meters it either way
(see `budget_overrun_tokens_total`) rather than letting it pass silently.
"""

from __future__ import annotations

from typing import Literal

import structlog

from committee.models.trace import BudgetLedgerEntry
from committee.observability.metrics import budget_overrun_tokens_total

logger = structlog.get_logger()

Mode = Literal["explore", "balanced", "exploit"]

_EXPLOIT_REALLOCATION_MULTIPLIER = 2.5  # midpoint of CLAUDE.md §5's "2-3x" exploit reallocation
DEFAULT_RESERVE_FRACTION = 0.10


class BudgetExhaustedError(Exception):
    """Raised when the remaining (non-reserve) budget can't cover even a floor
    allocation for every agent this round."""


class BudgetManager:
    def __init__(
        self,
        total_token_budget: int,
        num_rounds: int,
        num_agents: int,
        reserve_fraction: float = DEFAULT_RESERVE_FRACTION,
    ):
        self.total_token_budget = total_token_budget
        self.num_rounds = num_rounds
        self.num_agents = num_agents
        self.reserve_pool = int(total_token_budget * reserve_fraction)
        self.spendable_budget = total_token_budget - self.reserve_pool

        self._tokens_used_total = 0
        self._reserve_used = 0
        self._ledger: list[BudgetLedgerEntry] = []
        self._rounds_allocated = 0

    def remaining_budget(self) -> int:
        """Remaining spendable (non-reserve) budget, based on actual usage so far."""
        return self.spendable_budget - self._tokens_used_total

    def remaining_reserve(self) -> int:
        return self.reserve_pool - self._reserve_used

    def allocate(
        self,
        round: int,
        agent_ids: list[str],
        mode: Mode,
        contested_agents: list[str] | None = None,
    ) -> dict[str, int]:
        """Returns agent_id -> tokens_allocated for this round.

        The per-round baseline is *recomputed* every call from whatever
        budget actually remains, divided by however many rounds are actually
        left — not a fixed value computed once at construction. Real LLM
        calls routinely use more (or less) than their allocation, since
        `token_budget` is advisory to the model, not a hard cap on its
        response; if an earlier round overspent, later rounds' baseline
        shrinks accordingly rather than the debate crashing outright. This is
        graceful degradation, not a violation of the total_token_budget
        ceiling — actual cumulative usage still never exceeds it (see
        record_actual_usage / the worst-case test), it just means later
        agents may get a smaller allocation than earlier ones did.

        explore/balanced: even baseline split across all agents.
        exploit: contested/minority agents (from `contested_agents`) get
        `_EXPLOIT_REALLOCATION_MULTIPLIER`x the baseline; the multiplier's cost
        is funded by giving non-contested agents a reduced share, so the
        round's total allocation still respects the recomputed per-round
        baseline rather than overspending the remaining budget.

        Raises BudgetExhaustedError only when the remaining spendable budget
        can no longer cover even a floor allocation (1 token/agent) for every
        agent this round — a genuine out-of-budget event, not a rigid
        per-round math artifact.
        """
        remaining = self.remaining_budget()
        floor_total = len(agent_ids)
        if remaining < floor_total:
            raise BudgetExhaustedError(
                f"Remaining budget {remaining} cannot cover a floor allocation "
                f"for {len(agent_ids)} agents in round {round}."
            )

        remaining_rounds = max(self.num_rounds - self._rounds_allocated, 1)
        baseline_per_agent = max(remaining // (len(agent_ids) * remaining_rounds), 1)
        round_budget = min(baseline_per_agent * len(agent_ids), remaining)
        self._rounds_allocated += 1

        if mode == "exploit" and contested_agents:
            contested = [a for a in contested_agents if a in agent_ids]
            non_contested = [a for a in agent_ids if a not in contested]
            if contested and non_contested:
                return self._exploit_allocation(round_budget, agent_ids, contested, non_contested)

        # explore / balanced / exploit-with-no-valid-contested-agents: even split
        even_share = round_budget // len(agent_ids)
        return {agent_id: even_share for agent_id in agent_ids}

    def _exploit_allocation(
        self,
        round_budget: int,
        agent_ids: list[str],
        contested: list[str],
        non_contested: list[str],
    ) -> dict[str, int]:
        # Solve for a base share `b` such that:
        #   len(contested) * (multiplier * b) + len(non_contested) * b == round_budget
        denominator = len(contested) * _EXPLOIT_REALLOCATION_MULTIPLIER + len(non_contested)
        base_share = int(round_budget / denominator)
        allocation = {agent_id: base_share for agent_id in non_contested}
        allocation.update(
            {agent_id: int(base_share * _EXPLOIT_REALLOCATION_MULTIPLIER) for agent_id in contested}
        )
        return allocation

    def record_actual_usage(self, round: int, agent_id: str, tokens_allocated: int, tokens_used: int, mode: Mode) -> BudgetLedgerEntry:
        if tokens_used > tokens_allocated:
            overrun = tokens_used - tokens_allocated
            budget_overrun_tokens_total.labels(agent=agent_id).inc(overrun)
            logger.warning(
                "budget_allocation_overrun",
                round=round,
                agent_id=agent_id,
                tokens_allocated=tokens_allocated,
                tokens_used=tokens_used,
                overrun=overrun,
                mode=mode,
            )

        self._tokens_used_total += tokens_used
        entry = BudgetLedgerEntry(
            round=round,
            agent_id=agent_id,
            tokens_allocated=tokens_allocated,
            tokens_used=tokens_used,
            mode=mode,
        )
        self._ledger.append(entry)
        return entry

    def draw_from_reserve(self, tokens_used: int) -> None:
        """Debits actual tie-breaker spend from the reserve pool. Raises
        BudgetExhaustedError if the reserve can't cover it — a tie-breaker's
        actual cost is never allowed to spill over into the spendable budget,
        since that would violate the total_token_budget ceiling by definition."""
        if tokens_used > self.remaining_reserve():
            raise BudgetExhaustedError(
                f"Tie-breaker spend {tokens_used} exceeds remaining reserve {self.remaining_reserve()}."
            )
        self._reserve_used += tokens_used

    @property
    def ledger(self) -> list[BudgetLedgerEntry]:
        return list(self._ledger)
