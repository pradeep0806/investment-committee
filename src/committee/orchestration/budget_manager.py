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
the prompt itself. `record_actual_usage` logs/meters any such overrun
either way (see `budget_overrun_tokens_total`) rather than letting it pass
silently, escalating to error-level logging if it exceeds
`RETRY_BUDGET_MULTIPLIER`x the allocation — a real bug found via a live
debate against a weak local model (qwen3.5:9b via Ollama) showed this gap
was not actually small/bounded before structured_output.py's retry loop
was fixed to cap cumulative spend across all attempts, not just each
attempt individually: a model needing several retries to produce a valid
tool call could spend roughly `max_retries`x its allocation with nothing
capping the running total, observed live at ~3.8x (7044 tokens_used against
1853 tokens_allocated). See structured_output.py's RETRY_BUDGET_MULTIPLIER
for the actual fix.
"""

from __future__ import annotations

from typing import Literal

import structlog

from committee.llm.structured_output import RETRY_BUDGET_MULTIPLIER
from committee.models.trace import BudgetLedgerEntry
from committee.observability.metrics import budget_overrun_tokens_total

logger = structlog.get_logger()

Mode = Literal["explore", "balanced", "exploit"]

_EXPLOIT_REALLOCATION_MULTIPLIER = 2.5  # midpoint of CLAUDE.md §5's "2-3x" exploit reallocation
DEFAULT_RESERVE_FRACTION = 0.10

# Real bug found via a live debate against qwen3.5:9b on Ollama: allocate()'s
# only floor was 1 token/agent — far below what any real model needs to
# produce a valid structured tool call, and completely disconnected from
# structured_output.py's own MIN_MAX_TOKENS/RETRY_BUDGET_MULTIPLIER
# constants. By a later round, exploit-mode reallocation plus a shrinking
# remaining budget drove non-contested agents down to allocations as low as
# 723 tokens — above MIN_MAX_TOKENS so call_structured's own floor clamp
# never engaged, but still not enough for this model to reliably produce a
# valid tool call within RETRY_BUDGET_MULTIPLIER's retry-budget ceiling, so
# those agents were correctly excluded rather than allowed to overrun — but
# they were set up to fail from the allocation itself, not a fluke.
#
# What "viable" actually requires is provider/model-dependent, confirmed by
# re-verifying against the exact live trace that exposed this: every
# *successful* call in that debate (qwen3.5:9b) used between 1763 and 3779
# tokens, while this same schema against a stronger hosted model routinely
# succeeds well under 1000. A single hardcoded module-level constant can't
# be right for every provider — DEFAULT_MIN_VIABLE_ALLOCATION here is
# therefore a modest, provider-agnostic default (matches
# structured_output.py's own MIN_MAX_TOKENS — enough for that floor clamp
# to have room, without being so conservative it blocks a normal debate
# against a capable model), and BudgetManager takes the real value as a
# constructor parameter (Settings.min_viable_allocation_per_agent /
# DebateConfig.min_viable_allocation_per_agent, same per-debate-override
# pattern as the LLM provider/model fields) so a caller running against a
# weaker/local model can raise it to match what they've actually observed
# that model needs.
DEFAULT_MIN_VIABLE_ALLOCATION = 256


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
        min_viable_allocation: int = DEFAULT_MIN_VIABLE_ALLOCATION,
    ):
        self.total_token_budget = total_token_budget
        self.num_rounds = num_rounds
        self.num_agents = num_agents
        self.reserve_pool = int(total_token_budget * reserve_fraction)
        self.spendable_budget = total_token_budget - self.reserve_pool
        # See DEFAULT_MIN_VIABLE_ALLOCATION's module-level docstring for why
        # this is a constructor parameter rather than a fixed constant —
        # what's "viable" per agent per round depends on the provider/model
        # actually in use, not something this class can know on its own.
        self.min_viable_allocation = min_viable_allocation

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

        Raises BudgetExhaustedError when the remaining spendable budget can
        no longer cover a *viable* allocation (self.min_viable_allocation
        tokens) for every agent this round — a genuine out-of-budget event,
        not a rigid per-round math artifact. This floor is real, not the
        historical "1 token/agent" placeholder: a real bug found via a live
        debate showed that an allocation above 1 but below what a given
        provider/model actually needs sets an agent up to be excluded from
        the round, not to succeed on a smaller budget — better to stop the
        debate cleanly with fewer, valid rounds than to run agents that are
        allocated to fail (see DEFAULT_MIN_VIABLE_ALLOCATION's module-level
        docstring above for why this is configurable per debate).
        """
        remaining = self.remaining_budget()
        floor_total = self.min_viable_allocation * len(agent_ids)
        if remaining < floor_total:
            raise BudgetExhaustedError(
                f"Remaining budget {remaining} cannot cover a viable allocation "
                f"({self.min_viable_allocation} tokens/agent) for {len(agent_ids)} agents "
                f"in round {round}."
            )

        remaining_rounds = max(self.num_rounds - self._rounds_allocated, 1)
        baseline_per_agent = max(
            remaining // (len(agent_ids) * remaining_rounds), self.min_viable_allocation
        )
        # The floor can only ever push this round's total *up* relative to
        # the naive even split, never down — already guarded by the
        # floor_total check above, which confirms `remaining` can afford
        # self.min_viable_allocation for every agent even in the worst case
        # (remaining_rounds == 1). min(..., remaining) stays as the final
        # safety clamp so a multi-round baseline that's still below the
        # floor once remaining_rounds > 1 doesn't accidentally spend more
        # than a single round actually has available.
        round_budget = min(max(baseline_per_agent * len(agent_ids), floor_total), remaining)
        self._rounds_allocated += 1

        if mode == "exploit" and contested_agents:
            contested = [a for a in contested_agents if a in agent_ids]
            non_contested = [a for a in agent_ids if a not in contested]
            if contested and non_contested:
                return self._exploit_allocation(
                    round_budget, agent_ids, contested, non_contested
                )

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

        if base_share < self.min_viable_allocation:
            # Real bug found via a live debate: the 2.5x weighting toward
            # contested agents can push non-contested agents' *individual*
            # share below self.min_viable_allocation even when the round's
            # total budget (guaranteed >= min_viable_allocation *
            # len(agent_ids) by allocate()'s own floor check) would support
            # an even split — the reallocation itself, not the total, was
            # what starved them. Every agent gets at least the floor; the
            # 2.5x multiplier scales down (never below 1x, i.e. never below
            # what a non-contested agent gets) to whatever's actually
            # affordable with the floor already funded for everyone, so
            # contested agents still get preferential budget without agents
            # being allocated an amount they're structurally unable to
            # succeed on.
            floor_total = self.min_viable_allocation * len(agent_ids)
            surplus = max(round_budget - floor_total, 0)
            # Surplus is split among contested agents only (still
            # preferential treatment), on top of everyone's floor.
            bonus_per_contested = surplus // len(contested) if contested else 0
            allocation = {agent_id: self.min_viable_allocation for agent_id in non_contested}
            allocation.update(
                {
                    agent_id: self.min_viable_allocation + bonus_per_contested
                    for agent_id in contested
                }
            )
            return allocation

        allocation = {agent_id: base_share for agent_id in non_contested}
        allocation.update(
            {agent_id: int(base_share * _EXPLOIT_REALLOCATION_MULTIPLIER) for agent_id in contested}
        )
        return allocation

    def record_actual_usage(
        self,
        round: int,
        agent_id: str,
        tokens_allocated: int,
        tokens_used: int,
        mode: Mode,
        provider_used: str = "",
        excluded: bool = False,
    ) -> BudgetLedgerEntry:
        if tokens_used > tokens_allocated:
            overrun = tokens_used - tokens_allocated
            budget_overrun_tokens_total.labels(agent=agent_id).inc(overrun)
            # Severity re-scoped after fixing structured_output.py's retry-
            # accumulation gap (RETRY_BUDGET_MULTIPLIER): cumulative spend
            # across every attempt is now bounded at ~2x the allocation, so
            # any overrun *larger* than that ceiling can no longer be
            # explained by a normal prompt-size asymmetry or a bounded
            # retry sequence — it means the retry-budget cap itself didn't
            # do its job, which is a real anomaly worth escalating rather
            # than logging at the same level as an expected, small gap.
            log_method = (
                logger.error if overrun > tokens_allocated * RETRY_BUDGET_MULTIPLIER else logger.warning
            )
            log_method(
                "budget_allocation_overrun",
                round=round,
                agent_id=agent_id,
                tokens_allocated=tokens_allocated,
                tokens_used=tokens_used,
                overrun=overrun,
                mode=mode,
                excluded=excluded,
            )

        self._tokens_used_total += tokens_used
        entry = BudgetLedgerEntry(
            round=round,
            agent_id=agent_id,
            tokens_allocated=tokens_allocated,
            tokens_used=tokens_used,
            mode=mode,
            provider_used=provider_used,
            excluded=excluded,
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
