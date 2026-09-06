"""ConflictResolutionStrategy: the swappable extension point for handling a
final-round DisagreementRecord (CLAUDE.md §5 steps 5-6).

Adding a 4th strategy is one new module implementing this protocol + one
`@register(...)` line (registry.py) — synthesizer.py and orchestrator.py never
branch on which strategy is configured.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Protocol

from committee.models.agent_output import AgentOutput
from committee.models.synthesis import DisagreementRecord, SynthesisMemo

SpawnAgentFn = Callable[[list[AgentOutput], list[str]], Awaitable[AgentOutput]]


class ConflictResolutionStrategy(Protocol):
    strategy_id: str

    async def resolve(
        self,
        disagreement: DisagreementRecord,
        final_round_outputs: list[AgentOutput],
        remaining_budget: int,
        spawn_agent_fn: SpawnAgentFn | None = None,
    ) -> tuple[SynthesisMemo, DisagreementRecord]:
        """Returns (synthesis_memo, updated_disagreement_record) — the record
        comes back with `resolution_strategy_applied` and `resolved` set.

        `remaining_budget` and `spawn_agent_fn` only matter to tie_breaker;
        flag_unresolved and confidence_weighted ignore both.
        """
        ...
