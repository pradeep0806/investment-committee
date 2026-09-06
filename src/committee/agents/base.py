"""The AnalystAgent protocol: the single extension point for adding a new lens.

A new agent lens = one new module implementing this protocol + one
`@register("...")` line in its module (see registry.py). Nothing in
orchestrator.py or elsewhere needs to change.
"""

from __future__ import annotations

from typing import Protocol

from committee.models.agent_output import AgentOutput
from committee.models.requests import ThesisRequest


class AnalystAgent(Protocol):
    agent_id: str
    lens_name: str

    async def analyze(
        self,
        request: ThesisRequest,
        round: int,
        token_budget: int,
        prior_round_outputs: list[AgentOutput] | None,
        directive: str | None = None,
    ) -> AgentOutput:
        """Produce this agent's structured view for the given round.

        `prior_round_outputs` are all agents' outputs from the previous round
        (None in round 1), enabling rebuttals. `directive` is an optional
        orchestrator-injected instruction (e.g. explore mode's rotating
        "argue against the majority" prompt).
        """
        ...
