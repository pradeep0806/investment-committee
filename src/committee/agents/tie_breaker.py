"""TieBreakerAgent: spawned on-demand by the tie_breaker conflict resolution
strategy, not a standing committee member — deliberately NOT registered in
agents/registry.py's default four lenses.

Its context is bounded to the opposing AgentOutputs + contested factors only
(not the full multi-round transcript), per the resolved design decision in
conflict_resolution/tie_breaker.py's docstring.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from committee.agents.prompts.tie_breaker import SYSTEM_PROMPT
from committee.models.agent_output import AgentOutput, Stance
from committee.orchestration.budget_gate import BudgetGate


class _TieBreakerOutputSchema(BaseModel):
    stance: Stance
    confidence: int = Field(ge=0, le=100)
    key_factors: list[str] = Field(min_length=1, max_length=5)
    top_risk: str


class TieBreakerAgent:
    agent_id = "tie_breaker"
    lens_name = "Tie-Breaker"

    def __init__(self, budget_gate: BudgetGate):
        # Same rule as BaseAnalystAgent: no raw LLMClient reference held
        # here — BudgetGate is the only reachable path to the LLM.
        self._budget_gate = budget_gate

    async def resolve(
        self,
        opposing_outputs: list[AgentOutput],
        contested_factors: list[str],
        max_tokens: int,
    ) -> AgentOutput:
        """`max_tokens` caps the tie-breaker's own call the same way a
        regular agent's token_budget does — the natural cap here is
        whatever's left in BudgetManager's reserve pool, since the
        tie-breaker's spend is drawn from exactly that (see
        orchestrator.py's spawn_agent_fn, which passes
        budget_manager.remaining_reserve()). Required (not optional) since
        BudgetGate.call() refuses a None max_tokens — the gate cannot be
        opted out of."""
        user_prompt = self._build_prompt(opposing_outputs, contested_factors)
        result, tokens_used = await self._budget_gate.call(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=_TieBreakerOutputSchema,
            max_tokens=max_tokens,
        )
        return AgentOutput(
            agent_id=self.agent_id,
            round=max((o.round for o in opposing_outputs), default=1),
            stance=result.stance,
            confidence=result.confidence,
            key_factors=result.key_factors,
            top_risk=result.top_risk,
            tokens_used=tokens_used,
        )

    @staticmethod
    def _build_prompt(opposing_outputs: list[AgentOutput], contested_factors: list[str]) -> str:
        lines = ["Opposing positions to resolve:"]
        for output in opposing_outputs:
            lines.append(
                f"- [{output.agent_id}] stance={output.stance.value} "
                f"confidence={output.confidence} key_factors={output.key_factors} "
                f"top_risk={output.top_risk!r}"
            )
        if contested_factors:
            lines.append(f"Specifically contested factors: {contested_factors}")
        lines.append(
            "Respond only through the provided tool, with your own stance "
            "(Buy/Hold/Sell/Pass), confidence 0-100, 2-5 key_factors, and your top_risk."
        )
        return "\n".join(lines)
