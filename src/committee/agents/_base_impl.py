"""Shared concrete implementation backing every lens agent.

Not part of the public AnalystAgent protocol surface — it's an internal base
class so each concrete agent (fundamentals.py, market_sentiment.py, ...) only
has to supply its agent_id, lens_name, and system prompt, not re-implement the
LLM call plumbing.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from committee.agents.prompts.shared import build_user_prompt
from committee.models.agent_output import AgentOutput, Rebuttal, Stance
from committee.models.requests import ThesisRequest
from committee.orchestration.budget_gate import BudgetGate


class _LLMAgentOutputSchema(BaseModel):
    """Schema the LLM is asked to fill via tool-calling. Deliberately excludes
    agent_id/round/tokens_used — those are known to the orchestrator, not the
    model, and are attached after the validated call returns."""

    stance: Stance
    confidence: int = Field(ge=0, le=100)
    key_factors: list[str] = Field(min_length=1, max_length=5)
    # Concrete cited facts/data points supporting this stance — not just the
    # key_factors tag labels. Required (min_length=1) so every agent output
    # carries something the convergence classifier can compare against prior
    # rounds; an agent with literally nothing new or old to cite as evidence
    # isn't making an argument the classifier can evaluate.
    evidence: list[str] = Field(min_length=1, max_length=8)
    top_risk: str
    # See AgentOutput.executive_summary — requested in this same tool call so
    # producing it never costs a second LLM round trip.
    executive_summary: str = Field(min_length=1, max_length=280)
    rebuttals: list[Rebuttal] | None = None


class BaseAnalystAgent:
    agent_id: str
    lens_name: str
    system_prompt: str

    def __init__(
        self,
        budget_gate: BudgetGate,
        agent_id: str | None = None,
        lens_name: str | None = None,
        system_prompt: str | None = None,
    ):
        # No LLMClient reference held here at all — BudgetGate is the only
        # object this agent can reach the LLM through (see budget_gate.py).
        self._budget_gate = budget_gate
        # The four standing agents set agent_id/lens_name/system_prompt as
        # class attributes via subclassing and never pass these kwargs, so
        # the `or` falls back to the class attribute unchanged for them.
        # DynamicAnalystAgent (agents/dynamic.py) is the only caller that
        # passes these explicitly, since it has no subclass of its own —
        # it's built straight from a persona record at debate-start time.
        self.agent_id = agent_id or self.agent_id
        self.lens_name = lens_name or self.lens_name
        self.system_prompt = system_prompt or self.system_prompt

    async def analyze(
        self,
        request: ThesisRequest,
        round: int,
        token_budget: int,
        prior_round_outputs: list[AgentOutput] | None,
        directive: str | None = None,
    ) -> AgentOutput:
        user_prompt = build_user_prompt(
            request=request, round=round, prior_round_outputs=prior_round_outputs, directive=directive
        )
        result, tokens_used = await self._budget_gate.call(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response_model=_LLMAgentOutputSchema,
            max_tokens=token_budget,
        )
        return AgentOutput(
            agent_id=self.agent_id,
            agent_name=self.lens_name,
            round=round,
            stance=result.stance,
            confidence=result.confidence,
            key_factors=result.key_factors,
            evidence=result.evidence,
            top_risk=result.top_risk,
            executive_summary=result.executive_summary,
            rebuttals=result.rebuttals,
            tokens_used=tokens_used,
        )
