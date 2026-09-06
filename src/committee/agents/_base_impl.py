"""Shared concrete implementation backing every lens agent.

Not part of the public AnalystAgent protocol surface — it's an internal base
class so each concrete agent (fundamentals.py, market_sentiment.py, ...) only
has to supply its agent_id, lens_name, and system prompt, not re-implement the
LLM call plumbing.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from committee.agents.prompts.shared import build_user_prompt
from committee.llm.client import LLMClient
from committee.models.agent_output import AgentOutput, Rebuttal, Stance
from committee.models.requests import ThesisRequest


class _LLMAgentOutputSchema(BaseModel):
    """Schema the LLM is asked to fill via tool-calling. Deliberately excludes
    agent_id/round/tokens_used — those are known to the orchestrator, not the
    model, and are attached after the validated call returns."""

    stance: Stance
    confidence: int = Field(ge=0, le=100)
    key_factors: list[str] = Field(min_length=1, max_length=5)
    top_risk: str
    rebuttals: list[Rebuttal] | None = None


class BaseAnalystAgent:
    agent_id: str
    lens_name: str
    system_prompt: str

    def __init__(self, llm_client: LLMClient):
        self._llm_client = llm_client

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
        result, tokens_used = await self._llm_client.call(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response_model=_LLMAgentOutputSchema,
            max_tokens=token_budget,
        )
        return AgentOutput(
            agent_id=self.agent_id,
            round=round,
            stance=result.stance,
            confidence=result.confidence,
            key_factors=result.key_factors,
            top_risk=result.top_risk,
            rebuttals=result.rebuttals,
            tokens_used=tokens_used,
        )
