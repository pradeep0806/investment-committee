from typing import Literal

from pydantic import BaseModel, Field

from committee.models.agent_output import Stance


class ConvergenceSignal(BaseModel):
    round: int = Field(ge=1)
    stance_agreement: float = Field(ge=0, le=1)
    factor_overlap: float = Field(ge=0, le=1)
    confidence_spread: float = Field(ge=0, le=1)
    composite_score: float = Field(ge=0, le=1)
    mode_selected: Literal["explore", "balanced", "exploit"]


class DisagreementRecord(BaseModel):
    round_detected: int = Field(ge=1)
    agents_involved: list[str]
    opposing_stances: dict[str, Stance]
    contested_factors: list[str]
    resolution_strategy_applied: str | None = None
    resolved: bool = False


class SynthesisMemo(BaseModel):
    recommendation: Stance
    confidence: int = Field(ge=0, le=100)
    supporting_agents: list[str]
    dissenting_agents: list[str]
    dissent_appendix: str | None = None
    reasoning_trace_refs: list[str]
