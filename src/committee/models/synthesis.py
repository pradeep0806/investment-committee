from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from committee.models.agent_output import Stance


class ConvergenceType(str, Enum):
    """Per-agent classification of *why* this round's output agrees with a
    prior one — see orchestration/convergence_classifier.py. Only GENUINE
    convergence should count toward the explore->exploit trigger; an ECHO
    (same conclusion, no evidence beyond what was already on the table) is
    not trustworthy convergence, even though it looks identical to genuine
    convergence under stance/factor-tag comparison alone."""

    GENUINE = "genuine"
    ECHO = "echo"
    NONE = "none"


class ConvergenceSignal(BaseModel):
    round: int = Field(ge=1)
    stance_agreement: float = Field(ge=0, le=1)
    factor_overlap: float = Field(ge=0, le=1)
    confidence_spread: float = Field(ge=0, le=1)
    composite_score: float = Field(ge=0, le=1)
    mode_selected: Literal["explore", "balanced", "exploit"]
    # agent_id -> ConvergenceType for this round, additive field. Empty dict
    # for round 1 (nothing prior to compare against) or any caller that
    # doesn't pass classification into score().
    convergence_types: dict[str, ConvergenceType] = Field(default_factory=dict)


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
